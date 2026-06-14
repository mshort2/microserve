import pytest
import torch

from microserve.config import ModelConfig
from microserve.engine import Engine, LLMEngine, PagedLLMEngine
from microserve.kv_cache import FlatKVCache
from microserve.model import QwenForCausalLM, load_weights


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase 1 model tests require CUDA"
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The mitochondrion is the",
    "import numpy as np\n\ndef softmax(x):",
]


def test_weights_load():
    cfg = ModelConfig.qwen2_5_0_5b()
    weights = load_weights(MODEL_ID, dtype=torch.bfloat16, device="cuda")

    assert weights["embed.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    assert weights["norm.weight"].shape == (cfg.hidden_size,)

    q_dim = cfg.num_q_heads * cfg.head_dim
    kv_dim = cfg.num_kv_heads * cfg.head_dim
    for i in range(cfg.num_layers):
        assert weights[f"layers.{i}.attn.q_proj.weight"].shape == (q_dim, cfg.hidden_size)
        assert weights[f"layers.{i}.attn.q_proj.bias"].shape == (q_dim,)
        assert weights[f"layers.{i}.attn.k_proj.weight"].shape == (kv_dim, cfg.hidden_size)
        assert weights[f"layers.{i}.attn.k_proj.bias"].shape == (kv_dim,)
        assert weights[f"layers.{i}.attn.v_proj.weight"].shape == (kv_dim, cfg.hidden_size)
        assert weights[f"layers.{i}.attn.v_proj.bias"].shape == (kv_dim,)
        assert weights[f"layers.{i}.attn.o_proj.weight"].shape == (cfg.hidden_size, q_dim)
        assert f"layers.{i}.attn.o_proj.bias" not in weights
        assert weights[f"layers.{i}.mlp.gate_proj.weight"].shape == (
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        assert weights[f"layers.{i}.mlp.up_proj.weight"].shape == (
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        assert weights[f"layers.{i}.mlp.down_proj.weight"].shape == (
            cfg.hidden_size,
            cfg.intermediate_size,
        )
        assert weights[f"layers.{i}.input_norm.weight"].shape == (cfg.hidden_size,)
        assert weights[f"layers.{i}.post_attn_norm.weight"].shape == (cfg.hidden_size,)

    assert "lm_head.weight" not in weights


def test_logits_match_hf_bf16():
    """Smoke check: bf16 logits agree with HF within sdpa kernel-selection noise."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    ours = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.bfloat16)

    for prompt in PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            hf_logits = hf(input_ids).logits
            our_logits = ours(input_ids)

        assert hf_logits.shape == our_logits.shape
        max_abs_diff = (hf_logits - our_logits).abs().max().item()
        assert max_abs_diff < 2.5, (
            f"prompt={prompt!r} max_abs_diff={max_abs_diff} too large for bf16 noise"
        )


def test_logits_match_hf_fp32():
    """fp32 logits agree with HuggingFace within 1e-3 on 5 prompts.

    Tighter than the bf16 smoke check above — in fp32 the sdpa kernel-selection
    noise drops below 1e-4, so two correct implementations agree to ~1e-3.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    ours = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    for prompt in PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            hf_logits = hf(input_ids).logits
            our_logits = ours(input_ids)

        assert hf_logits.shape == our_logits.shape
        max_abs_diff = (hf_logits - our_logits).abs().max().item()
        assert max_abs_diff < 1e-3, (
            f"prompt={prompt!r} fp32 max_abs_diff={max_abs_diff} — real architecture bug"
        )


def test_hidden_states_match_hf_fp32():
    """Per-layer hidden-state diff < 1e-3 vs HuggingFace in fp32.

    The strongest architecture test: hooks every site in the residual stream
    (embed + 24 layers + final norm) and asserts each matches HF independently.
    Greedy-token matching is unreliable across implementations because tied
    argmax decisions flip under tiny numerical noise; hidden-state matching is
    the continuous, robust signal.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    ours = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    hf_states: dict[str, torch.Tensor] = {}
    our_states: dict[str, torch.Tensor] = {}

    def _grab(store: dict, name: str):
        def hook(module, inputs, output):
            store[name] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = []
    handles.append(hf.model.embed_tokens.register_forward_hook(_grab(hf_states, "embed")))
    handles.append(ours.embed.register_forward_hook(_grab(our_states, "embed")))
    for i in range(len(hf.model.layers)):
        handles.append(hf.model.layers[i].register_forward_hook(_grab(hf_states, f"layer_{i:02d}")))
        handles.append(ours.layers[i].register_forward_hook(_grab(our_states, f"layer_{i:02d}")))
    handles.append(hf.model.norm.register_forward_hook(_grab(hf_states, "final_norm")))
    handles.append(ours.norm.register_forward_hook(_grab(our_states, "final_norm")))

    try:
        input_ids = tok(PROMPTS[0], return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            hf(input_ids)
            ours(input_ids)

        sites = ["embed"] + [f"layer_{i:02d}" for i in range(len(hf.model.layers))] + ["final_norm"]
        for name in sites:
            h = hf_states[name].float()
            o = our_states[name].float()
            assert h.shape == o.shape, f"{name}: shape {h.shape} vs {o.shape}"
            diff = (h - o).abs().max().item()
            assert diff < 1e-3, (
                f"{name}: fp32 max_abs_diff={diff:.4e} — architecture bug in this component"
            )
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# KV cache + cached greedy decode
# ---------------------------------------------------------------------------


def _greedy_no_cache(model: QwenForCausalLM, input_ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Greedy decode via repeated full prefill — no KV cache."""
    output_ids = input_ids
    for _ in range(n_new):
        with torch.no_grad():
            logits = model(output_ids)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_tok], dim=-1)
    return output_ids


def test_kv_cache_write_read_roundtrip():
    """Pure unit test: write at two positions, read back, expect equality."""
    cfg = ModelConfig.qwen2_5_0_5b()
    cache = FlatKVCache(cfg, max_seq_len=16, dtype=torch.float32, device="cuda")

    k1 = torch.randn(1, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    v1 = torch.randn(1, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    cache.write(layer=3, start_pos=0, k=k1, v=v1)

    k2 = torch.randn(1, cfg.num_kv_heads, 1, cfg.head_dim, device="cuda")
    v2 = torch.randn(1, cfg.num_kv_heads, 1, cfg.head_dim, device="cuda")
    cache.write(layer=3, start_pos=5, k=k2, v=v2)

    k_read, v_read = cache.read(layer=3, end_pos=6)
    assert k_read.shape == (1, cfg.num_kv_heads, 6, cfg.head_dim)
    assert torch.equal(k_read[:, :, :5, :], k1)
    assert torch.equal(k_read[:, :, 5:6, :], k2)
    assert torch.equal(v_read[:, :, :5, :], v1)
    assert torch.equal(v_read[:, :, 5:6, :], v2)


def test_kv_cache_overflow_raises():
    cfg = ModelConfig.qwen2_5_0_5b()
    cache = FlatKVCache(cfg, max_seq_len=8, dtype=torch.float32, device="cuda")
    k = torch.zeros(1, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    v = torch.zeros(1, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    with pytest.raises(RuntimeError, match="overflow"):
        cache.write(layer=0, start_pos=5, k=k, v=v)


def test_greedy_cached_matches_no_cache_fp32():
    """Cache-based decode produces the same token IDs as no-cache repeated prefill (fp32).

    The no-cache path is validated against HuggingFace by the architecture
    tests above; matching against it here transitively validates the cache.
    A failure is necessarily a cache bug, not an architecture bug — the
    architecture path is unchanged from the no-cache reference.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)
    engine = Engine(model, tok, max_seq_len=512)

    n_new = 50
    for prompt in PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        no_cache_out = _greedy_no_cache(model, input_ids, n_new)
        cached_out = engine.generate_ids(input_ids, max_new_tokens=n_new)

        if torch.equal(no_cache_out, cached_out):
            continue

        if no_cache_out.shape != cached_out.shape:
            pytest.fail(
                f"prompt={prompt!r}\nshape mismatch: "
                f"no_cache={no_cache_out.shape} cached={cached_out.shape}"
            )
        diff_positions = (no_cache_out[0] != cached_out[0]).nonzero(as_tuple=True)[0]
        first = diff_positions[0].item()
        pytest.fail(
            f"prompt={prompt!r}\n"
            f"cache divergence at position {first} "
            f"(token index past prompt: {first - input_ids.shape[1]})\n"
            f"no_cache tokens [{first}:{first+5}] = {no_cache_out[0, first:first+5].tolist()}\n"
            f"cached   tokens [{first}:{first+5}] = {cached_out[0, first:first+5].tolist()}\n"
            f"no_cache text: {tok.decode(no_cache_out[0])!r}\n"
            f"cached   text: {tok.decode(cached_out[0])!r}"
        )


# ---------------------------------------------------------------------------
# Static batching with padding
# ---------------------------------------------------------------------------


def test_kv_cache_batched_roundtrip():
    """Per-batch-element writes don't leak across batch slots."""
    cfg = ModelConfig.qwen2_5_0_5b()
    B = 4
    cache = FlatKVCache(
        cfg, max_seq_len=16, dtype=torch.float32, device="cuda", batch_size=B
    )
    k = torch.randn(B, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    v = torch.randn(B, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    cache.write(layer=2, start_pos=3, k=k, v=v)

    k_read, v_read = cache.read(layer=2, end_pos=8)
    assert k_read.shape == (B, cfg.num_kv_heads, 8, cfg.head_dim)
    assert torch.equal(k_read[:, :, 3:8, :], k)
    assert torch.equal(v_read[:, :, 3:8, :], v)

    # Each batch element's slot is independent: writing seq 0 doesn't touch seq 1
    k_new = torch.randn(1, cfg.num_kv_heads, 2, cfg.head_dim, device="cuda")
    v_new = torch.randn(1, cfg.num_kv_heads, 2, cfg.head_dim, device="cuda")
    # Hack: write only into the first slot by indexing the cache directly
    cache.cache[2, 0, 0:1, :, 0:2, :] = k_new
    cache.cache[2, 1, 0:1, :, 0:2, :] = v_new
    k_after, _ = cache.read(layer=2, end_pos=8)
    assert torch.equal(k_after[0:1, :, 0:2, :], k_new)
    # Other batch elements untouched at the same positions
    assert torch.equal(k_after[1:, :, 3:8, :], k[1:, :, :, :])


def test_batched_matches_single_fp32():
    """Batch of 8 copies of one prompt produces the same tokens as a single run, fp32.

    With identical prompts, no padding is needed; this isolates the batched
    cache + batched attention plumbing from padding correctness.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)
    engine = Engine(model, tok, max_seq_len=512)

    prompt = "The capital of France is"
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.squeeze(0).tolist()
    n_new = 30
    B = 8

    # Reference: single-sequence path, repeated B times (deterministic, so all equal)
    input_ids = torch.tensor([prompt_ids], device="cuda")
    single_out = engine.generate_ids(input_ids, max_new_tokens=n_new)[0]  # [prompt_len + n_new]

    # Batched path
    batch_outs = engine.generate_batch_ids([prompt_ids] * B, max_new_tokens=n_new)
    assert len(batch_outs) == B

    for i, batch_out in enumerate(batch_outs):
        if not torch.equal(batch_out, single_out):
            diff_positions = (batch_out != single_out).nonzero(as_tuple=True)[0]
            first = diff_positions[0].item() if len(diff_positions) else "n/a"
            pytest.fail(
                f"batch element {i}: divergence at position {first}\n"
                f"single: {tok.decode(single_out)!r}\n"
                f"batch:  {tok.decode(batch_out)!r}"
            )


def test_padding_correctness_fp32():
    """Variable-length prompts in a batch produce the same per-sequence tokens
    as running each prompt individually (fp32).

    The actual test of the padding+causal mask: short prompts get left-padded
    to the longest, and their RoPE positions and attention masking must be
    set up so they generate identically to the unpadded single-prompt path.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)
    engine = Engine(model, tok, max_seq_len=512)

    prompts = [
        "Hi",
        "Once upon a time,",
        "The capital of France is",
        "def fibonacci(n):",
    ]
    prompt_token_lists = [
        tok(p, return_tensors="pt").input_ids.squeeze(0).tolist() for p in prompts
    ]
    n_new = 30

    # Reference: run each prompt through the single-sequence path
    single_outs = []
    for p_ids in prompt_token_lists:
        input_ids = torch.tensor([p_ids], device="cuda")
        out = engine.generate_ids(input_ids, max_new_tokens=n_new)[0]
        single_outs.append(out)

    # Batched (left-padded) path
    batch_outs = engine.generate_batch_ids(prompt_token_lists, max_new_tokens=n_new)
    assert len(batch_outs) == len(prompts)

    for i, (single, batched) in enumerate(zip(single_outs, batch_outs)):
        if single.shape != batched.shape:
            pytest.fail(
                f"prompt {i} ({prompts[i]!r}): shape {single.shape} vs {batched.shape}"
            )
        if not torch.equal(single, batched):
            diff_positions = (single != batched).nonzero(as_tuple=True)[0]
            first = diff_positions[0].item()
            pytest.fail(
                f"prompt {i} ({prompts[i]!r}): padding-path divergence at position {first}\n"
                f"single  text: {tok.decode(single)!r}\n"
                f"batched text: {tok.decode(batched)!r}"
            )


# ---------------------------------------------------------------------------
# Continuous batching (scheduler-driven)
# ---------------------------------------------------------------------------


STRESS_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
    "The mitochondrion is the",
    "import numpy as np\n\ndef softmax(x):",
    "Hello, my name is",
    "The quick brown fox",
    "In a galaxy far far away,",
    "def quicksort(arr):",
    "The history of computing began",
    "Hi there, I want to learn about",
    "Write a function that returns",
]


def test_continuous_batching_matches_single_fp32():
    """N prompts run through LLMEngine produce the same per-prompt token IDs
    as running each prompt individually through the single-sequence Engine.

    Tests the full continuous-batching stack: scheduler admit/retire, per-slot
    KV cache scatter/gather, variable-context-length attention mask, RoPE
    positions tied to each sequence's real progress rather than batch index.

    max_batch_size=3 with 5 prompts forces at least one sequence to wait,
    exercising the admission-on-completion path.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    n_new = 20

    # Reference: single-sequence outputs (one at a time, fresh cache each time)
    ref_engine = Engine(model, tok, max_seq_len=512)
    single_outs: list[list[int]] = []
    for p in PROMPTS:
        input_ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        out = ref_engine.generate_ids(input_ids, max_new_tokens=n_new)[0]
        single_outs.append(out.tolist())

    # Continuous batching: max_batch_size < len(PROMPTS) forces queueing
    llm = LLMEngine(model, tok, max_batch_size=3, max_seq_len=512)
    seqs = [llm.add_request(p, max_new_tokens=n_new) for p in PROMPTS]
    llm.run_until_done()

    for i, (p, seq) in enumerate(zip(PROMPTS, seqs)):
        cb_full = seq.prompt_token_ids + seq.output_token_ids
        single = single_outs[i]
        if cb_full == single:
            continue

        n = min(len(cb_full), len(single))
        first = next((j for j in range(n) if cb_full[j] != single[j]), n)
        pytest.fail(
            f"prompt {i} ({p!r}): cb vs single divergence at token index {first}\n"
            f"single        ({len(single)} tokens): {tok.decode(single)!r}\n"
            f"cont-batched  ({len(cb_full)} tokens): {tok.decode(cb_full)!r}"
        )


# ---------------------------------------------------------------------------
# Continuous batching: stress tests (heavy admission/retirement churn)
# ---------------------------------------------------------------------------


def _run_single_seq_references(
    tok, model, prompts: list[str], max_new_tokens: int
) -> list[list[int]]:
    """Run each prompt through the single-sequence Engine to get the ground truth.

    Shared between stress tests so we don't re-run reference generation 3x in
    the same pytest session (still expensive but at least factored).
    """
    ref_engine = Engine(model, tok, max_seq_len=512)
    refs: list[list[int]] = []
    for p in prompts:
        input_ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        out = ref_engine.generate_ids(input_ids, max_new_tokens=max_new_tokens)[0]
        refs.append(out.tolist())
    return refs


def test_continuous_batching_high_churn_fp32():
    """12 prompts through max_batch_size=2 forces 10 sequences to wait.

    Each admission requires the freed-slot reuse path, and each running
    sequence shares a forward pass with whichever other sequence is in the
    other slot — different decode partner on every step as completions stagger.

    Verifies per-prompt outputs still match the single-sequence reference
    despite this churn, proving the scheduler's slot reuse and the cache's
    per-slot scatter/gather don't bleed state across sequence reassignments.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    n_new = 12
    single_outs = _run_single_seq_references(tok, model, STRESS_PROMPTS, n_new)

    # max_batch_size=2 with 12 prompts → at any time only 2 run, 10+ wait.
    # Expect ~10 retirement→admission cycles over the full run.
    llm = LLMEngine(model, tok, max_batch_size=2, max_seq_len=512)
    seqs = [llm.add_request(p, max_new_tokens=n_new) for p in STRESS_PROMPTS]
    llm.run_until_done()

    for i, (p, seq) in enumerate(zip(STRESS_PROMPTS, seqs)):
        cb_full = seq.prompt_token_ids + seq.output_token_ids
        single = single_outs[i]
        if cb_full == single:
            continue
        n = min(len(cb_full), len(single))
        first = next((j for j in range(n) if cb_full[j] != single[j]), n)
        pytest.fail(
            f"prompt {i} ({p!r}): high-churn divergence at token {first}\n"
            f"single ({len(single)}): {tok.decode(single)!r}\n"
            f"cb     ({len(cb_full)}): {tok.decode(cb_full)!r}"
        )


def test_continuous_batching_varied_max_tokens_fp32():
    """Same prompt, 8 sequences, each with a different max_new_tokens.

    The sequences all generate identical tokens (greedy on identical inputs)
    but stop at different decode steps: seq[0] retires after 6 tokens,
    seq[1] after 10, ..., seq[7] after 34. This exercises asymmetric
    retirement (sequences finishing at different points while others continue)
    and verifies that early retirees don't corrupt later sequences sharing
    the same batch on subsequent steps.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    prompt = "The capital of France is"
    max_tokens_list = [6, 10, 14, 18, 22, 26, 30, 34]

    # Reference: run the prompt once at the maximum budget; each shorter
    # variant is just a prefix of this run (greedy + same prompt is deterministic).
    ref_engine = Engine(model, tok, max_seq_len=512)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    full_ref = ref_engine.generate_ids(
        prompt_ids, max_new_tokens=max(max_tokens_list)
    )[0].tolist()

    # CB: 8 copies of the prompt with varying max_new_tokens.
    # max_batch_size=3 keeps 5 in waiting initially.
    llm = LLMEngine(model, tok, max_batch_size=3, max_seq_len=512)
    seqs = [llm.add_request(prompt, max_new_tokens=n) for n in max_tokens_list]
    llm.run_until_done()

    for n_target, seq in zip(max_tokens_list, seqs):
        cb_full = seq.prompt_token_ids + seq.output_token_ids
        # Expected length: prompt + up to n_target generated tokens (or fewer if EOS hit).
        # Compare against the corresponding prefix of the longest single run.
        ref_prefix = full_ref[: len(cb_full)]
        if cb_full == ref_prefix:
            assert seq.output_len <= n_target, (
                f"max_new_tokens={n_target} overran: produced {seq.output_len} tokens"
            )
            continue
        n = min(len(cb_full), len(ref_prefix))
        first = next((j for j in range(n) if cb_full[j] != ref_prefix[j]), n)
        pytest.fail(
            f"max_new_tokens={n_target}: divergence at token {first}\n"
            f"ref prefix ({len(ref_prefix)}): {tok.decode(ref_prefix)!r}\n"
            f"cb         ({len(cb_full)}): {tok.decode(cb_full)!r}"
        )


def test_continuous_batching_mid_run_admission_fp32():
    """Add 3 prompts, run several steps, then add 3 more while the first
    batch is still decoding.

    Verifies that mid-flight admission (new requests joining a running engine)
    doesn't disrupt in-progress sequences and that the late arrivals still
    produce correct outputs after starting from a non-empty engine state.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    first_wave = STRESS_PROMPTS[:3]
    second_wave = STRESS_PROMPTS[3:6]
    all_prompts = first_wave + second_wave
    n_new = 15

    single_outs = _run_single_seq_references(tok, model, all_prompts, n_new)

    llm = LLMEngine(model, tok, max_batch_size=4, max_seq_len=512)

    # First wave: add 3, run several steps so they get past prefill and into decode
    seqs = [llm.add_request(p, max_new_tokens=n_new) for p in first_wave]
    for _ in range(8):
        if not llm.has_work():
            break
        llm.step()

    # Mid-flight: add 3 more while the first wave is still decoding
    seqs += [llm.add_request(p, max_new_tokens=n_new) for p in second_wave]

    # Now drive to completion
    llm.run_until_done()

    for i, (p, seq) in enumerate(zip(all_prompts, seqs)):
        cb_full = seq.prompt_token_ids + seq.output_token_ids
        single = single_outs[i]
        wave = "1st" if i < len(first_wave) else "2nd"
        if cb_full == single:
            continue
        n = min(len(cb_full), len(single))
        first = next((j for j in range(n) if cb_full[j] != single[j]), n)
        pytest.fail(
            f"prompt {i} ({p!r}, {wave} wave): mid-run divergence at token {first}\n"
            f"single ({len(single)}): {tok.decode(single)!r}\n"
            f"cb     ({len(cb_full)}): {tok.decode(cb_full)!r}"
        )


def test_engine_handles_seq_length_overflow():
    """A sequence whose total length exceeds max_seq_len triggers a clean
    RuntimeError from the KV cache. The engine doesn't silently corrupt:
      - sequences that fit complete before the crash and are retired correctly
      - the error message clearly identifies the overflow
      - scheduler bookkeeping (running set, free slots) is intact after the
        exception, so the crash is recoverable in principle

    This is the only "out of cache memory" failure mode the current preallocated
    design exposes. There's no graceful preemption — overflow on one sequence
    aborts the step (and currently the engine), which is the right semantics
    for a fail-loud educational implementation.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    # Tight cache: 30 positions per slot. A 5-token completion fits comfortably
    # (prompt is ~5 tokens + 5 output = ~10 ≤ 30); 100 output tokens overflows
    # well before completion.
    SHORT_TOKENS = 5
    LONG_TOKENS = 100
    llm = LLMEngine(model, tok, max_batch_size=2, max_seq_len=30)

    good = llm.add_request("The capital of France is", max_new_tokens=SHORT_TOKENS)
    bad = llm.add_request("The capital of France is", max_new_tokens=LONG_TOKENS)

    with pytest.raises(RuntimeError, match="overflow"):
        llm.run_until_done()

    # The short sequence completed and was retired before the crash.
    assert good.is_finished, "good seq should have been retired before the crash"
    assert good.output_len == SHORT_TOKENS, (
        f"good seq should have produced {SHORT_TOKENS} tokens, got {good.output_len}"
    )

    # The long sequence got past the short budget before crashing.
    assert bad.output_len > SHORT_TOKENS, (
        f"bad seq should have produced > {SHORT_TOKENS} tokens before the crash, "
        f"got {bad.output_len}"
    )

    # Scheduler bookkeeping is intact: bad is still tracked, good's slot is free.
    assert bad in llm.scheduler.running, "bad seq should still be tracked as running"
    assert good not in llm.scheduler.running, "good seq should have been retired"
    assert len(llm.scheduler.free_slots) == 1, (
        f"good's slot should be back in the free pool; "
        f"free_slots={llm.scheduler.free_slots}"
    )


def test_continuous_batching_heavy_load_stress_fp32():
    """Sustained-load stress test: 64 identical requests through max_batch_size=4.

    Drives the engine through 16 batch waves (~60 admission/retirement cycles).
    Each wave does 4 single-seq prefills then ~39 batched decode steps to
    completion, for ~688 total forward passes. Runs ~25-35 s on A10 in fp32.

    Because every prompt is identical and decoding is deterministic-greedy in
    fp32, all 64 sequences MUST produce the exact same output as the
    single-sequence reference. Any divergence indicates cross-sequence
    corruption — wrong slot scatter, mask off-by-one, RoPE position mixup,
    or stale data leaking from a previous occupant of a reused slot.

    A test failing here when the smaller stress tests pass would point at
    cumulative-state bugs that only appear under sustained churn — exactly
    the kind of bug that slips through 5- or 12-sequence tests.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    N_REQUESTS = 64
    N_NEW_TOKENS = 40
    MAX_BATCH_SIZE = 4

    prompt = "The capital of France is"

    # Single-sequence reference — run once; all repeated requests must match it.
    ref_engine = Engine(model, tok, max_seq_len=512)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    reference = ref_engine.generate_ids(
        input_ids, max_new_tokens=N_NEW_TOKENS
    )[0].tolist()

    # Continuous-batching run.
    llm = LLMEngine(
        model, tok, max_batch_size=MAX_BATCH_SIZE, max_seq_len=512
    )
    seqs = [
        llm.add_request(prompt, max_new_tokens=N_NEW_TOKENS)
        for _ in range(N_REQUESTS)
    ]
    llm.run_until_done()

    # Every one of N_REQUESTS sequences should match the reference exactly.
    failures: list[tuple[int, int]] = []  # (seq_index, first_diverging_token_idx)
    for i, seq in enumerate(seqs):
        cb_full = seq.prompt_token_ids + seq.output_token_ids
        if cb_full == reference:
            continue
        n = min(len(cb_full), len(reference))
        first = next((j for j in range(n) if cb_full[j] != reference[j]), n)
        failures.append((i, first))

    if failures:
        first_idx, first_tok = failures[0]
        seq = seqs[first_idx]
        cb_full = seq.prompt_token_ids + seq.output_token_ids
        failed_seq_indices = [i for i, _ in failures]
        head = failed_seq_indices[:10]
        more = f" (and {len(failed_seq_indices) - 10} more)" if len(failed_seq_indices) > 10 else ""
        pytest.fail(
            f"{len(failures)}/{N_REQUESTS} sequences diverged from the single-seq reference.\n"
            f"first divergence: seq index {first_idx} at token index {first_tok}\n"
            f"failed seq indices: {head}{more}\n"
            f"reference: {tok.decode(reference)!r}\n"
            f"diverged:  {tok.decode(cb_full)!r}"
        )


# ---------------------------------------------------------------------------
# Paged attention (PyTorch reference implementation)
# ---------------------------------------------------------------------------


def test_paged_attention_matches_single_seq_fp32():
    """N prompts through PagedLLMEngine produce token-identical outputs to
    running each prompt individually through the single-sequence Engine.

    The paged path lays out K/V in 16-token blocks tracked by per-sequence
    block tables instead of fixed slots. Mathematically the gathered K/V
    must equal the contiguous flat-cache layout — same values, different
    physical addresses. Failure here is a bug in the BlockManager scatter/
    gather, in the per-prefill / per-decode block bookkeeping, or in the
    paged attention mask construction.

    Uses an intentionally tight block pool (just enough to hold ~4 sequences
    at once given typical prompt lengths) so admission has to defer waiting
    sequences until earlier ones finish and return their blocks.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    n_new = 20

    # Single-sequence reference: one prompt at a time, fresh flat cache.
    ref_engine = Engine(model, tok, max_seq_len=512)
    single_outs: list[list[int]] = []
    for p in PROMPTS:
        input_ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        out = ref_engine.generate_ids(input_ids, max_new_tokens=n_new)[0]
        single_outs.append(out.tolist())

    # Paged engine. Small block pool: 32 blocks * 16 = 512 token-slots total,
    # enough to hold the test prompts but tight enough that not all 5 fit
    # simultaneously. max_batch_size=3 mirrors the LLMEngine integration test.
    paged = PagedLLMEngine(
        model, tok, num_blocks=32, block_size=16, max_batch_size=3
    )
    seqs = [paged.add_request(p, max_new_tokens=n_new) for p in PROMPTS]
    paged.run_until_done()

    for i, (p, seq) in enumerate(zip(PROMPTS, seqs)):
        paged_full = seq.prompt_token_ids + seq.output_token_ids
        single = single_outs[i]
        if paged_full == single:
            continue
        n = min(len(paged_full), len(single))
        first = next((j for j in range(n) if paged_full[j] != single[j]), n)
        pytest.fail(
            f"prompt {i} ({p!r}): paged vs single divergence at token index {first}\n"
            f"single ({len(single)}): {tok.decode(single)!r}\n"
            f"paged  ({len(paged_full)}): {tok.decode(paged_full)!r}"
        )


def test_paged_engine_releases_blocks_on_completion():
    """After all sequences finish, every block should be back in the free pool.

    Catches block-table-leak bugs: if `_retire` forgets to free a sequence's
    blocks, the pool slowly drains across many requests until admission
    deadlocks. We don't test this directly under load, but a single completion
    cycle is sufficient to catch the basic leak.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    NUM_BLOCKS = 16
    paged = PagedLLMEngine(
        model, tok, num_blocks=NUM_BLOCKS, block_size=16, max_batch_size=2
    )

    # Add and run a few requests to completion
    for _ in range(3):
        paged.add_request("The capital of France is", max_new_tokens=10)
    paged.run_until_done()

    # Pool should be fully refilled
    assert paged.block_manager.num_free == NUM_BLOCKS, (
        f"block leak: only {paged.block_manager.num_free} of {NUM_BLOCKS} free "
        f"after all sequences finished"
    )
