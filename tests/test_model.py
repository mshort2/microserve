import pytest
import torch

from microserve.config import ModelConfig
from microserve.engine import Engine, LLMEngine
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
