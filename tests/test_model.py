import pytest
import torch

from microserve.config import ModelConfig
from microserve.engine import Engine
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
    """Smoke check: bf16 logits agree with HF within sdpa kernel-selection noise.

    HF and microserve both call sdpa in bf16, but HF uses an explicit causal
    mask while we use is_causal=True. sdpa then picks different fused kernels
    (FlashAttention vs memory-efficient vs math) and accumulates bf16
    differently across 24 layers. Empirically ~1-2 max diff is normal. A real
    architectural bug shows diffs of 10+.
    """
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
    """Phase 1a exit gate (logits). In fp32 the sdpa kernel-selection noise
    drops to <1e-4, so two correct implementations should agree on logits
    to ~1e-3. Failure here is a real architecture bug.
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
    """Phase 1a exit gate (architecture). Per-layer hidden state diff in fp32.

    Walks all 24 decoder layers and asserts the output residual stream matches
    HF within 1e-3 at every layer. This proves every component (embed, attn,
    MLP, norms, RoPE, GQA) is correct independently — not just that the final
    logits happen to land close.

    This is the *real* Phase 1a gate. Greedy-token matching is unreliable for
    cross-implementation comparison because close-call argmax decisions flip
    under tiny numerical noise (this is true even between two correct fp32
    implementations of the same model).
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
# Phase 1b tests: KV cache + cached greedy decode
# ---------------------------------------------------------------------------


def _greedy_no_cache(model: QwenForCausalLM, input_ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Greedy decode via repeated full prefill — no KV cache. Validated against HF in Phase 1a."""
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

    # Other layers must be untouched (not equal to layer 3's written values)
    k_other, _ = cache.read(layer=0, end_pos=6)
    assert not torch.equal(k_other, k_read), "writes leaked across layers"


def test_kv_cache_overflow_raises():
    """Writing past max_seq_len should raise loudly, not silently corrupt."""
    cfg = ModelConfig.qwen2_5_0_5b()
    cache = FlatKVCache(cfg, max_seq_len=8, dtype=torch.float32, device="cuda")
    k = torch.zeros(1, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    v = torch.zeros(1, cfg.num_kv_heads, 5, cfg.head_dim, device="cuda")
    with pytest.raises(RuntimeError, match="overflow"):
        cache.write(layer=0, start_pos=5, k=k, v=v)


def test_greedy_cached_matches_no_cache_fp32():
    """Phase 1b exit gate: cache-based decode == no-cache repeated prefill (in fp32).

    The no-cache path is validated against HF in Phase 1a, so a match here
    transitively validates the cache. A failure here is necessarily a bug in
    FlatKVCache or the cache-aware code in Attention.forward — not in the
    architecture, since the architecture path is unchanged from Phase 1a.
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
