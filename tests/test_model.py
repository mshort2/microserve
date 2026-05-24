import pytest
import torch

from microserve.config import ModelConfig
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
