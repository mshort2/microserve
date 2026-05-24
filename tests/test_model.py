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
    """Smoke check: bf16 logits agree approximately with HF.

    Tolerance is loose because sdpa picks different fused kernels for HF vs us
    (contiguity, repeat_kv ordering) and FlashAttention bf16 accumulation
    differs by ~0.1-0.5 across 24 layers. Real architecture bugs would show
    diffs of 1+. The tight architecture gate lives in test_greedy_matches_hf_fp32.
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
        assert max_abs_diff < 1.0, (
            f"prompt={prompt!r} max_abs_diff={max_abs_diff} too large for bf16 noise — likely a real bug"
        )


def _greedy_no_cache(model: QwenForCausalLM, input_ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Generate n_new tokens greedily via repeated full prefill (no KV cache).

    O(N^2) per prompt — slow, but validates the architecture across positions
    without depending on the Phase 1b KV cache.
    """
    output_ids = input_ids
    for _ in range(n_new):
        with torch.no_grad():
            logits = model(output_ids)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_tok], dim=-1)
    return output_ids


def test_greedy_matches_hf_fp32():
    """Phase 1a exit gate: 5 prompts x 50 greedy tokens identical to HF, in fp32.

    Run in fp32 to remove bf16 sdpa kernel-selection noise as a confound. In
    fp32, two correct implementations of the same architecture should produce
    bit-identical greedy sequences. A failure here is a real architecture bug.

    Uses repeated full prefill (no KV cache) — failure mode is bisectable to
    the model, not Phase 1b's cache. Bf16 inference at runtime is still fine;
    it just won't match HF token-for-token because of fused-kernel differences.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    ours = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    n_new = 50
    for prompt in PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            hf_out = hf.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_new,
                min_new_tokens=n_new,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        our_out = _greedy_no_cache(ours, input_ids, n_new)

        if torch.equal(hf_out, our_out):
            continue

        if hf_out.shape != our_out.shape:
            pytest.fail(
                f"prompt={prompt!r}\nshape mismatch: hf={hf_out.shape} ours={our_out.shape}"
            )
        diff_positions = (hf_out[0] != our_out[0]).nonzero(as_tuple=True)[0]
        first = diff_positions[0].item()
        pytest.fail(
            f"prompt={prompt!r}\n"
            f"first divergence at position {first} (token index past prompt: {first - input_ids.shape[1]})\n"
            f"hf   tokens [{first}:{first+5}] = {hf_out[0, first:first+5].tolist()}\n"
            f"ours tokens [{first}:{first+5}] = {our_out[0, first:first+5].tolist()}\n"
            f"hf   text: {tok.decode(hf_out[0])!r}\n"
            f"ours text: {tok.decode(our_out[0])!r}"
        )
