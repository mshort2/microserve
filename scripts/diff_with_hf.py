"""Layer-by-layer comparison of microserve vs HuggingFace Qwen2.5-0.5B in fp32.

Run once on a GPU box to localize where our forward pass diverges from HF:
    python scripts/diff_with_hf.py

Output shows max and mean absolute diff of the residual stream after each
named submodule. The first row where max_diff jumps by >10x from the previous
row is where the bug lives.

Interpretation:
  - embed diff != 0     -> embedding weight loading or lookup is broken
  - layer_N diff jumps  -> bug is in layer N (RoPE, attn, MLP, norms)
  - norm diff jumps     -> bug is in the final norm (unlikely)
  - All tiny but logits diverge -> bug is in lm_head / tied embed direction
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microserve.model import QwenForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    ours = QwenForCausalLM.from_pretrained(MODEL_ID, device="cuda", dtype=torch.float32)

    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")

    hf_outs: dict[str, torch.Tensor] = {}
    our_outs: dict[str, torch.Tensor] = {}

    def _grab(store, name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            store[name] = t.detach()
        return hook

    handles = []
    handles.append(hf.model.embed_tokens.register_forward_hook(_grab(hf_outs, "embed")))
    handles.append(ours.embed.register_forward_hook(_grab(our_outs, "embed")))
    for i in range(len(hf.model.layers)):
        handles.append(hf.model.layers[i].register_forward_hook(_grab(hf_outs, f"layer_{i:02d}")))
        handles.append(ours.layers[i].register_forward_hook(_grab(our_outs, f"layer_{i:02d}")))
    handles.append(hf.model.norm.register_forward_hook(_grab(hf_outs, "final_norm")))
    handles.append(ours.norm.register_forward_hook(_grab(our_outs, "final_norm")))

    with torch.no_grad():
        hf_logits = hf(input_ids).logits
        our_logits = ours(input_ids)

    for h in handles:
        h.remove()

    print(f"\nPrompt: {PROMPT!r}  ({input_ids.shape[1]} tokens, fp32)")
    print(f"{'site':<14} {'max_abs_diff':<14} {'mean_abs_diff':<14} {'hf_norm':<10} {'our_norm':<10}")
    print("-" * 70)
    names = ["embed"] + [f"layer_{i:02d}" for i in range(len(hf.model.layers))] + ["final_norm"]
    for name in names:
        h = hf_outs[name].float()
        o = our_outs[name].float()
        if h.shape != o.shape:
            print(f"{name:<14} SHAPE MISMATCH hf={tuple(h.shape)} ours={tuple(o.shape)}")
            continue
        diff = (h - o).abs()
        print(
            f"{name:<14} {diff.max().item():<14.4e} {diff.mean().item():<14.4e} "
            f"{h.norm().item():<10.4f} {o.norm().item():<10.4f}"
        )

    print("-" * 70)
    logits_diff = (hf_logits.float() - our_logits.float()).abs()
    print(f"{'logits':<14} {logits_diff.max().item():<14.4e} {logits_diff.mean().item():<14.4e}")


if __name__ == "__main__":
    main()
