"""CLI: greedy text generation against Qwen 2.5 0.5B (single or batched).

Examples:
    python examples/generate.py "The capital of France is"
    python examples/generate.py "Tell me about CUDA" --max-tokens 100
    python examples/generate.py "def fib(n):" --dtype fp32

    # Multiple prompts (batched):
    python examples/generate.py "Hi" "Once upon a time," "def fib(n):"
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from microserve.engine import Engine
from microserve.model import QwenForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "prompts", nargs="+", help="One or more prompts. >1 enables batched path."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Max new tokens to generate."
    )
    parser.add_argument(
        "--dtype", choices=list(_DTYPE_MAP), default="bf16", help="Compute dtype."
    )
    args = parser.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = QwenForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    engine = Engine(model, tokenizer)

    if len(args.prompts) == 1:
        completion = engine.generate(args.prompts[0], max_new_tokens=args.max_tokens)
        print(f"{args.prompts[0]}{completion}")
    else:
        completions = engine.generate_batch(
            args.prompts, max_new_tokens=args.max_tokens
        )
        for p, c in zip(args.prompts, completions):
            print(f"--- prompt: {p!r} ---")
            print(f"{p}{c}\n")


if __name__ == "__main__":
    main()
