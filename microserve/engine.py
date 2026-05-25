from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from microserve.kv_cache import FlatKVCache
from microserve.model import QwenForCausalLM

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class Engine:
    """Single-sequence greedy inference: tokenize -> prefill -> decode loop -> detokenize.

    Phase 1b only handles one request at a time. Phase 3 replaces this with a
    Scheduler that batches many requests through the same model per step.
    """

    def __init__(
        self,
        model: QwenForCausalLM,
        tokenizer: "PreTrainedTokenizerBase",
        max_seq_len: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        """Greedy-decode up to max_new_tokens. Returns prompt + generated as token IDs.

        For test use. Stops early on EOS.
        """
        assert input_ids.shape[0] == 1, "Phase 1b engine is single-sequence"
        prompt_len = input_ids.shape[1]

        cache = FlatKVCache(
            self.model.cfg,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            device=self.device,
        )

        # Prefill the whole prompt at cache positions [0, prompt_len)
        logits = self.model(input_ids, cache=cache, cache_start=0)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]

        eos = self.tokenizer.eos_token_id
        for step in range(max_new_tokens - 1):
            if next_token.item() == eos:
                break
            pos = prompt_len + step
            positions = torch.tensor([[pos]], device=self.device)
            logits = self.model(
                next_token,
                positions=positions,
                cache=cache,
                cache_start=pos,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        return torch.cat([input_ids, *generated], dim=1)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """User-facing: prompt string in, generated string out (excludes the prompt)."""
        input_ids = (
            self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        )
        prompt_len = input_ids.shape[1]
        out_ids = self.generate_ids(input_ids, max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(
            out_ids[0, prompt_len:], skip_special_tokens=True
        )
        assert isinstance(text, str)
        return text
