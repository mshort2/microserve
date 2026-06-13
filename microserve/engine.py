from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from microserve.kv_cache import FlatKVCache
from microserve.model import QwenForCausalLM
from microserve.scheduler import Scheduler
from microserve.sequence import Sequence

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class Engine:
    """Single-sequence and static-batched greedy inference.

    - `generate(prompt)` / `generate_ids(input_ids)`: one sequence at a time.
    - `generate_batch(prompts)` / `generate_batch_ids(token_lists)`: a fixed
      batch with left-padded prompts and a combined padding+causal attention
      mask.
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

    # ---- Single-sequence ---------------------------------------------------

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        """Greedy-decode one sequence. Returns prompt + generated as token IDs.

        Stops early on EOS.
        """
        assert input_ids.shape[0] == 1, (
            "generate_ids is single-sequence; use generate_batch_ids for B>1"
        )
        prompt_len = input_ids.shape[1]

        cache = FlatKVCache(
            self.model.cfg,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            device=self.device,
            batch_size=1,
        )

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
        """User-facing single-prompt generation. Returns the completion only."""
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

    # ---- Batched (static, left-padded prompts) -----------------------------

    def _build_attn_mask(
        self,
        pad_lens: torch.Tensor,
        T: int,
        end_pos: int,
    ) -> torch.Tensor:
        """Build [B, 1, T, end_pos] additive attention mask for the batched path.

        Combines:
          - padding mask on KEYS (keys at positions < pad_lens[b] are invalid)
          - causal mask within the prompt (q >= k) during prefill (T > 1)

        Padding QUERIES in prefill would otherwise have all keys masked, which
        produces NaN under sdpa. We force their diagonal True so they attend to
        themselves; the NaN-free garbage output sits in padding positions and
        never gets read.

        Decode (T == 1): the query is always the most-recently-generated token
        at a real position, so it has at least one valid key — no diagonal fix
        needed.
        """
        key_positions = torch.arange(end_pos, device=self.device)
        pad_valid = key_positions[None, :] >= pad_lens[:, None]  # [B, end_pos]

        if T == 1:
            combined = pad_valid.unsqueeze(1)  # [B, 1, end_pos]
        else:
            query_positions = torch.arange(end_pos - T, end_pos, device=self.device)
            causal_valid = query_positions[:, None] >= key_positions[None, :]  # [T, end_pos]
            combined = pad_valid.unsqueeze(1) & causal_valid.unsqueeze(0)  # [B, T, end_pos]
            # Allow padding queries to attend to themselves to avoid NaN softmax.
            diag_idx = torch.arange(T, device=self.device)
            # combined shape [B, T, end_pos]; for prefill end_pos == T.
            combined = combined.clone()
            combined[:, diag_idx, diag_idx] = True

        additive = torch.zeros(combined.shape, dtype=self.dtype, device=self.device)
        additive.masked_fill_(~combined, torch.finfo(self.dtype).min)
        return additive.unsqueeze(1)  # [B, 1, T, end_pos]

    @torch.no_grad()
    def generate_batch_ids(
        self,
        prompt_token_lists: list[list[int]],
        max_new_tokens: int = 50,
    ) -> list[torch.Tensor]:
        """Greedy-decode a batch of variable-length prompts (left-padded).

        Generates exactly `max_new_tokens` per sequence (no EOS early-stop) so
        the returned lengths are deterministic for testing. Returns one 1D
        tensor per input prompt: prompt token IDs + generated token IDs.
        """
        B = len(prompt_token_lists)
        max_prompt_len = max(len(p) for p in prompt_token_lists)
        real_prompt_lens = torch.tensor(
            [len(p) for p in prompt_token_lists],
            device=self.device,
            dtype=torch.long,
        )
        pad_lens = max_prompt_len - real_prompt_lens

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        padded = torch.tensor(
            [[pad_id] * (max_prompt_len - len(p)) + p for p in prompt_token_lists],
            device=self.device,
            dtype=torch.long,
        )

        # Per-sequence RoPE positions during prefill: real positions 0..real_len-1
        # at the real token slots, clamped-to-0 (i.e. don't care) at padding slots.
        positions_2d = (
            torch.arange(max_prompt_len, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
            - pad_lens.unsqueeze(1)
        ).clamp(min=0)

        cache = FlatKVCache(
            self.model.cfg,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            device=self.device,
            batch_size=B,
        )

        prefill_mask = self._build_attn_mask(
            pad_lens, T=max_prompt_len, end_pos=max_prompt_len
        )
        logits = self.model(
            padded,
            positions=positions_2d,
            cache=cache,
            cache_start=0,
            attn_mask=prefill_mask,
        )
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        generated = [next_tokens]

        for step in range(max_new_tokens - 1):
            cache_pos = max_prompt_len + step
            # Per-sequence RoPE position: real_prompt_len + step.
            decode_positions = (real_prompt_lens + step).unsqueeze(-1)  # [B, 1]
            decode_mask = self._build_attn_mask(
                pad_lens, T=1, end_pos=cache_pos + 1
            )
            logits = self.model(
                next_tokens,
                positions=decode_positions,
                cache=cache,
                cache_start=cache_pos,
                attn_mask=decode_mask,
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tokens)

        all_generated = torch.cat(generated, dim=1)  # [B, max_new_tokens]

        outputs = []
        for i, prompt in enumerate(prompt_token_lists):
            full = torch.tensor(
                prompt + all_generated[i].tolist(),
                device=self.device,
                dtype=torch.long,
            )
            outputs.append(full)
        return outputs

    def generate_batch(
        self, prompts: list[str], max_new_tokens: int = 50
    ) -> list[str]:
        """User-facing batched generation. Each output is the completion (no prompt)."""
        prompt_token_lists = [
            self.tokenizer(p, return_tensors="pt").input_ids.squeeze(0).tolist()
            for p in prompts
        ]
        out_id_tensors = self.generate_batch_ids(
            prompt_token_lists, max_new_tokens=max_new_tokens
        )
        completions = []
        for prompt_ids, full in zip(prompt_token_lists, out_id_tensors):
            generated_ids = full[len(prompt_ids):]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            assert isinstance(text, str)
            completions.append(text)
        return completions


class LLMEngine:
    """Continuous-batching inference engine with persistent state.

    Owns a `Scheduler`, a pre-allocated KV cache for `max_batch_size`
    sequences, and the model. Each `step()` runs ONE forward pass — either a
    prefill of one waiting sequence (admitted into a free slot) or a decode
    step for all currently running sequences. Prefill is prioritized whenever
    capacity allows.

    Unlike `Engine`, this class is stateful: requests can be added at any time,
    they enter the running set as space frees up, and finished sequences free
    their slot for the next request.
    """

    def __init__(
        self,
        model: QwenForCausalLM,
        tokenizer: "PreTrainedTokenizerBase",
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype

        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.cache = FlatKVCache(
            model.cfg,
            max_seq_len=max_seq_len,
            dtype=self.dtype,
            device=self.device,
            batch_size=max_batch_size,
        )

    def add_request(self, prompt: str, max_new_tokens: int = 50) -> Sequence:
        token_ids = (
            self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).tolist()
        )
        seq = Sequence(prompt_token_ids=token_ids, max_new_tokens=max_new_tokens)
        self.scheduler.add(seq)
        return seq

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    @torch.no_grad()
    def step(self) -> list[Sequence]:
        """One scheduler+forward step. Returns sequences that finished this step."""
        sched = self.scheduler

        # Prefer prefill: admit one waiting sequence and run its prefill.
        if sched.waiting and len(sched.running) < self.max_batch_size:
            seq = sched.admit_one()
            assert seq is not None
            self._do_prefill(seq)
            if self._is_finished(seq):
                sched.retire(seq)
                return [seq]
            return []

        # Otherwise decode all running sequences in one batched forward pass.
        if sched.running:
            self._do_decode(sched.running)
            finished = []
            for seq in list(sched.running):
                if self._is_finished(seq):
                    sched.retire(seq)
                    finished.append(seq)
            return finished

        return []

    def run_until_done(self, max_steps: int = 100_000) -> dict[int, Sequence]:
        """Run until no work remains. Returns {seq_id: finished Sequence}."""
        finished: dict[int, Sequence] = {}
        for _ in range(max_steps):
            if not self.has_work():
                break
            for seq in self.step():
                finished[seq.seq_id] = seq
        return finished

    def _is_finished(self, seq: Sequence) -> bool:
        if not seq.output_token_ids:
            return False
        if seq.output_token_ids[-1] == self.tokenizer.eos_token_id:
            return True
        if seq.output_len >= seq.max_new_tokens:
            return True
        return False

    def _do_prefill(self, seq: Sequence) -> None:
        input_ids = torch.tensor(
            [seq.prompt_token_ids], device=self.device, dtype=torch.long
        )
        T = input_ids.shape[1]
        positions = torch.arange(T, device=self.device).unsqueeze(0)  # [1, T]
        slot_idxs = torch.tensor([seq.slot_idx], device=self.device, dtype=torch.long)
        cache_starts = torch.tensor([0], device=self.device, dtype=torch.long)

        logits = self.model(
            input_ids,
            positions=positions,
            cache=self.cache,
            slot_idxs=slot_idxs,
            cache_starts=cache_starts,
        )
        next_token = int(logits[0, -1, :].argmax(dim=-1).item())
        seq.append_token(next_token)

    def _do_decode(self, sequences: list[Sequence]) -> None:
        B = len(sequences)
        # Each sequence's "last output token" is the input to this decode step.
        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in sequences],
            device=self.device,
            dtype=torch.long,
        )

        # Per-sequence RoPE position: the position where the NEW token lives.
        # That's seq.context_len (= cache entries already in place).
        context_lens = torch.tensor(
            [s.context_len for s in sequences],
            device=self.device,
            dtype=torch.long,
        )
        positions = context_lens.unsqueeze(-1)  # [B, 1]
        slot_idxs = torch.tensor(
            [s.slot_idx for s in sequences], device=self.device, dtype=torch.long
        )
        cache_starts = context_lens

        # Attention mask over the variable-extent K/V history: valid where
        # key_pos < context_lens[b] + 1 (i.e., includes the K/V just written).
        end_positions = context_lens + 1
        max_ctx = int(end_positions.max().item())
        key_positions = torch.arange(max_ctx, device=self.device)
        valid = key_positions[None, :] < end_positions[:, None]  # [B, max_ctx]
        additive = torch.zeros((B, max_ctx), dtype=self.dtype, device=self.device)
        additive.masked_fill_(~valid, torch.finfo(self.dtype).min)
        attn_mask = additive.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, max_ctx]

        logits = self.model(
            input_ids,
            positions=positions,
            cache=self.cache,
            slot_idxs=slot_idxs,
            cache_starts=cache_starts,
            attn_mask=attn_mask,
        )
        next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
        for s, t in zip(sequences, next_tokens):
            s.append_token(int(t))
