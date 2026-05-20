# microserve — Project Plan

**CMU Senior · AI Infra Portfolio Project · May 2026**

A minimal but credible LLM inference engine: continuous batching, paged KV cache, and a custom CUDA attention kernel with block-table indirection. These are the three concepts that define modern LLM serving — the project demonstrates each as a working artifact with benchmark numbers.

Target roles:

- **NVIDIA** — compiler / CUDA / AI infra new grad
- **Anthropic** — infra / systems engineering
- **Google DeepMind** — ML systems / serving infra
- **Amazon Annapurna** — silicon and inference stack

The 15-418 TA background is what makes the CUDA kernel work credible rather than a stretch — lean into that in interviews.

---

## What this plan is *not*

Explicit cut list, to keep the project legible and the scope finishable:

- **No HTTP server.** No FastAPI, no Pydantic schemas, no `/generate` endpoint. None of that demonstrates the skills these teams hire for.
- **Not a vLLM beater.** The honest comparison baseline is vLLM itself; the expected outcome is 30–70% of its throughput with a defensible explanation of the gap (FlashAttention-3, chunked prefill, CUDA graphs — all explicitly out of scope).
- **No tensor parallelism, no quantization, no LoRA, no speculative decoding, no prefix caching, no chunked prefill.** Each is a real project on its own; bundling them dilutes signal.

---

## Hardware

| Use | Machine | When |
|---|---|---|
| Dev | RTX 3090 Ti, 24 GB, sm_86 (remote Linux via SSH) | Phases 0–4 |
| Benchmark headline | A100 40 GB on Lambda Labs (~$1.10/hr) | Phase 5 |
| Hopper portability | H100 on Vast.ai (~$2–3/hr), optional | Phase 5 |

Total cloud budget: **$50–100**. Avoid AWS/GCP/Azure — 5–10× pricing premium for no benefit at this scale.

CUDA kernel arch targets: `sm_86` primary, add `sm_90` at build time for the H100 run.

---

## Model choice

- **Development:** `Qwen/Qwen2.5-0.5B-Instruct`. GQA (14 Q heads / 2 KV heads), Llama-shape architecture, ungated on HF, fits in < 2 GB fp16 so KV cache experiments are unconstrained.
- **Benchmark headline:** `meta-llama/Meta-Llama-3-8B`, on the rented A100 only.

Develop fast against the small model; run the big model once everything works. Llama-3-8B in fp16 is ~16 GB of weights alone — iterating against it on a 3090 Ti would be miserable and on a Colab T4 (16 GB) impossible. The resume bullet still says "Llama 3 8B" because that's what the benchmark numbers come from.

---

## Phase overview

| Phase | Focus | Realistic | Hardware |
|---|---|---|---|
| 0 | Skeleton + HF weight loading | 2 days | Local |
| 1 | Llama forward pass; greedy decode matches HF | 1 week | 3090 Ti |
| 2 | Static batched generation | 3 days | 3090 Ti |
| 3 | Continuous batching scheduler | 1.5 weeks | 3090 Ti |
| 4a | BlockManager + paged attention (pure PyTorch) | 1 week | 3090 Ti |
| 4b | CUDA paged-attention decode kernel | 3–4 weeks | 3090 Ti |
| 5 | A100 / H100 benchmark runs | 2 days | Rented |
| 6 | Writeup, README, Nsight analysis | 1 week | Local |

**Total: 10–12 weeks realistic, 8 weeks aspirational.** The original plan claimed 8 weeks; that's possible only with optimal conditions. Plan for 12, ship at 8 if it goes well.

---

## Phase detail

### Phase 0 — Skeleton (2 days)

- `pyproject.toml`, `microserve/` package, `pip install -e .` works
- HF weights for Qwen2.5-0.5B downloaded and mapped to internal parameter names
- `Sequence` dataclass: `seq_id`, `status`, prompt/output token lists, length properties

### Phase 1 — Foundation (1 week)

- RMSNorm, RoPE, GQA attention (eager — no kernel yet), SwiGLU FFN
- Flat KV cache: one tensor `[layers, 2, max_seq, n_kv_heads, head_dim]`
- Greedy decode loop, single sequence

**Exit criterion:** for 5 different prompts, microserve and `transformers` produce identical token sequences for 50+ greedy tokens.

### Phase 2 — Static batching (3 days)

- Multiple sequences padded to longest, attention mask hides padding
- Per-sequence state (positions, finished flags)

**Exit criterion:** batch of 8 of the same prompt produces the same tokens as batch of 1 repeated 8×. Padding correctness verified.

### Phase 3 — Continuous batching (1.5 weeks)

- `Scheduler` with `waiting` queue and `running` set
- State machine: `WAITING → RUNNING → FINISHED`. Skip `PREFILL`/`DECODING` split (track per-sequence by position); add `PREEMPTED` later only if you implement eviction.
- Per-step admission: each scheduler step decides who runs, builds the batch, runs one forward pass, appends tokens, retires finished
- Prefill and decode in separate steps (no chunked prefill yet)

**Exit criterion:** 32 concurrent requests with varied prompt lengths process correctly. Throughput exceeds static batching at the same VRAM.

### Phase 4a — Paged KV cache, pure PyTorch (1 week)

- `BlockManager` owns `[num_blocks, 2, block_size, n_kv_heads, head_dim]` (single allocation, block_size = 16)
- Per-sequence `block_table: list[int]` (just a field on `Sequence`, **not** a separate class)
- Lazy block allocation as sequences grow; free on finish
- Attention gathers K/V via block table using `torch.index_select` — slow but correct

**Exit criterion:** logits identical to flat-cache path on the same workload. Achievable batch size at fixed VRAM ≥ 2× the flat-cache equivalent.

### Phase 4b — CUDA paged-attention decode kernel (3–4 weeks)

The differentiating piece. Single CUDA kernel for the decode-step attention.

- Tiled attention with **FlashAttention-style online softmax** (running max + sum), no full attention matrix materialized
- Kernel signature accepts `block_table` and `context_lens` — loads K/V via indirection through the block table
- Shared-memory tiling for Q / K / V tiles
- BF16 in/out, FP32 accumulation
- Tensor-core matmul via `mma.sync` (or WMMA initially if friction is too high)
- `cp.async` for overlapping K/V loads with compute
- PyTorch binding via `torch.utils.cpp_extension`, AOT build via `setup.py`

**The 15-418 connection:** the shared-memory tiling pattern is the same one used for matmul, extended to attention. The new ideas are the **online softmax** (which keeps the working set in registers) and the **block-table indirection** (the kernel dereferences a logical position through `block_table[pos // block_size]` to find the physical KV location).

**Exit criteria:**

- **Correctness:** kernel output matches Phase 4a reference within `atol=1e-2, rtol=1e-2` (bf16), across 100 randomized sequence lengths and block-table layouts
- **Performance:** ≥ 30% of FlashAttention-2's paged-attention throughput at decode. (You won't match it; the gap is your interview talking point.)
- **Profile:** Nsight Compute report committed showing occupancy, achieved memory bandwidth, and roofline classification (memory- vs compute-bound)

### Phase 5 — Benchmarks (2 days on rented GPU)

On A100 40 GB with Llama-3-8B:

1. **Throughput vs batch size** — tokens/sec at batch ∈ {1, 4, 16, 32, 64}, compared against:
   - vLLM (the honest baseline)
   - HuggingFace `generate` with no batching (lower bound for the chart)
2. **TTFT and ITL at P50 / P95 / P99** on ShareGPT prompts at batch 32
3. **Achievable concurrency at fixed VRAM** — paged vs flat KV cache, with OOM points marked
4. **Kernel microbenchmark** — paged-attention decode time vs sequence length ∈ {512, 1024, 2048, 4096}, your kernel vs FlashAttention-2's paged kernel

Optional: one H100 session running the same suite for the Hopper portability story.

### Phase 6 — Writeup (1 week)

- **README** — 1-paragraph elevator pitch, architecture diagram, the 4 benchmark charts, a "what's missing vs production" section that *names* the things you cut and *why*
- **DESIGN.md** (3–5 pages) — scheduler algorithm, BlockManager allocation policy, one non-trivial tradeoff you wrestled with (e.g. "why block size 16 vs 32")
- **Nsight screenshots** (2–3) in the README, annotated
- **Resume bullet** (filled in):

  > Built mini-LLM inference engine from scratch in PyTorch + CUDA: continuous-batching scheduler, paged KV cache (PagedAttention), and custom tiled CUDA attention kernel with block-table indirection and FlashAttention-style online softmax. Achieved [X]% of vLLM throughput on Llama-3-8B / A100 at batch 32; documented the gap to production kernels.

---

## Directory layout

```
microserve/
├── README.md
├── PLAN.md                         # this file
├── DESIGN.md                       # Phase 6 deliverable
├── pyproject.toml                  # pip install -e .
├── Makefile                        # kernel build targets
├── microserve/                     # the package
│   ├── __init__.py
│   ├── config.py                   # ModelConfig
│   ├── sequence.py                 # Sequence + Status
│   ├── model.py                    # QwenForCausalLM + weight loader
│   ├── kv_cache.py                 # naive flat cache (Phases 1–3)
│   ├── block_manager.py            # paged KV (Phase 4) — replaces kv_cache
│   ├── scheduler.py                # Phase 3
│   ├── engine.py                   # top-level loop
│   └── kernels/
│       ├── __init__.py
│       ├── paged_attn.py           # Python wrapper
│       ├── paged_attn.cu           # the kernel
│       └── setup.py                # AOT build (sm_86 + sm_90)
├── tests/
│   ├── test_model.py               # matches HF token-for-token
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   └── test_paged_attn.py          # kernel vs Phase 4a reference
├── benchmarks/
│   ├── throughput.py
│   ├── latency.py                  # TTFT, ITL
│   ├── concurrency.py              # paged vs flat at VRAM ceiling
│   └── kernel_microbench.py        # vs FlashAttention-2
└── examples/
    └── generate.py
```

Principles behind the layout:

- **The package goes in a subdirectory** (`microserve/microserve/`) so `pip install -e .` works and `from microserve.engine import LLMEngine` resolves cleanly.
- **One file per concept**, not per noun. No separate `allocator.py` / `block_manager.py` / `block_table.py` — they're one concept.
- **`tests/` is non-negotiable.** A test that asserts the CUDA kernel matches a PyTorch reference is what lets you say "I verified correctness" in an interview.
- **No file until there's code that wants to live in it.** Empty stubs lock in premature splits.

---

## Key data structures

### `Sequence` (`microserve/sequence.py`)

- `seq_id: int` — unique request ID
- `status: Status` — `WAITING | RUNNING | FINISHED`
- `prompt_token_ids: list[int]` — fixed after creation
- `output_token_ids: list[int]` — grows each decode step
- `block_table: list[int]` — added in Phase 4; **just a field**, not a class
- Properties: `prompt_len`, `output_len`, `total_len`, `is_finished`

### `BlockManager` (`microserve/block_manager.py`)

- `kv_cache: Tensor[num_blocks, 2, block_size, n_kv_heads, head_dim]` — single contiguous allocation
- `free_blocks: deque[int]` — free pool
- `allocate(n) -> list[int]`, `free(block_ids)`, `can_allocate(n) -> bool`
- `slot_for_token(pos) -> (block_id, slot)` where `block_id = pos // block_size`, `slot = pos % block_size`
- `block_size = 16` — tunable

No separate `Allocator` class. No separate `BlockTable` class. The "block table" is `Sequence.block_table: list[int]`. The "allocator" is the `BlockManager`. One concept, one file.

### `Scheduler` (`microserve/scheduler.py`)

- `waiting: deque[Sequence]`
- `running: list[Sequence]`
- `step() -> ScheduleOutput` — returns the batch to execute this iteration, after admissions and evictions

---

## Reading list

Read in order. Start coding after #4 — the papers make more sense once you have working code.

| # | Resource | Why |
|---|---|---|
| 1 | Karpathy — *Let's Build GPT* (YouTube) | Cleanest transformer reference |
| 2 | Jay Alammar — *The Illustrated Transformer* | KV cache mental model |
| 3 | vLLM blog post (vllm.ai/blog) | More accessible than the paper |
| 4 | Orca (OSDI 2022) | Origin of continuous batching |
| 5 | vLLM (SOSP 2023) | Core reference — read twice |
| 6 | FlashAttention 1 (Dao 2022) | Online softmax algorithm — essential for the kernel |
| 7 | FlashAttention 2 (Dao 2023) | Parallelization strategy for the kernel |
| 8 | karpathy/llama2.c (GitHub) | Cleanest C inference reference (note: `llm.c` is *training*, wrong reference) |
| 9 | vLLM source — `vllm/core/block_manager.py` | Read after the paper |
| 10 | CUDA Mode lectures — FlashAttention episode | GPU-for-ML specifics |
| 11 | NVIDIA CUDA C++ Programming Guide — async copy, mma | Reference for the kernel |

---

## Cloud workflow

1. All development on the 3090 Ti via SSH.
2. Rent A100 only when the full benchmark suite is locally green at Qwen-0.5B scale.
3. One focused session (~6 hrs) covers all Llama-3-8B benchmark runs.
4. Save Nsight Compute reports to `benchmarks/profiles/` for the writeup.

---

## "Done" checklist

- [ ] All `tests/` pass (model matches HF, kernel matches reference)
- [ ] README has architecture diagram + 4 benchmark charts + "what's missing" section
- [ ] DESIGN.md exists and is honest about tradeoffs
- [ ] Nsight screenshots committed in `benchmarks/profiles/`
- [ ] Repo is public, clone-and-run works (`pip install -e . && pytest && python examples/generate.py`)
- [ ] Resume bullet filled in with real X / Y numbers

Final interview talking point:

> "I built a paged-attention inference engine in PyTorch and CUDA. The decode kernel handles block-table indirection with FlashAttention-style online softmax. On Llama-3-8B / A100 at batch 32 I get X% of vLLM throughput; the gap is FlashAttention-3, chunked prefill, and CUDA graphs, which I scoped out. Happy to walk through the scheduler or the kernel."

---

*microserve project plan · revised May 2026 · CMU → AI infra recruiting*
