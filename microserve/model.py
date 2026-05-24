from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from microserve.config import ModelConfig


def _rename_hf_key(hf_key: str) -> str | None:
    if hf_key == "lm_head.weight":
        return None  # tied to embed; not stored separately in our state dict
    if hf_key == "model.embed_tokens.weight":
        return "embed.weight"
    if hf_key == "model.norm.weight":
        return "norm.weight"
    if hf_key.startswith("model.layers."):
        rest = hf_key[len("model.layers.") :]
        rest = rest.replace(".self_attn.", ".attn.")
        rest = rest.replace(".input_layernorm.", ".input_norm.")
        rest = rest.replace(".post_attention_layernorm.", ".post_attn_norm.")
        return f"layers.{rest}"
    return hf_key


def load_weights(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    raw: dict[str, torch.Tensor] = {}
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No safetensors files in {path}")
    for shard in shards:
        raw.update(load_file(str(shard), device=str(device)))

    renamed: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        new_key = _rename_hf_key(k)
        if new_key is None:
            continue
        renamed[new_key] = v.to(dtype=dtype)
    return renamed


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32).to(in_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float, max_pos: int):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [B, T] -> [B, T, head_dim]
        return self.cos_cached[positions], self.sin_cached[positions]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, T, D]; cos, sin: [B, T, D]
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, T, D] -> [B, n_kv * n_rep, T, D]. Matches HF's path so sdpa picks the same backend."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        q_dim = cfg.num_q_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = x.shape
        cfg = self.cfg

        q = self.q_proj(x).view(B, T, cfg.num_q_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        n_rep = cfg.num_q_heads // cfg.num_kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class QwenForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(
            cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings
        )
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        if positions is None:
            positions = (
                torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            )
        x = self.embed(input_ids)
        cos, sin = self.rope(positions)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x @ self.embed.weight.T

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "QwenForCausalLM":
        cfg = ModelConfig.qwen2_5_0_5b()
        model = cls(cfg).to(device=device, dtype=dtype)
        weights = load_weights(model_id, dtype=dtype, device=device)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            raise RuntimeError(f"Missing weights from checkpoint: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected weights in checkpoint: {unexpected}")
        model.eval()
        return model
