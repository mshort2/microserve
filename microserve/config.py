from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool

    @classmethod
    def qwen2_5_0_5b(cls) -> "ModelConfig":
        return cls(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_layers=24,
            num_q_heads=14,
            num_kv_heads=2,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=32768,
            tie_word_embeddings=True,
            attention_bias=True,
        )
