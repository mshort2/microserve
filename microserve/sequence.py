from dataclasses import dataclass, field
import enum
import itertools


class Status(enum.Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


_id_counter = itertools.count()


@dataclass
class Sequence:
    prompt_token_ids: list[int]
    seq_id: int = field(default_factory=lambda: next(_id_counter))
    output_token_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    block_table: list[int] = field(default_factory=list)
    # Per-request stop condition (used by the scheduler).
    max_new_tokens: int = 50
    # Index into the engine's KV-cache slot pool. -1 until admitted to running.
    slot_idx: int = -1

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def context_len(self) -> int:
        """Number of K/V entries currently in the cache for this sequence.

        After prefill writes `prompt_len` tokens and we sample the first output
        token: context_len = prompt_len, output_len = 1. Each subsequent decode
        writes one more K/V before sampling, so context_len keeps pace at
        prompt_len + output_len - 1 once output_len >= 1.
        """
        if self.output_len == 0:
            return 0
        return self.prompt_len + self.output_len - 1

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    @property
    def is_finished(self) -> bool:
        return self.status is Status.FINISHED
