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

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    @property
    def is_finished(self) -> bool:
        return self.status is Status.FINISHED
