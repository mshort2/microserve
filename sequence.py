from dataclasses import dataclass
import enum

class Status(enum.Enum):
    WAITING = 0
    IN_PROGRESS = 1
    COMPLETED = 2

@dataclass
class Sequence:
