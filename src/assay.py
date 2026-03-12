from dataclasses import dataclass


@dataclass(frozen=True)
class Config: ...

class RuntimeContext:
    config: Config
    