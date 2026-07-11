import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol
from dvclive import Live

from src.data import Db


# === CONSTANTS ===


JUDGE_MODELS = ["gpt-5.4-mini", "gemini-2.5-flash", "qwen3.7-plus"]


# === TYPES ===


@dataclass(frozen=True)
class Config:
    save: str
    assay: str
    model: str


@dataclass
class RuntimeContext:
    cfg: Config
    exp: Live
    assay_db: Db


class AssayDelegate(Protocol):
    def __call__(self, ctx: RuntimeContext) -> pl.DataFrame: ...


# === UTILS ===


def save_assay_df(df: pl.DataFrame, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
