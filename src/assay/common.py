import polars as pl
from pathlib import Path
import statistics
import json
from dataclasses import dataclass
from typing import Mapping, Literal, Protocol, Tuple
from dvclive import Live
import math

from src.model import Model


# === TYPES ===


Assay = Literal[
    "head-to-head", 
    "rank", 
    "consideration-set", 
    "describe-sentiment",
    "forced-selection"
]


@dataclass(frozen=True)
class Config:
    save: str
    assay: Assay
    model: Model
    num_samples_per_instance: int


@dataclass
class RuntimeContext:
    cfg: Config
    exp: Live
    db: Mapping[str, pl.DataFrame]


class AssayDelegate(Protocol):
    def __call__(self, ctx: RuntimeContext) -> pl.DataFrame: ...


# === UTILS ===


def save_assay_df(df: pl.DataFrame, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def build_entity_lookup(entity_df: pl.DataFrame) -> dict[str, dict]:
    return {row["id"]: row for row in entity_df.iter_rows(named=True)}


def get_comparison_set_entities(
    comparison_set_df: pl.DataFrame,
    entity_lookup: dict[str, dict],
    comparison_set_id: str,
) -> list[dict]:
    entity_ids = (
        comparison_set_df.filter(pl.col("id") == comparison_set_id)
        .select("entity_ids")
        .to_series()
        .item()
    )

    entities = []
    for entity_id in sorted(entity_ids):
        entity = entity_lookup[entity_id]
        entities.append(
            {
                "entity_id": entity["id"],
                "entity_name": entity["name"],
                "aliases": entity["aliases"],
            }
        )

    return entities


def mean_coalesce(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def std_coalesce(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def sem_coalesce(values: list[float]) -> float:
    if not values:
        return float("nan")
    return std_coalesce(values) / math.sqrt(len(values))


def build_estimand_result(
    estimand: str, values: list[float]
) -> Tuple[dict[str, str]]:
    return ({
        "estimand": estimand,
        "num_samples": len(values),
        "sample_mean": mean_coalesce(values),
        "sample_std": std_coalesce(values) 
    },)
