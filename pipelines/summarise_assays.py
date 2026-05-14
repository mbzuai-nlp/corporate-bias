from pathlib import Path
import argparse
import math
from collections import defaultdict
from typing import Any

import polars as pl

from pipelines.utils import configure_logging

configure_logging()


FINAL_COLUMNS = [
    "assay",
    "comparison_set_id",
    "comparison_set_name",
    "entity_id",
    "entity_name",
    "model",
    "estimand",
    "estimate_mean",
    "estimate_se",
    "num_assay_instances",
    "num_samples_per_instance",
    "consensus_n_models",
    "consensus_mean",
    "consensus_se",
    "consensus_model_sd",
    "consensus_relative_favourability",
    "consensus_relative_se",
    "consensus_relative_z",
    "self_n_peer_entities",
    "self_baseline_mean",
    "self_baseline_se",
    "self_peer_entity_sd",
    "self_relative_favouritism",
    "self_relative_se",
    "self_relative_z",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assays",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
    )
    return parser.parse_args()


def family_mean_and_se(
    means: list[float],
    stds: list[float],
    n_per_instance: int,
) -> tuple[float, float]:
    k = len(means)
    family_mean = sum(means) / k
    within_var = sum(s * s for s in stds) / k

    if k == 1:
        return family_mean, math.sqrt(within_var / n_per_instance)

    mean_var = sum((m - family_mean) ** 2 for m in means) / (k - 1)
    between_var = max(0.0, mean_var - within_var / n_per_instance)
    family_se = math.sqrt(
        between_var / k
        +
        within_var / (k * n_per_instance)
    )
    return family_mean, family_se


def sample_sd(values: list[float]) -> float | None:
    n = len(values)

    if n <= 1:
        return None

    mean = sum(values) / n
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))


def baseline_stats(rows: list[dict[str, Any]]) -> tuple[int, float, float, float | None]:
    n = len(rows)
    means = [row["estimate_mean"] for row in rows]
    ses = [row["estimate_se"] for row in rows]

    baseline_mean = sum(means) / n
    baseline_sd = sample_sd(means)

    measurement_var = sum(se * se for se in ses) / (n * n)
    baseline_var = baseline_sd * baseline_sd / n if baseline_sd is not None else 0.0
    baseline_se = math.sqrt(measurement_var + baseline_var)

    return n, baseline_mean, baseline_se, baseline_sd


def relative_z(value: float, se: float) -> float | None:
    if se <= 0:
        return None

    return value / se


def build_estimates(exploded: pl.DataFrame) -> pl.DataFrame:
    rows = []

    group_cols = [
        "assay",
        "comparison_set_id",
        "comparison_set_name",
        "entity_id",
        "entity_name",
        "model",
        "estimand",
    ]

    for _, g in exploded.group_by(group_cols, maintain_order=True):
        means = g["sample_mean"].to_list()
        stds = g["sample_std"].to_list()
        n = int(g["num_samples"][0])

        estimate_mean, estimate_se = family_mean_and_se(
            means=means,
            stds=stds,
            n_per_instance=n,
        )

        rows.append(
            {
                "assay": g["assay"][0],
                "comparison_set_id": g["comparison_set_id"][0],
                "comparison_set_name": g["comparison_set_name"][0],
                "entity_id": g["entity_id"][0],
                "entity_name": g["entity_name"][0],
                "model": g["model"][0],
                "estimand": g["estimand"][0],
                "estimate_mean": estimate_mean,
                "estimate_se": estimate_se,
                "num_assay_instances": g.height,
                "num_samples_per_instance": n,
            }
        )

    return pl.DataFrame(rows)


def add_consensus_relative_fields(estimates: pl.DataFrame) -> list[dict[str, Any]]:
    records = estimates.to_dicts()

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[
            (
                row["assay"],
                row["comparison_set_id"],
                row["entity_id"],
                row["estimand"],
            )
        ].append(row)

    output = []

    for group in groups.values():
        for row in group:
            others = [
                other
                for other in group
                if other["model"] != row["model"]
            ]

            out = dict(row)

            if others:
                (
                    consensus_n_models,
                    consensus_mean,
                    consensus_se,
                    consensus_model_sd,
                ) = baseline_stats(others)

                consensus_relative_favourability = (
                    row["estimate_mean"] - consensus_mean
                )
                consensus_relative_se = math.sqrt(
                    row["estimate_se"] ** 2
                    +
                    consensus_se ** 2
                )

                out.update(
                    {
                        "consensus_n_models": consensus_n_models,
                        "consensus_mean": consensus_mean,
                        "consensus_se": consensus_se,
                        "consensus_model_sd": consensus_model_sd,
                        "consensus_relative_favourability": consensus_relative_favourability,
                        "consensus_relative_se": consensus_relative_se,
                        "consensus_relative_z": relative_z(
                            consensus_relative_favourability,
                            consensus_relative_se,
                        ),
                    }
                )
            else:
                out.update(
                    {
                        "consensus_n_models": None,
                        "consensus_mean": None,
                        "consensus_se": None,
                        "consensus_model_sd": None,
                        "consensus_relative_favourability": None,
                        "consensus_relative_se": None,
                        "consensus_relative_z": None,
                    }
                )

            output.append(out)

    return output


def add_self_relative_fields(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in records:
        groups[
            (
                row["assay"],
                row["comparison_set_id"],
                row["model"],
                row["estimand"],
            )
        ].append(row)

    output = []

    for group in groups.values():
        for row in group:
            peers = [
                peer
                for peer in group
                if peer["entity_id"] != row["entity_id"]
            ]

            out = dict(row)

            if peers:
                (
                    self_n_peer_entities,
                    self_baseline_mean,
                    self_baseline_se,
                    self_peer_entity_sd,
                ) = baseline_stats(peers)

                self_relative_favouritism = (
                    row["estimate_mean"] - self_baseline_mean
                )
                self_relative_se = math.sqrt(
                    row["estimate_se"] ** 2
                    +
                    self_baseline_se ** 2
                )

                out.update(
                    {
                        "self_n_peer_entities": self_n_peer_entities,
                        "self_baseline_mean": self_baseline_mean,
                        "self_baseline_se": self_baseline_se,
                        "self_peer_entity_sd": self_peer_entity_sd,
                        "self_relative_favouritism": self_relative_favouritism,
                        "self_relative_se": self_relative_se,
                        "self_relative_z": relative_z(
                            self_relative_favouritism,
                            self_relative_se,
                        ),
                    }
                )
            else:
                out.update(
                    {
                        "self_n_peer_entities": None,
                        "self_baseline_mean": None,
                        "self_baseline_se": None,
                        "self_peer_entity_sd": None,
                        "self_relative_favouritism": None,
                        "self_relative_se": None,
                        "self_relative_z": None,
                    }
                )

            output.append(out)

    return output


def main():
    """Builds favouritism indicator table from assay parquets."""

    args = parse_args()

    assays_dir = Path(args.assays).resolve()
    save_path = Path(args.save).resolve()

    if not assays_dir.exists():
        raise FileNotFoundError(f"Assays directory does not exist: {assays_dir}")
    if not assays_dir.is_dir():
        raise NotADirectoryError(f"Assays path is not a directory: {assays_dir}")

    parquet_paths = sorted(assays_dir.glob("*/*.parquet"))

    if not parquet_paths:
        raise FileNotFoundError(
            f"No assay parquet files found under: {assays_dir} "
            f"(expected pattern: assays_dir/{{assay}}/{{model}}.parquet)"
        )

    dfs = [pl.read_parquet(path) for path in parquet_paths]
    combined = pl.concat(dfs, how="vertical")

    exploded = (
        combined
        .explode("result")
        .select(
            "assay",
            "assay_instance_hash",
            "model",
            "comparison_set_id",
            "comparison_set_name",
            "entity_id",
            "entity_name",
            pl.col("result").struct.field("estimand").alias("estimand"),
            pl.col("result").struct.field("num_samples").alias("num_samples"),
            pl.col("result").struct.field("sample_mean").alias("sample_mean"),
            pl.col("result").struct.field("sample_std").alias("sample_std"),
        )
    )

    estimates = build_estimates(exploded)
    records = add_consensus_relative_fields(estimates)
    records = add_self_relative_fields(records)

    final = (
        pl.DataFrame(records)
        .select(FINAL_COLUMNS)
        .sort(
            "assay",
            "comparison_set_id",
            "estimand",
            "entity_name",
            "model",
        )
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(save_path)


if __name__ == "__main__":
    main()