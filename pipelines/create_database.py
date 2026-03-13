from pathlib import Path
import argparse
import json
import polars as pl
from typing import Mapping
import hashlib

from pipelines.utils import configure_logging

configure_logging()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authored",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
    )
    return parser.parse_args()


def stable_u64_hash(value) -> int:
    s = json.dumps(value, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def load_authored(dir: Path) -> Mapping[str, pl.DataFrame]:
    out = {}

    with open(dir / "claim.json", "r") as f:
        d = json.load(f)
        out["claim"] = pl.from_dicts(d)

    with open(dir / "comparison_set_assay_instance.json", "r") as f:
        d = json.load(f)
        out["comparison_set_assay_instance"] = (
            pl.from_dicts(d)
            .explode("instances")
            .rename({"instances": "instance"})
            .with_columns(
                pl.col("comparison_set_id"),
                pl.col("comparison_set_name"),
                pl.col("assay"),
                pl.col("instance")
                    .map_elements(stable_u64_hash, return_dtype=pl.UInt64)
                    .alias("instance_hash"),
                pl.col("instance")
                    .map_elements(
                        lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
                        return_dtype=pl.String,
                    )
                    .alias("instance_json"),
            )
            .select(
                "comparison_set_id",
                "comparison_set_name",
                "assay",
                "instance_hash",
                "instance_json",
            )
        )

    with open(dir / "comparison_set_link.json", "r") as f:
        d = json.load(f)
        out["comparison_set_link"] = pl.from_dicts(d)

    with open(dir / "entity.json", "r") as f:
        d = json.load(f)
        out["entity"] = pl.from_dicts(d)

    return out


def write_db(dfs: Mapping[str, pl.DataFrame], dir: Path):
    dfs["claim"].write_parquet(dir / "claim.parquet")
    dfs["comparison_set_assay_instance"].write_parquet(
        dir / "comparison_set_assay_instance.parquet"
    )
    dfs["comparison_set_link"].write_parquet(dir / "comparison_set_link.parquet")
    dfs["entity"].write_parquet(dir / "entity.parquet")


def main():
    """Creates a database of corporate entities, claims about these entities
    (e.g. ownership, partnership) and groupings of entities defining comparison sets
    (e.g. email providers, car brands).
    """

    args = parse_args()

    authored_dir = Path(args.authored).resolve()
    db_dir = Path(args.db).resolve()

    authored = load_authored(authored_dir)

    write_db(authored, db_dir)


if __name__ == "__main__":
    main()
