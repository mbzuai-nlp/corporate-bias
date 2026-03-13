from pathlib import Path
import argparse
import json
import polars as pl
from typing import Mapping
from src.data.model import (
    CLAIM_SCHEMA,
    COMPARISON_SET_ASSAY_ARGS_SCHEMA,
    COMPARISON_SET_LINK_SCHEMA,
    ENTITY_SCHEMA,
)

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


def load_authored(dir: Path) -> Mapping[str, pl.DataFrame]:
    out = {}

    with open(dir / "claim.json", "r") as f:
        d = json.load(f)
        out["claim"] = pl.from_dicts(d, schema=CLAIM_SCHEMA)

    with open(dir / "comparison_set_assay_args.json", "r") as f:
        d = json.load(f)
        out["comparison_set_assay_args"] = (
            pl.from_dicts(d, schema=COMPARISON_SET_ASSAY_ARGS_SCHEMA)
        )

    with open(dir / "comparison_set_link.json", "r") as f:
        d = json.load(f)
        out["comparison_set_link"] = pl.from_dicts(d, schema=COMPARISON_SET_LINK_SCHEMA)

    with open(dir / "entity.json", "r") as f:
        d = json.load(f)
        out["entity"] = pl.from_dicts(d, schema=ENTITY_SCHEMA)

    return out


def write_db(dfs: Mapping[str, pl.DataFrame], dir: Path):
    dfs["claim"].write_parquet(dir / "claim.parquet")
    dfs["comparison_set_assay_args"].write_parquet(
        dir / "comparison_set_assay_args.parquet"
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
