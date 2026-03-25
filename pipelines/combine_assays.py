from pathlib import Path
import argparse
import polars as pl

from pipelines.utils import configure_logging

configure_logging()


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


def main():
    """Combines assay parquets into single parquet."""

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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(save_path)


if __name__ == "__main__":
    main()
