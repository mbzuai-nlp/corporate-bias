from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import argparse
import logging
from typing import Any

import pandas as pd
import polars as pl

from pipelines.utils import configure_logging
from src.data.model import REGRESSION_EFFECT_SCHEMA


configure_logging()
logger = logging.getLogger(__name__)
logging.getLogger("rpy2").setLevel(logging.WARNING)


SCORE_ESTIMANDS = {
    ("describe-sentiment", "stance_score"),
}

HEAD_TO_HEAD_ESTIMANDS = {
    ("head-to-head", "signed_win"),
}

STEERING_ESTIMANDS = {
    ("forced-selection", "steered_to_score"),
}

R_MODEL_EFFECTS_FILE = (
    Path(__file__).resolve().parents[1] / "src" / "model_effects.R"
)

REQUIRED_MODEL_COLUMNS = [
    "score",
    "assay",
    "assay_instance_hash",
    "sample_number",
    "model",
    "comparison_set_id",
    "comparison_set_name",
    "entity_id",
    "entity_name",
]

REQUIRED_HEAD_TO_HEAD_MODEL_COLUMNS = [
    *REQUIRED_MODEL_COLUMNS,
    "opponent_entity_id",
    "ordered_pair_id",
]

REQUIRED_STEERING_MODEL_COLUMNS = [
    *REQUIRED_MODEL_COLUMNS,
    "target_entity_id",
    "target_entity_name",
    "directed_pair_id",
]

SCORE_STRING_COLUMNS = [
    "model",
    "entity_id",
    "comparison_set_id",
    "assay_instance_hash",
]

HEAD_TO_HEAD_STRING_COLUMNS = [
    *SCORE_STRING_COLUMNS,
    "opponent_entity_id",
    "ordered_pair_id",
]

STEERING_STRING_COLUMNS = [
    *SCORE_STRING_COLUMNS,
    "target_entity_id",
    "target_entity_name",
    "directed_pair_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs regression effects modelling on assay results."
    )
    parser.add_argument("--assays", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser.parse_args()


def load_assay_results(assays_dir: Path) -> pl.DataFrame:
    parquet_paths = sorted(assays_dir.glob("*/*.parquet"))
    logger.info("Loading %s assay parquet files from %s", len(parquet_paths), assays_dir)
    dfs = [pl.read_parquet(path) for path in parquet_paths]
    return pl.concat(dfs, how="vertical")


def build_score_dataframe(
    df: pl.DataFrame,
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    return (
        df.explode("measurements")
        .unnest("measurements")
        .filter(pl.col("assay") == assay)
        .filter(pl.col("measurand") == measurand)
        .rename({"value": "score"})
        .select(*REQUIRED_MODEL_COLUMNS)
    )


def build_head_to_head_dataframe(
    df: pl.DataFrame,
    assay: str,
) -> pl.DataFrame:
    return (
        df.explode("measurements")
        .unnest("measurements")
        .filter(pl.col("assay") == assay)
        .filter(pl.col("measurand").str.starts_with("beats:"))
        .with_columns(
            opponent_entity_id=pl.col("measurand").str.strip_prefix("beats:"),
            score=(2 * pl.col("value") - 1),
        )
        .with_columns(
            ordered_pair_id=pl.concat_str(
                ["entity_id", "opponent_entity_id"],
                separator=">",
            )
        )
        .select(*REQUIRED_HEAD_TO_HEAD_MODEL_COLUMNS)
    )


def build_steering_dataframe(
    df: pl.DataFrame,
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    source_columns = [col for col in REQUIRED_MODEL_COLUMNS if col != "score"]

    source_df = (
        df
        .filter(pl.col("assay") == assay)
        .select(*source_columns)
    )

    # Create all possible targets
    target_df = (
        source_df
        .select(
            "comparison_set_id",
            pl.col("entity_id").alias("target_entity_id"),
            pl.col("entity_name").alias("target_entity_name"),
        )
        .unique()
    )

    # Create all source -> target paths
    grid = (
        source_df
        .join(target_df, on="comparison_set_id", how="inner") # creates cartesian prod
        .filter(pl.col("entity_id") != pl.col("target_entity_id")) # drop self -> self
        .with_columns(
            directed_pair_id=pl.concat_str(
                ["entity_id", "target_entity_id"],
                separator=">",
            )
        )
    )

    observed = (
        df
        .filter(pl.col("assay") == assay)
        .explode("measurements")
        .unnest("measurements")
        .filter(pl.col("measurand").str.starts_with(f"{measurand}:"))
        .with_columns(
            target_entity_id=pl.col("measurand").str.strip_prefix(f"{measurand}:"),
            score=pl.col("value").cast(pl.Float64),
        )
        .select(
            *source_columns,
            "target_entity_id",
            "score",
        )
    )

    join_keys = [
        *source_columns,
        "target_entity_id",
    ]

    return (
        grid
        .join(observed, on=join_keys, how="left")
        .with_columns(
            # ensure unobserved steering paths are given 0
            score=pl.col("score").fill_null(0.0),
        )
        .select(*REQUIRED_STEERING_MODEL_COLUMNS)
    )


def prepare_for_r(
    score_df: pl.DataFrame,
    string_columns: list[str],
) -> pd.DataFrame:
    pdf = score_df.to_pandas()

    pdf["score"] = pd.to_numeric(pdf["score"], errors="raise")

    for col in string_columns:
        pdf[col] = pdf[col].astype("string")

    return pdf


def fit_lm_with_r(
    pdf: pd.DataFrame,
    r_function_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    logger.info("Calling %s with %s rows", r_function_name, len(pdf))

    ro.r["source"](str(R_MODEL_EFFECTS_FILE))
    r_fit_lm = ro.globalenv[r_function_name]

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(pdf.drop(columns=["sample_number"]))

    r_result = r_fit_lm(r_df)

    with localconverter(ro.default_converter + pandas2ri.converter):
        coefficients = ro.conversion.rpy2py(r_result.rx2("coefficients"))
        regression_statistics = ro.conversion.rpy2py(
            r_result.rx2("regression_statistics")
        )

    return coefficients, regression_statistics.iloc[0].to_dict()


def nullable_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def nullable_uint(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def normalize_regression_statistics(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "nobs": nullable_uint(stats["nobs"]),
        "rank": nullable_uint(stats["rank"]),
        "df_residual": nullable_uint(stats["df_residual"]),
        "r_squared": nullable_float(stats["r_squared"]),
        "adj_r_squared": nullable_float(stats["adj_r_squared"]),
        "sigma": nullable_float(stats["sigma"]),
        "f_statistic": nullable_float(stats["f_statistic"]),
        "f_numdf": nullable_float(stats["f_numdf"]),
        "f_dendf": nullable_float(stats["f_dendf"]),
        "f_p_value": nullable_float(stats["f_p_value"]),
        "aic": nullable_float(stats["aic"]),
        "bic": nullable_float(stats["bic"]),
    }


def coefficients_to_effects_frame(
    coefficients: pd.DataFrame,
    regression_statistics: dict[str, Any],
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    rows = []
    stats = normalize_regression_statistics(regression_statistics)

    for row in coefficients.itertuples(index=False):
        record = row._asdict()

        rows.append(
            {
                "assay": assay,
                "measurand": measurand,
                "term": str(record["term"]),
                "estimate": nullable_float(record["estimate"]),
                "std_error": nullable_float(record["std_error"]),
                "statistic": nullable_float(record["statistic"]),
                "statistic_type": "t",
                "p_value": nullable_float(record["p_value"]),
                "aliased": bool(record["aliased"]),
                "regression_statistics": stats,
            }
        )

    return pl.DataFrame(rows, schema=REGRESSION_EFFECT_SCHEMA)


def log_regression_warnings(
    effects: pl.DataFrame,
    assay: str,
    measurand: str,
) -> None:
    stats = effects.select("regression_statistics").row(0, named=True)[
        "regression_statistics"
    ]

    nobs = stats["nobs"]
    rank = stats["rank"]
    df_residual = stats["df_residual"]

    if rank == 0:
        logger.warning(
            "Regression warning for %s/%s: rank is 0",
            assay,
            measurand,
        )

    if df_residual == 0:
        logger.warning(
            "Regression warning for %s/%s: df_residual is 0",
            assay,
            measurand,
        )

    if nobs is not None and rank is not None and rank > nobs:
        logger.warning(
            "Regression warning for %s/%s: rank %s exceeds nobs %s",
            assay,
            measurand,
            rank,
            nobs,
        )

    n_aliased = effects.filter(pl.col("aliased")).height

    if n_aliased:
        logger.warning(
            "Regression warning for %s/%s: %s aliased terms",
            assay,
            measurand,
            n_aliased,
        )


def sorted_unique(series: pd.Series) -> list[str]:
    return sorted(series.astype(str).unique())


def unique_tuples(pdf: pd.DataFrame, cols: list[str]) -> list[tuple[str, ...]]:
    frame = (
        pdf[cols]
        .astype(str)
        .drop_duplicates()
        .sort_values(cols)
    )
    return list(frame.itertuples(index=False, name=None))


def expected_score_terms(pdf: pd.DataFrame) -> set[str]:
    terms = {"(Intercept)"}

    for model in sorted_unique(pdf["model"]):
        terms.add(f"model[{model}]")

    for comparison_set_id in sorted_unique(pdf["comparison_set_id"]):
        terms.add(f"comparison_set_id[{comparison_set_id}]")

    for model, comparison_set_id in unique_tuples(
        pdf,
        ["model", "comparison_set_id"],
    ):
        terms.add(f"model[{model}]:comparison_set_id[{comparison_set_id}]")

    for comparison_set_id, entity_id in unique_tuples(
        pdf,
        ["comparison_set_id", "entity_id"],
    ):
        terms.add(f"entity_id[{entity_id}]|comparison_set_id[{comparison_set_id}]")

    for comparison_set_id, assay_instance_hash in unique_tuples(
        pdf,
        ["comparison_set_id", "assay_instance_hash"],
    ):
        terms.add(
            f"assay_instance_hash[{assay_instance_hash}]"
            f"|comparison_set_id[{comparison_set_id}]"
        )

    for model, comparison_set_id, entity_id in unique_tuples(
        pdf,
        ["model", "comparison_set_id", "entity_id"],
    ):
        terms.add(
            f"model[{model}]:entity_id[{entity_id}]"
            f"|comparison_set_id[{comparison_set_id}]"
        )

    return terms


def expected_head_to_head_terms(pdf: pd.DataFrame) -> set[str]:
    terms = {"(Intercept)"}

    for model in sorted_unique(pdf["model"]):
        terms.add(f"model[{model}]")

    for comparison_set_id in sorted_unique(pdf["comparison_set_id"]):
        terms.add(f"comparison_set_id[{comparison_set_id}]")

    for model, comparison_set_id in unique_tuples(
        pdf,
        ["model", "comparison_set_id"],
    ):
        terms.add(f"model[{model}]:comparison_set_id[{comparison_set_id}]")

    for comparison_set_id, ordered_pair_id in unique_tuples(
        pdf,
        ["comparison_set_id", "ordered_pair_id"],
    ):
        terms.add(f"beats[{ordered_pair_id}]|comparison_set_id[{comparison_set_id}]")

    for comparison_set_id, assay_instance_hash in unique_tuples(
        pdf,
        ["comparison_set_id", "assay_instance_hash"],
    ):
        terms.add(
            f"assay_instance_hash[{assay_instance_hash}]"
            f"|comparison_set_id[{comparison_set_id}]"
        )

    for model, comparison_set_id, ordered_pair_id in unique_tuples(
        pdf,
        ["model", "comparison_set_id", "ordered_pair_id"],
    ):
        terms.add(
            f"model[{model}]:beats[{ordered_pair_id}]"
            f"|comparison_set_id[{comparison_set_id}]"
        )

    return terms


def expected_steering_terms(pdf: pd.DataFrame) -> set[str]:
    terms = {"(Intercept)"}

    for model in sorted_unique(pdf["model"]):
        terms.add(f"model[{model}]")

    for comparison_set_id in sorted_unique(pdf["comparison_set_id"]):
        terms.add(f"comparison_set_id[{comparison_set_id}]")

    for model, comparison_set_id in unique_tuples(
        pdf,
        ["model", "comparison_set_id"],
    ):
        terms.add(f"model[{model}]:comparison_set_id[{comparison_set_id}]")

    for comparison_set_id, directed_pair_id in unique_tuples(
        pdf,
        ["comparison_set_id", "directed_pair_id"],
    ):
        terms.add(
            f"steered[{directed_pair_id}]|comparison_set_id[{comparison_set_id}]"
        )

    for comparison_set_id, assay_instance_hash in unique_tuples(
        pdf,
        ["comparison_set_id", "assay_instance_hash"],
    ):
        terms.add(
            f"assay_instance_hash[{assay_instance_hash}]"
            f"|comparison_set_id[{comparison_set_id}]"
        )

    for model, comparison_set_id, directed_pair_id in unique_tuples(
        pdf,
        ["model", "comparison_set_id", "directed_pair_id"],
    ):
        terms.add(
            f"model[{model}]:steered[{directed_pair_id}]"
            f"|comparison_set_id[{comparison_set_id}]"
        )

    return terms


def validate_effects(
    effects: pl.DataFrame,
    pdf: pd.DataFrame,
    expected_terms_fn: Callable[[pd.DataFrame], set[str]],
) -> None:
    observed = effects.get_column("term").to_list()
    observed_set = set(observed)
    expected_set = expected_terms_fn(pdf)

    duplicate_terms = sorted(
        term for term in observed_set if observed.count(term) > 1
    )
    missing_terms = sorted(expected_set - observed_set)
    unexpected_terms = sorted(observed_set - expected_set)

    if duplicate_terms:
        raise ValueError(f"Duplicate regression effect terms: {duplicate_terms}")

    if missing_terms:
        raise ValueError(f"Missing regression effect terms: {missing_terms}")

    if unexpected_terms:
        raise ValueError(f"Unexpected regression effect terms: {unexpected_terms}")

    if len(observed) != len(expected_set):
        raise ValueError(
            f"Expected {len(expected_set)} effects, found {len(observed)}"
        )


def fit_score_estimand(
    combined: pl.DataFrame,
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    logger.info("Fitting score estimand %s/%s", assay, measurand)
    score_df = build_score_dataframe(combined, assay, measurand)
    pdf = prepare_for_r(score_df, SCORE_STRING_COLUMNS)

    coefficients, regression_statistics = fit_lm_with_r(pdf, "fit_score_lm")

    effects = coefficients_to_effects_frame(
        coefficients=coefficients,
        regression_statistics=regression_statistics,
        assay=assay,
        measurand=measurand,
    )

    log_regression_warnings(effects, assay, measurand)
    validate_effects(effects, pdf, expected_score_terms)
    logger.info("Finished score estimand %s/%s with %s effects", assay, measurand, effects.height)

    return effects


def fit_head_to_head_estimand(
    combined: pl.DataFrame,
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    logger.info("Fitting head-to-head estimand %s/%s", assay, measurand)
    score_df = build_head_to_head_dataframe(combined, assay)
    pdf = prepare_for_r(score_df, HEAD_TO_HEAD_STRING_COLUMNS)

    coefficients, regression_statistics = fit_lm_with_r(
        pdf,
        "fit_head_to_head_lpm",
    )

    effects = coefficients_to_effects_frame(
        coefficients=coefficients,
        regression_statistics=regression_statistics,
        assay=assay,
        measurand=measurand,
    )

    log_regression_warnings(effects, assay, measurand)
    validate_effects(effects, pdf, expected_head_to_head_terms)
    logger.info("Finished head-to-head estimand %s/%s with %s effects", assay, measurand, effects.height)

    return effects


def fit_steering_estimand(
    combined: pl.DataFrame,
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    logger.info("Fitting steering estimand %s/%s", assay, measurand)
    score_df = build_steering_dataframe(combined, assay, measurand)
    pdf = prepare_for_r(score_df, STEERING_STRING_COLUMNS)

    coefficients, regression_statistics = fit_lm_with_r(
        pdf,
        "fit_steering_lm",
    )

    effects = coefficients_to_effects_frame(
        coefficients=coefficients,
        regression_statistics=regression_statistics,
        assay=assay,
        measurand=measurand,
    )

    log_regression_warnings(effects, assay, measurand)
    validate_effects(effects, pdf, expected_steering_terms)
    logger.info("Finished steering estimand %s/%s with %s effects", assay, measurand, effects.height)

    return effects


def main() -> None:
    args = parse_args()

    assays_dir = Path(args.assays).resolve()
    save_path = Path(args.save_path).resolve()

    combined = load_assay_results(assays_dir)
    logger.info("Loaded combined assay results with %s rows", combined.height)

    effects = pl.concat(
        [
            # *[
            #     fit_score_estimand(combined, assay, measurand)
            #     for assay, measurand in sorted(SCORE_ESTIMANDS)
            # ],
            # *[
            #     fit_head_to_head_estimand(combined, assay, measurand)
            #     for assay, measurand in sorted(HEAD_TO_HEAD_ESTIMANDS)
            # ],
            *[
                fit_steering_estimand(combined, assay, measurand)
                for assay, measurand in sorted(STEERING_ESTIMANDS)
            ],
        ],
        how="vertical",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    effects.write_parquet(save_path)

    logger.info("Wrote regression effects to %s", save_path)


if __name__ == "__main__":
    main()