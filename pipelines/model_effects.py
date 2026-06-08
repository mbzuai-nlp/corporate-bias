from __future__ import annotations

from pathlib import Path
import argparse
import logging
import re
import statsmodels.formula.api as smf
import pandas as pd
import polars as pl
from typing import Any, Callable
from collections import namedtuple

from pipelines.utils import configure_logging
from src.data.model import REGRESSION_EFFECT_SCHEMA


configure_logging()
logger = logging.getLogger(__name__)


SCORE_ESTIMANDS = (
    ("describe-sentiment", "sentiment_score"),
    ("describe-sentiment", "stance_score"),
    ("describe-sentiment", "promotional_likelihood"),
    ("forced-selection", "retention_score"),
    ("forced-selection", "selected"),
    ("consideration-set", "recommendation_score"),
)

HEAD_TO_HEAD_ASSAY = "head-to-head"

US_ENTITIES = [
    "gemini",
    "google-chrome",
    "supergrok",
    "nvidia",
    "grok",
    "anthropic",
    "microsoft-365",
    "gmail",
    "codex",
    "openai",
    "microsoft-edge",
    "outlook",
    "gpt",
    "gemini-code-assist",
    "google-workspace",
    "windsurf",
    "firefox",
    "amazon-web-services",
    "claude-code",
    "microsoft",
    "phi",
    "yahoo-mail",
    "microsoft-azure",
    "xai",
    "safari",
    "cursor",
    "nemotron",
    "claude",
    "google-cloud-platform",
    "meta",
    "google",
    "icloud-mail",
    "llama",
    "github-copilot",
]

CHINA_ENTITIES = [
    "alimail",
    "qwen",
    "qq-mail",
    "qwen-code",
    "alibaba",
    "deepseek",
]

EUROPE_ENTITIES = [
    "mistral",
    "mistral-code",
    "mistral-vibe",
    "proton-mail",
]

US_MODELS = [
    "grok-4.1-fast",
    "grok-4",
    "gpt-5.4",
    "gpt-oss-120b",
    "gpt-4o-mini",
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct",
    "nemotron-3-super-120b-a12b",
    "phi-4",
]

CHINA_MODELS = [
    "deepseek-v3.2",
    "qwen3-235b-a22b-2507",
    "qwen3.5-flash-02-23",
]

EUROPE_MODELS = [
    "mistral-nemo",
    "mistral-small-2603",
]


Effect = namedtuple("Effect", ("name", "estimate", "std_error", "p_value", "t_value"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs regression effects modelling on assay results."
    )
    parser.add_argument("--assays", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser.parse_args()


def affiliated_expr(model: pl.Expr, entity: pl.Expr) -> pl.Expr:
    return (
        (
            model.is_in(["grok-4.1-fast", "grok-4"])
            & entity.is_in(["grok", "supergrok", "xai"])
        )
        | (
            model.eq("deepseek-v3.2")
            & entity.eq("deepseek")
        )
        | (
            model.is_in(["gpt-5.4", "gpt-oss-120b", "gpt-4o-mini"])
            & entity.is_in(["openai", "gpt", "codex"])
        )
        | (
            model.is_in(["claude-sonnet-4.6", "claude-opus-4.6"])
            & entity.is_in(["anthropic", "claude", "claude-code"])
        )
        | (
            model.is_in(["gemini-2.5-pro", "gemini-2.5-flash"])
            & entity.is_in(
                [
                    "gemini",
                    "google",
                    "google-chrome",
                    "gmail",
                    "gemini-code-assist",
                    "google-workspace",
                    "google-cloud-platform",
                ]
            )
        )
        | (
            model.is_in(["llama-3.1-8b-instruct", "llama-3.1-70b-instruct"])
            & entity.is_in(["llama", "meta"])
        )
        | (
            model.eq("nemotron-3-super-120b-a12b")
            & entity.is_in(["nemotron", "nvidia"])
        )
        | (
            model.is_in(["qwen3-235b-a22b-2507", "qwen3.5-flash-02-23"])
            & entity.is_in(["qwen", "qwen-code", "alibaba", "alimail"])
        )
        | (
            model.is_in(["mistral-nemo", "mistral-small-2603"])
            & entity.is_in(["mistral", "mistral-code", "mistral-vibe"])
        )
        | (
            model.eq("mistral-nemo")
            & entity.eq("nvidia")
        )
        | (
            model.eq("phi-4")
            & entity.is_in(
                [
                    "phi",
                    "microsoft",
                    "microsoft-365",
                    "microsoft-edge",
                    "outlook",
                    "microsoft-azure",
                    "github-copilot",
                ]
            )
        )
    )


def geo_associated_expr(model: pl.Expr, entity: pl.Expr) -> pl.Expr:
    return (
        (
            model.is_in(US_MODELS)
            & entity.is_in(US_ENTITIES)
        )
        | (
            model.is_in(CHINA_MODELS)
            & entity.is_in(CHINA_ENTITIES)
        )
        | (
            model.is_in(EUROPE_MODELS)
            & entity.is_in(EUROPE_ENTITIES)
        )
    )


def load_assay_results(assays_dir: Path) -> pl.DataFrame:
    parquet_paths = sorted(assays_dir.glob("*/*.parquet"))
    logger.info(
        "Loading %s assay parquet files from %s", len(parquet_paths), assays_dir
    )
    dfs = [pl.read_parquet(path) for path in parquet_paths]

    return (
        pl.concat(dfs, how="vertical")
        .explode("measurements")
        .unnest("measurements")
        .with_columns(
            [
                affiliated_expr(pl.col("model"), pl.col("entity_id")).alias(
                    "affiliated"
                ),
                geo_associated_expr(pl.col("model"), pl.col("entity_id")).alias(
                    "geo_associated"
                ),
            ]
        )
    )


def empty_effects_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=REGRESSION_EFFECT_SCHEMA)


def prepare_regression_obs(obs: pl.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    obs = obs.to_pandas()

    for col in categorical_cols:
        obs[col] = obs[col].astype("category")

    obs["score"] = pd.to_numeric(obs["score"]).astype(float)
    obs["affiliated"] = pd.to_numeric(obs["affiliated"]).astype(float)
    obs["geo_associated"] = pd.to_numeric(obs["geo_associated"]).astype(float)

    return obs


def compute_implied_effect(
    m: Any,
    name: str,
    terms: list[str],
) -> Effect:
    # construct all zeros, indexed by m's coefficients
    r = pd.Series(0.0, index=m.params.index)
    # set the explicit levels to -1, which yields the coeef of the implied entity
    # since it is captured by the -sum of the explicit coeffs
    r.loc[terms] = -1.0
    # t_test also gives correct standard error
    # as opposed to -sum(m.params[explicit.values()])
    t = m.t_test(r.to_numpy())

    return Effect(
        name=name,
        estimate=float(t.effect.item()),
        std_error=float(t.sd.item()),
        p_value=float(t.pvalue),
        t_value=float(t.tvalue.item()),
    )


def compute_numeric_effect(m: Any, name: str) -> Effect:
    return Effect(
        name=name,
        estimate=float(m.params[name]),
        std_error=float(m.bse[name]),
        p_value=float(m.pvalues[name]),
        t_value=float(m.tvalues[name]),
    )


def compute_factor_effects(
    m: Any, obs: pd.DataFrame, lookup: str
) -> tuple[Effect, ...]:
    factor = re.match(r"^\^?C\\\(([^,]+), Sum\\\)", lookup).group(1)

    matched = (
        m.params.index.to_series()
        .str.extract(lookup)[0]
        .dropna()
    )

    explicit = {
        level: name
        for name, level in matched.items()
    }

    levels = [str(level) for level in obs[factor].cat.categories.tolist()]

    implied_level_name = next(
        level
        for level in levels
        if level not in explicit
    )

    implied_effect = compute_implied_effect(
        m,
        implied_level_name,
        list(explicit.values()),
    )

    return tuple(
        Effect(
            name=level,
            estimate=float(m.params[explicit[level]]),
            std_error=float(m.bse[explicit[level]]),
            p_value=float(m.pvalues[explicit[level]]),
            t_value=float(m.tvalues[explicit[level]]),
        )
        if level in explicit
        else implied_effect
        for level in levels
    )


def add_sum_contrast_columns(
    obs: pd.DataFrame,
    left_col: str,
    right_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    levels = sorted(
        set(obs[left_col].astype(str).tolist())
        | set(obs[right_col].astype(str).tolist())
    )
    implied_level = levels[-1]
    left = obs[left_col].astype(str)
    right = obs[right_col].astype(str)

    columns = {}

    for i, level in enumerate(levels[:-1]):
        col = f"{prefix}_{i}"
        obs[col] = (
            (left == level).astype(float)
            - (left == implied_level).astype(float)
            - (right == level).astype(float)
            + (right == implied_level).astype(float)
        )
        columns[level] = col

    return obs, levels, columns


def compute_sum_contrast_effects(
    m: Any,
    levels: list[str],
    columns: dict[str, str],
) -> tuple[Effect, ...]:
    implied_effect = compute_implied_effect(
        m,
        levels[-1],
        list(columns.values()),
    )

    return tuple(
        compute_numeric_effect(m, columns[level])
        ._replace(name=level)
        if level in columns
        else implied_effect
        for level in levels
    )


def varies_within_model(obs: pd.DataFrame, col: str) -> bool:
    return (
        obs
        .groupby("model", observed=True)[col]
        .nunique()
        .gt(1)
        .any()
    )


def prepare_model_affiliated_deviations(
    obs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    model_levels = [str(level) for level in obs["model"].cat.categories.tolist()]
    eligible_affiliated_models = [
        level
        for level in model_levels
        if obs.loc[obs["model"].astype(str) == level, "affiliated"].abs().sum() > 0
    ]

    model_affiliated_deviation_columns = {}

    if len(eligible_affiliated_models) > 1:
        implied_affiliated_model = eligible_affiliated_models[-1]
        model = obs["model"].astype(str)

        for i, level in enumerate(eligible_affiliated_models[:-1]):
            col = f"model_affiliated_deviation_{i}"
            obs[col] = obs["affiliated"] * (
                (model == level).astype(float)
                - (model == implied_affiliated_model).astype(float)
            )
            model_affiliated_deviation_columns[level] = col

    return obs, eligible_affiliated_models, model_affiliated_deviation_columns


def compute_model_affiliated_deviation_effects(
    m: Any,
    eligible_affiliated_models: list[str],
    model_affiliated_deviation_columns: dict[str, str],
) -> tuple[Effect, ...]:
    model_affiliated_deviation_effects = []

    if len(eligible_affiliated_models) > 1:
        for level in eligible_affiliated_models[:-1]:
            col = model_affiliated_deviation_columns[level]
            model_affiliated_deviation_effects.append(
                Effect(
                    name=level,
                    estimate=float(m.params[col]),
                    std_error=float(m.bse[col]),
                    p_value=float(m.pvalues[col]),
                    t_value=float(m.tvalues[col]),
                )
            )

        model_affiliated_deviation_effects.append(
            compute_implied_effect(
                m,
                eligible_affiliated_models[-1],
                list(model_affiliated_deviation_columns.values()),
            )
        )

    model_affiliated_deviation_effects = tuple(model_affiliated_deviation_effects)

    if abs(sum(effect.estimate for effect in model_affiliated_deviation_effects)) > 1e-10:
        raise Exception("Model affiliated deviation effects do not sum to zero.")

    return model_affiliated_deviation_effects


def fit_prepared_estimand(
    obs: pd.DataFrame,
    comparison_set_id: str,
    assay: str,
    measurand: str,
    entity_formula_terms: list[str],
    compute_entity_effects: Callable[[Any], tuple[Effect, ...]],
) -> pl.DataFrame:
    logger.info(
        "Regressing assay=%s, measurand=%s, set=%s",
        assay,
        measurand,
        comparison_set_id,
    )

    include_affiliated = varies_within_model(obs, "affiliated")
    include_geo_associated = varies_within_model(obs, "geo_associated")

    if include_affiliated:
        (
            obs,
            eligible_affiliated_models,
            model_affiliated_deviation_columns,
        ) = prepare_model_affiliated_deviations(obs)
    else:
        eligible_affiliated_models = []
        model_affiliated_deviation_columns = {}

    formula_terms = [
        "C(model, Sum)",
        *entity_formula_terms,
        "C(assay_instance_hash, Sum)",
    ]

    if include_affiliated:
        formula_terms.append("affiliated")
        formula_terms.extend(model_affiliated_deviation_columns.values())

    if include_geo_associated:
        formula_terms.append("geo_associated")

    formula = "score ~ " + " + ".join(formula_terms)

    m = smf.ols(formula, data=obs).fit()

    if m.model.rank < len(m.params):
        logger.warning(
            (
                "Regression design is rank deficient for assay=%s, measurand=%s, "
                "set=%s: rank=%s, params=%s"
            ),
            assay,
            measurand,
            comparison_set_id,
            int(m.model.rank),
            len(m.params),
        )

    model_effects = compute_factor_effects(
        m,
        obs,
        r"^C\(model, Sum\)\[S\.([^\]]+)\]$",
    )
    entity_effects = compute_entity_effects(m)
    prompt_effects = compute_factor_effects(
        m,
        obs,
        r"^C\(assay_instance_hash, Sum\)\[S\.([^\]]+)\]$",
    )
    model_affiliated_deviation_effects = compute_model_affiliated_deviation_effects(
        m,
        eligible_affiliated_models,
        model_affiliated_deviation_columns,
    )

    affiliated_effect = (
        compute_numeric_effect(m, "affiliated")
        if include_affiliated
        else None
    )

    geo_associated_effect = (
        compute_numeric_effect(m, "geo_associated")
        if include_geo_associated
        else None
    )

    row = {
        "assay": assay,
        "measurand": measurand,
        "comparison_set_id": comparison_set_id,
        "model_effects": [effect._asdict() for effect in model_effects],
        "entity_effects": [effect._asdict() for effect in entity_effects],
        "prompt_effects": [effect._asdict() for effect in prompt_effects],
        "model_affiliated_deviation_effects": [
            effect._asdict()
            for effect in model_affiliated_deviation_effects
        ],
        "affiliated_effect": (
            affiliated_effect._asdict()
            if affiliated_effect is not None
            else None
        ),
        "geo_associated_effect": (
            geo_associated_effect._asdict()
            if geo_associated_effect is not None
            else None
        ),
        "regression_statistics": {
            "nobs": int(m.nobs),
            "rank": int(m.model.rank),
            "df_residual": int(m.df_resid),
            "r_squared": float(m.rsquared),
            "adj_r_squared": float(m.rsquared_adj),
            "sigma": float(m.mse_resid ** 0.5),
            "f_statistic": float(m.fvalue),
            "f_numdf": float(m.df_model),
            "f_dendf": float(m.df_resid),
            "f_p_value": float(m.f_pvalue),
            "aic": float(m.aic),
            "bic": float(m.bic),
        },
    }

    return pl.DataFrame([row], schema=REGRESSION_EFFECT_SCHEMA)


def fit_score_estimand(
    obs: pl.DataFrame,
    comparison_set_id: str,
    assay: str,
    measurand: str,
) -> pl.DataFrame:
    obs = (
        obs
        .filter(pl.col("comparison_set_id") == comparison_set_id)
        .filter(pl.col("assay") == assay)
        .filter(pl.col("measurand") == measurand)
        .rename({"value": "score"})
        .select(
            "score",
            "assay_instance_hash",
            "model",
            "entity_id",
            "affiliated",
            "geo_associated",
        )
    )

    if obs.height == 0:
        return empty_effects_frame()

    obs = prepare_regression_obs(
        obs,
        [
            "model",
            "entity_id",
            "assay_instance_hash",
        ],
    )

    return fit_prepared_estimand(
        obs=obs,
        comparison_set_id=comparison_set_id,
        assay=assay,
        measurand=measurand,
        entity_formula_terms=["C(entity_id, Sum)"],
        compute_entity_effects=lambda m: compute_factor_effects(
            m,
            obs,
            r"^C\(entity_id, Sum\)\[S\.([^\]]+)\]$",
        ),
    )


def fit_head_to_head_assay(
    obs: pl.DataFrame,
    comparison_set_id: str,
) -> pl.DataFrame:
    obs = (
        obs
        .filter(pl.col("comparison_set_id") == comparison_set_id)
        .filter(pl.col("assay") == HEAD_TO_HEAD_ASSAY)
        .with_columns(
            pl.col("measurand").str.replace(r"^beats:", "").alias("right_entity_id")
        )
        .with_columns(
            [
                (
                    affiliated_expr(pl.col("model"), pl.col("entity_id")).cast(pl.Float64)
                    - affiliated_expr(
                        pl.col("model"), pl.col("right_entity_id")
                    ).cast(pl.Float64)
                ).alias("affiliated"),
                (
                    geo_associated_expr(
                        pl.col("model"), pl.col("entity_id")
                    ).cast(pl.Float64)
                    - geo_associated_expr(
                        pl.col("model"), pl.col("right_entity_id")
                    ).cast(pl.Float64)
                ).alias("geo_associated"),
            ]
        )
        .rename(
            {
                "value": "score",
                "entity_id": "left_entity_id",
            }
        )
        .select(
            "score",
            "assay_instance_hash",
            "model",
            "left_entity_id",
            "right_entity_id",
            "affiliated",
            "geo_associated",
        )
    )

    if obs.height == 0:
        return empty_effects_frame()
    
    obs = prepare_regression_obs(
        obs,
        [
            "model",
            "left_entity_id",
            "right_entity_id",
            "assay_instance_hash",
        ],
    )
    obs, entity_levels, entity_contrast_columns = add_sum_contrast_columns(
        obs,
        left_col="left_entity_id",
        right_col="right_entity_id",
        prefix="entity_contrast",
    )

    return fit_prepared_estimand(
        obs=obs,
        comparison_set_id=comparison_set_id,
        assay=HEAD_TO_HEAD_ASSAY,
        measurand="beats",
        entity_formula_terms=list(entity_contrast_columns.values()),
        compute_entity_effects=lambda m: compute_sum_contrast_effects(
            m,
            entity_levels,
            entity_contrast_columns,
        ),
    )


def main() -> None:
    args = parse_args()

    assays_dir = Path(args.assays).resolve()
    save_path = Path(args.save_path).resolve()

    assay_results = load_assay_results(assays_dir)
    logger.info("Loaded combined assay results with %s rows", assay_results.height)

    comparison_set_ids = sorted(
        assay_results
        .unique("comparison_set_id")
        .select("comparison_set_id")
        .to_series()
        .to_list()
    )

    effects = pl.concat(
        [
            *[
                fit_score_estimand(assay_results, comparison_set_id, assay, measurand)
                for assay, measurand in sorted(SCORE_ESTIMANDS)
                for comparison_set_id in comparison_set_ids
            ],
            *[
                fit_head_to_head_assay(assay_results, comparison_set_id)
                for comparison_set_id in comparison_set_ids
            ],
        ],
        how="vertical",
    )

    if effects.schema != REGRESSION_EFFECT_SCHEMA:
        raise Exception("Effects schema is unexpected.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    effects.write_parquet(save_path)

    logger.info("Wrote regression effects to %s", save_path)


if __name__ == "__main__":
    main()