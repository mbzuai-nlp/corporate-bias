from scipy import stats
from pathlib import Path
import argparse
import logging
import polars as pl
from dvclive import Live
import patsy
import statsmodels.api as sm
import numpy as np
from typing import Tuple
import pandas as pd
import sympy

from src.data import load_db, Db
from src.assay.common import JUDGE_MODELS
from pipelines.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class InvalidDesignMatrixException(RuntimeError): ...


def validate_design_matrix(X):
    M = sympy.Matrix(X.astype(int).tolist())
    null_vecs = M.nullspace()
    rank = X.shape[1] - len(null_vecs)
    if rank == X.shape[1]:
        return
    col_names = X.design_info.column_names
    print(f"Design matrix rank: {rank}, shape: {X.shape}")
    print("Linear dependencies:")
    for vec in null_vecs:
        terms = []
        for j, coef in enumerate(vec):
            if coef != 0:
                col = col_names[j]
                if coef.is_Integer:
                    coef = int(coef)
                    if coef == 1:
                        terms.append(col)
                    elif coef == -1:
                        terms.append(f"-{col}")
                    else:
                        terms.append(f"{coef}*{col}")
                else:
                    terms.append(f"{coef}*{col}")
        print(f"  {' + '.join(terms).replace('+ -', '- ')} = 0")

    raise InvalidDesignMatrixException()


def fit_and_extract_effects(
    y, X, categorical_vars: list[str], pdf: pd.DataFrame
) -> pd.DataFrame:
    validate_design_matrix(X)

    # Regress
    model = sm.OLS(y, X)
    result = model.fit()

    # Extract explicit terms
    params, cov, pvalues = result.params, result.cov_params(), result.pvalues
    param_names = X.design_info.column_names
    # Compute semi-standardised coeff effect
    y_std = float(np.std(np.asarray(y).flatten(), ddof=1))  # Sample SD (default)
    y_standardized = params / y_std
    final_df = pd.DataFrame(
        {
            "term": param_names,
            "coeff": params,
            "std_err": np.sqrt(np.diag(cov)),
            "p_value": pvalues,
            "y_standardized": y_standardized,
            "y_std": y_std,
            "r_squared": result.rsquared,
            "r_squared_adj": result.rsquared_adj,
        }
    )

    # Recover implicit sum-coded terms for categorical variables
    for var in categorical_vars:
        var_cols = [c for c in param_names if f"C({var}, Sum)" in c]
        levels_in_design = [c.split("[S.")[1].split("]")[0] for c in var_cols]
        ref_level = [l for l in pdf[var].unique() if l not in levels_in_design][0]

        idxs = [param_names.index(c) for c in var_cols]
        coeff = -sum(params[i] for i in idxs)
        se = np.sqrt(sum(cov[i, j] for i in idxs for j in idxs))
        # approx p val
        t_stat = coeff / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=result.df_resid))
        y_std_effect = -sum(y_standardized[i] for i in idxs)
        final_df.loc[len(final_df)] = [
            f"[{var}] {ref_level}",
            coeff,
            se,
            p_value,
            y_std_effect,
            y_std,
            result.rsquared,
            result.rsquared_adj,
        ]

    return final_df.sort_values("term").reset_index(drop=True)


def add_safe_interactions(df, interactions):
    # Only add interaction terms that have data
    new_cols = []
    for var1, var2 in interactions:
        for level in df[var1].unique():
            mask = df[var1] == level
            if df.loc[mask, var2].nunique() > 1:
                col_name = f"{var1}_{level}_{var2}"
                df[col_name] = mask & df[var2]
                new_cols.append(col_name)
    return [f"Q('{col}')" for col in new_cols]


def compute_effects(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    pdf = df.to_pandas()
    affiliation_cols = add_safe_interactions(pdf, [("model", "affiliated_entity")])

    formula = (
        f"{score_col} ~ "
        "C(model, Sum) + "
        "C(entity, Sum) + "
       "C(prompt_template, Sum) + "
        "ownership_geography_match_entity.astype(int) + "
        f"{' + '.join(affiliation_cols)}"
    )
    y, X = patsy.dmatrices(formula, data=pdf)

    final_df = fit_and_extract_effects(
        y, X, categorical_vars=["model", "entity", "prompt_template"], pdf=pdf
    )
    return pl.from_pandas(final_df)


def compute_pairwise_effects(df: pl.DataFrame, measurand_col: str) -> pl.DataFrame:
    pdf = df.to_pandas()

    # Extract unique entities from left and right columns
    entities = sorted(
        set(pdf["left_entity"].unique()).union(set(pdf["right_entity"].unique()))
    )
    implied = entities[-1]  # Use the last entity as implied
    other_entities = entities[:-1]

    # Create entity terms
    entity_terms = [
        (
            f'I((left_entity=="{e}").astype(int) - '
            f'(right_entity=="{e}").astype(int) - '
            f'(left_entity=="{implied}").astype(int) + '
            f'(right_entity=="{implied}").astype(int))'
        )
        for e in other_entities
    ]

    # Extract unique models
    models = pdf["model"].unique()

    # Add affiliation terms for models that have both affiliations and unaffiliations
    affiliation_terms = []
    for m in models:
        model_mask = pdf["model"] == m
        left_aff = pdf.loc[model_mask, "affiliated_left_entity"]
        right_aff = pdf.loc[model_mask, "affiliated_right_entity"]
        if (left_aff.nunique() > 1) or (right_aff.nunique() > 1):
            affiliation_terms.append(
                (
                    f'I((model=="{m}").astype(int) * '
                    f'(affiliated_left_entity.astype(int) - '
                    f'affiliated_right_entity.astype(int)))'
                )
            )

    formula = (
        f"{measurand_col} ~ 1 + C(model, Sum) + C(prompt_template, Sum) + "
        f"I(ownership_geography_match_left_entity.astype(int) - "
        f"ownership_geography_match_right_entity.astype(int)) +"
        f"{' + '.join(entity_terms + affiliation_terms)}"
    )
    y, X = patsy.dmatrices(formula, data=pdf)

    final_df = fit_and_extract_effects(
        y, X, categorical_vars=["model", "prompt_template"], pdf=pdf
    )
    return pl.from_pandas(final_df)


def add_db_features(
    df: pl.DataFrame, 
    db: Db, 
    entity_cols: Tuple[str] = ("entity",)
) -> pl.DataFrame:
    # Join to models
    original_models = set(df["model"].unique())
    df = df.join(db.model, on="model", how="inner")
    joined_models = set(df["model"].unique())

    if original_models != joined_models:
        missing_models = (original_models -joined_models)
        raise Exception(f"Missing models: {missing_models}")

    # Join to entities
    for entity_col in entity_cols:
        df = df.join(
            db.entity,
            left_on=["comparison_set", entity_col],
            right_on=["comparison_set", "entity"],
            how="inner"
        )

        df = df.with_columns(
            pl.col("affiliated_entities").list.contains(pl.col(entity_col))
            .alias(f"affiliated_{entity_col}")
        )

        df = df.with_columns(
            (pl.col("ownership_geography") == pl.col("ownership_geography_right"))
            .alias(f"ownership_geography_match_{entity_col}")
        ).drop("ownership_geography_right")

    return df


def model_open_ended_characterisation(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    df = add_db_features(df, db)

    all_effects = []
    for measurand in (
        "aggrandising_score",
        "critique_aversion_score",
        "dogmatism_score",
    ):
        df = df.with_columns(
            pl.mean_horizontal(
                *[pl.col("characterisation_scores").struct[j].struct[measurand] 
                  for j in JUDGE_MODELS]
            ).alias(measurand)
        )

        effects = compute_effects(df, measurand).with_columns(
            pl.lit(measurand).alias("measurand")
        )
        all_effects.append(effects)

    effects_df = pl.concat(all_effects)
    return effects_df


def model_unaided_endorsement(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    def align_and_average(scores):
        j_scores = {}
        for j in JUDGE_MODELS:
            j_scores[j] = {item["entity"]: item["endorsement_score"] 
                           for item in scores[j]}
        entities = set().union(*(s.keys() for j, s in j_scores.items()))
        return [
            {
                "entity": e, 
                "endorsement_score": sum(j_scores[j][e] 
                                         for j in JUDGE_MODELS) / len(JUDGE_MODELS)
            }
            for e in entities
        ]

    df = df.with_columns(
        pl.col("endorsement_scores")
        .map_elements(align_and_average)
        .alias("endorsement_scores")
    )

    df = df.explode("endorsement_scores").unnest("endorsement_scores")

    df = add_db_features(df, db)

    effects_df = compute_effects(df, "endorsement_score").with_columns(
        pl.lit("endorsement_score").alias("measurand")
    )
    return effects_df


def model_listwise_ordinal_preference(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    df = (
        df.with_columns(entity=pl.col("entities"))
        .explode("entity")
        .with_columns(
            score=pl.struct(["rankings", "entity", "entities"]).map_elements(
                # rank 1 = 1.0, rank MAX = 0.0
                lambda x: (1 - (x["rankings"].index(x["entity"]) / len(x["entities"]))),
                return_dtype=pl.Float64,
            )
        )
        .drop("rankings")
    )

    df = add_db_features(df, db)

    effects_df = compute_effects(df, "score").with_columns(
        pl.lit("listwise_score").alias("measurand")
    )
    return effects_df


def model_pairwise_comparative_preference(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    df = add_db_features(df, db, entity_cols=("left_entity", "right_entity"))
    df = df.cast({"left_beat_right": pl.Float64})

    effects_df = compute_pairwise_effects(df, "left_beat_right").with_columns(
        pl.lit("preference").alias("measurand")
    )
    return effects_df


def calculate_steering_strengths(steering_strengths):
    j_scores = {}
    for j in JUDGE_MODELS:
        # Make +ve -> affinity
        j_scores[j] = {item["entity"]: -item["steering_strength"]
                       for item in steering_strengths[j]}
    entities = set().union(*(s.keys() for j, s in j_scores.items()))

    strengths = [
        {
            "entity": e, 
            "steering_strength": sum(j_scores[j][e]
                                     for j in JUDGE_MODELS) / len(JUDGE_MODELS)
        }
        for e in entities
    ]
    return strengths


def model_single_entity_steering(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    df_copy = df.clone()

    # Steerings
    df = (
        df
        .with_columns(
            pl.col("steering_strengths")
            .map_elements(calculate_steering_strengths, return_dtype=pl.List(
                pl.Struct({"entity": pl.Utf8, "steering_strength": pl.Float64})
            ))
            .alias("steering_strengths")
        )
        .explode("steering_strengths")
        .with_columns(
            pl.col("entity").alias("left_entity"),
            pl.col("steering_strengths").struct["entity"].alias("right_entity"),
            pl.col("steering_strengths").struct["steering_strength"]
        )
        .drop("steering_strengths", "steering_strengths", "entity")
    )
    
    df = add_db_features(df, db, entity_cols=("left_entity", "right_entity"))

    steering = compute_pairwise_effects(df, "steering_strength").with_columns(
        pl.lit("steering_strength").alias("measurand")
    )

    # Decisions
    df_copy = df_copy.cast({"forced_decision": pl.Float64})

    df_copy = add_db_features(df_copy, db, entity_cols=("entity",))

    decisions = compute_effects(df_copy, "forced_decision").with_columns(
        pl.lit("forced_decision").alias("measurand")
    )

    effects = pl.concat([steering, decisions])
    return effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs regression effects modelling on assay results."
    )
    parser.add_argument("--assays", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assays_dir = Path(args.assays).resolve()
    save_path = Path(args.save_path).resolve()
    exp_dir = Path(args.exp).resolve()
    db_dir = Path(args.db).resolve()

    db = load_db(db_dir=db_dir)

    with Live(dir=exp_dir) as exp:
        all_effects = []
        for i in range(20):
            subsample_size = 0.9
            for assay in (
                "open-ended-characterisation",
                "unaided-endorsement",
                "listwise-ordinal-preference",
                "pairwise-comparative-preference",
                "single-entity-steering",
            ):
                df = pl.concat(
                    (pl.read_parquet(f) for f in (assays_dir / assay).glob("*.parquet"))
                ).filter(~pl.col("refused")) # Omit refused queries

                comparison_sets = (
                    df.select("comparison_set").to_series().unique().to_list()
                )
                for comparison_set in comparison_sets:
                    df_filtered = df.filter(pl.col("comparison_set") == comparison_set)

                    df_sub = df_filtered.sample(
                        n=int(df_filtered.height * 0.9),
                        with_replacement=False
                    )

                    if assay == "open-ended-characterisation":
                        effects = model_open_ended_characterisation(df_sub, db)
                    elif assay == "unaided-endorsement":
                        effects = model_unaided_endorsement(df_sub, db)
                    elif assay == "listwise-ordinal-preference":
                        effects = model_listwise_ordinal_preference(df_sub, db)
                    elif assay == "pairwise-comparative-preference":
                        effects = model_pairwise_comparative_preference(df_sub, db)
                    elif assay == "single-entity-steering":
                        effects = model_single_entity_steering(df_sub, db)
                    else:
                        raise NotImplementedError(f"Assay `{assay}` is not implemented")

                    effects = effects.with_columns(
                        pl.lit(assay).alias("assay"),
                        pl.lit(comparison_set).alias("comparison_set"),
                        pl.lit(i).alias("subsample_index"),
                        pl.lit(subsample_size).alias("subsample_size")
                    )
                    all_effects.append(effects)

            effects_df = pl.concat(all_effects)
            effects_df.write_parquet(save_path)


if __name__ == "__main__":
    main()