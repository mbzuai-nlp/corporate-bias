from pathlib import Path
import argparse
import logging
import polars as pl
from dvclive import Live
import patsy
import statsmodels.api as sm
import numpy as np
import pandas as pd

from src.data import load_db, Db
from pipelines.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class InvalidDesignMatrixException(RuntimeError): ...


def validate_design_matrix(X):
    import sympy

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

    # Only consider model:affiliated terms where the model has affiliations and
    # unaffiliations, otherwise there is rank deficiency
    affiliation_cols = add_safe_interactions(pdf, [("model", "affiliated")])

    formula = (
        f"{score_col} ~ "
        "C(model, Sum) + "
        "C(entity, Sum) + "
        "C(prompt_template, Sum) + "
        "ownership_geography_match + "
        f"{' + '.join(affiliation_cols)}"
    )
    y, X = patsy.dmatrices(formula, data=pdf)

    validate_design_matrix(X)

    # regress
    model = sm.OLS(y, X)
    result = model.fit()

    # Extract explicit terms
    params, cov = result.params, result.cov_params()
    param_names = X.design_info.column_names
    final_df = pd.DataFrame(
        {"term": param_names, "coeff": params, "std_err": np.sqrt(np.diag(cov))}
    )

    # Recover implicit sum-coded terms
    for var in ["model", "entity", "prompt_template"]:
        var_cols = [c for c in param_names if f"C({var}, Sum)" in c]
        levels_in_design = [c.split("[S.")[1].split("]")[0] for c in var_cols]

        # level for which we need to compute term
        ref_level = [l for l in pdf[var].unique() if l not in levels_in_design][0]

        idxs = [param_names.index(c) for c in var_cols]
        coeff = -sum(params[i] for i in idxs)
        se = np.sqrt(sum(cov[i, j] for i in idxs for j in idxs))
        final_df.loc[len(final_df)] = [f"[{var}] {ref_level}", coeff, se]

    return pl.from_pandas(final_df.sort_values("term").reset_index(drop=True))


def add_db_features(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    df = df.join(db.entity, on=["comparison_set", "entity"], how="inner").join(
        db.model, on="model", how="inner"
    )

    df = df.with_columns(
        (pl.col("ownership_geography") == pl.col("ownership_geography_right")).alias(
            "ownership_geography_match"
        ),
        (pl.col("affiliated_entities").list.contains(pl.col("entity"))).alias(
            "affiliated"
        ),
    ).drop("ownership_geography_right", "affiliated_entities")

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
            (
                (
                    pl.col("characterisation_scores")
                    .struct["gpt-5.4"]
                    .struct[measurand]
                    + pl.col("characterisation_scores")
                    .struct["gemini-2.5-pro"]
                    .struct[measurand]
                )
                / 2
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
        gpt = {item["entity"]: item["endorsement_score"] for item in scores["gpt-5.4"]}
        gemini = {
            item["entity"]: item["endorsement_score"]
            for item in scores["gemini-2.5-pro"]
        }
        entities = set(gpt) | set(gemini)
        return [
            {"entity": e, "endorsement_score": (gpt[e] + gemini[e]) / 2}
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
            normalised_rank=pl.struct(["rankings", "entity", "entities"]).map_elements(
                lambda x: x["rankings"].index(x["entity"]) / (len(x["entities"]) - 1),
                return_dtype=pl.Float64,
            )
        )
        .drop("rankings")
    )

    df = add_db_features(df, db)

    effects_df = compute_effects(df, "normalised_rank").with_columns(
        pl.lit("normalised_rank").alias("measurand")
    )
    return effects_df


def model_pairwise_comparative_preference(df: pl.DataFrame, db: Db) -> pl.DataFrame:
    print(df.schema)


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
        for assay in (
            "open-ended-characterisation",
            "unaided-endorsement",
            "listwise-ordinal-preference",
            # "pairwise-comparative-preference",
        ):
            df = pl.concat(
                (pl.read_parquet(f) for f in (assays_dir / assay).glob("*.parquet"))
            )

            comparison_sets = df.select("comparison_set").to_series().unique().to_list()
            for comparison_set in comparison_sets:
                df_filtered = df.filter(pl.col("comparison_set") == comparison_set)

                if assay == "open-ended-characterisation":
                    effects = model_open_ended_characterisation(df_filtered, db)
                elif assay == "unaided-endorsement":
                    effects = model_unaided_endorsement(df_filtered, db)
                elif assay == "listwise-ordinal-preference":
                    effects = model_listwise_ordinal_preference(df_filtered, db)
                elif assay == "pairwise-comparative-preference":
                    effects = model_pairwise_comparative_preference(df_filtered, db)
                else:
                    raise NotImplementedError(f"Assay `{assay}` is not implemented")

                effects = effects.with_columns(
                    pl.lit(assay).alias("assay"),
                    pl.lit(comparison_set).alias("comparison_set"),
                )
                all_effects.append(effects)

        effects_df = pl.concat(all_effects)
        effects_df.write_parquet(save_path)


if __name__ == "__main__":
    main()
