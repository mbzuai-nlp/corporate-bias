import polars as pl

ENTITY_SCHEMA = {
    "id": pl.String,  # key
    "name": pl.String,
    "aliases": pl.List(pl.String),
    "source": pl.String,
    "facets": pl.List(pl.Struct({"dimension": pl.Utf8, "value": pl.Utf8})),
}

CLAIM_SCHEMA = {
    "predicate": pl.String,  # key
    "subject_id": pl.String,  # key
    "subject_name": pl.String,
    "object_id": pl.String,  # key
    "object_name": pl.String,
    "source": pl.String,
}

COMPARISON_SET_SCHEMA = {
    "id": pl.String,  # key
    "name": pl.String,
    "entity_ids": pl.List(pl.String),
}

COMPARISON_SET_ASSAY_INSTANCE_SCHEMA = {
    "comparison_set_id": pl.String,  # key
    "comparison_set_name": pl.String,
    "assay": pl.String,  # key
    "instance_hash": pl.UInt64,  # key
    "instance": pl.Object,
}

ASSAY_SCHEMA = {
    "assay": pl.String, # key
    "assay_instance_hash": pl.UInt64, # key
    "sample_number": pl.UInt64, # key
    "model": pl.String,  # key
    "comparison_set_id": pl.String,  # key
    "comparison_set_name": pl.String,
    "entity_id": pl.String,  # key,
    "entity_name": pl.String,
    "debug_json": pl.String,
    "measurements": pl.List(pl.Struct({
        "measurand": pl.Utf8, 
        "value": pl.Float64
    }))
}

REGRESSION_EFFECT_SCHEMA = {
    "assay": pl.String, # key
    "measurand": pl.String, # key
    "term": pl.String, # key
    "estimate": pl.Float64,
    "std_error": pl.Float64,
    "statistic": pl.Float64,
    "statistic_type": pl.String,
    "p_value": pl.Float64,
    "aliased": pl.Boolean,
    "regression_statistics": pl.Struct(
        {
            "nobs": pl.UInt64,
            "rank": pl.UInt64,
            "df_residual": pl.UInt64,
            "r_squared": pl.Float64,
            "adj_r_squared": pl.Float64,
            "sigma": pl.Float64,
            "f_statistic": pl.Float64,
            "f_numdf": pl.Float64,
            "f_dendf": pl.Float64,
            "f_p_value": pl.Float64,
            "aic": pl.Float64,
            "bic": pl.Float64,
        }
    ),
}