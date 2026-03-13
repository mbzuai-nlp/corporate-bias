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

COMPARISON_SET_LINK_SCHEMA = {
    "comparison_set_id": pl.String,  # key
    "comparison_set_name": pl.String,
    "entity_id": pl.String,  # key
    "entity_name": pl.String,
}

COMPARISON_SET_ASSAY_ARGS_SCHEMA = {
    "comparison_set_id": pl.String, # key
    "comparison_set_name": pl.String,
    "assay": pl.String, # key
    "serialised_args": pl.String
}

ASSAY_SCHEMA = {
    "assay": pl.String,  # key
    "model": pl.String,  # key
    "comparison_set_id": pl.String,  # key
    "serialised_result": pl.String,
}

