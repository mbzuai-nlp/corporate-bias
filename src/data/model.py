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
    "id": pl.String,
    "name": pl.String,
    "entities": pl.List(
        pl.Struct(
            {
                "entity_id": pl.String,
                "entity_name": pl.String,
            }
        )
    ),
}

COMPARISON_SET_ASSAY_INSTANCE_SCHEMA = {
    "comparison_set_id": pl.String, # key
    "comparison_set_name": pl.String,
    "assay": pl.String, # key
    "instance_hash": pl.UInt64, # key
    "instance": pl.Object
}

ASSAY_SCHEMA = {
    "assay": pl.String,  # key
    "assay_instance_hash": pl.UInt64, # key
    "model": pl.String,  # key
    "comparison_set_id": pl.String,  # key
    "comparison_set_name": pl.String,
    "entity_id": pl.String, # key,
    "entity_name": pl.String,
    "result": pl.List(pl.Struct({"estimand": pl.Utf8, "value": pl.Utf8})),
}
