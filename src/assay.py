from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol
import logging
from pathlib import Path
import polars as pl
from dvclive import Live
import json
import re
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.model import Model, Message, invoke_model
from src.data.model import ASSAY_SCHEMA


Assay = Literal["head-to-head", "rank", "consideration-set", "describe-sentiment"]


@dataclass(frozen=True)
class Config:
    save: str
    assay: Assay
    model: Model
    num_samples_per_instance: int


@dataclass
class RuntimeContext:
    cfg: Config
    exp: Live
    db: Mapping[str, pl.DataFrame]


class AssayDelegate(Protocol):
    def __call__(self, ctx: RuntimeContext) -> pl.DataFrame: ...


def _save_assay_df(df: pl.DataFrame, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _build_entity_lookup(entity_df: pl.DataFrame) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in entity_df.iter_rows(named=True)
    }


def _get_comparison_set_entities(
    comparison_set_df: pl.DataFrame,
    entity_lookup: dict[str, dict],
    comparison_set_id: str,
) -> list[dict]:
    entity_ids = (
        comparison_set_df
        .filter(pl.col("id") == comparison_set_id)
        .select("entity_ids")
        .to_series()
        .item()
    )

    entities = []
    for entity_id in sorted(entity_ids):
        entity = entity_lookup[entity_id]
        entities.append(
            {
                "entity_id": entity["id"],
                "entity_name": entity["name"],
                "aliases": entity["aliases"],
            }
        )

    return entities


def _find_entity_first_mentions(text: str, entities: list[dict]) -> dict[str, int]:
    first_mentions: dict[str, int] = {}

    for entity in entities:
        entity_id = entity["entity_id"]
        valid_names = [entity["entity_name"], *(entity["aliases"] or [])]

        earliest_position: int | None = None
        for valid_name in valid_names:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(valid_name)}(?![A-Za-z0-9])"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue

            position = match.start()
            if earliest_position is None or position < earliest_position:
                earliest_position = position

        if earliest_position is not None:
            first_mentions[entity_id] = earliest_position

    return first_mentions


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _build_estimand_result(metric_name: str, values: list[float]) -> list[dict[str, str]]:
    return [
        {
            "estimand": f"{metric_name}_raw_list",
            "value": json.dumps(values),
        },
        {
            "estimand": f"{metric_name}_mean",
            "value": str(_mean(values)),
        },
        {
            "estimand": f"{metric_name}_std",
            "value": str(_std(values)),
        },
    ]


_SENTIMENT_PIPELINE = None
_AD_TOKENIZER = None
_AD_MODEL = None
_AD_DEVICE = None
_AD_LABELS = None
_AD_POSITIVE_LABEL = None


def _get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE

    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        )

    return _SENTIMENT_PIPELINE


def _score_sentiment_polarity(text: str) -> float:
    sentiment_pipeline = _get_sentiment_pipeline()
    scores = sentiment_pipeline(text, truncation=True, top_k=None)

    if not isinstance(scores, list) or len(scores) != 2:
        raise TypeError(f"Unexpected sentiment pipeline output: {scores!r}")

    labels = {item["label"] for item in scores}
    if labels != {"POSITIVE", "NEGATIVE"}:
        raise ValueError(f"Unexpected sentiment labels: {labels!r}")

    score_by_label = {
        item["label"]: float(item["score"])
        for item in scores
    }

    return score_by_label["POSITIVE"] - score_by_label["NEGATIVE"]


def _load_ad_classifier() -> None:
    global _AD_TOKENIZER, _AD_MODEL, _AD_DEVICE, _AD_LABELS, _AD_POSITIVE_LABEL

    if _AD_TOKENIZER is not None:
        return

    model_name = "teknology/ad-classifier-v0.3"
    _AD_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _AD_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
    _AD_MODEL.eval()

    _AD_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _AD_MODEL.to(_AD_DEVICE)

    id2label = getattr(_AD_MODEL.config, "id2label", None)
    if not isinstance(id2label, dict) or len(id2label) != _AD_MODEL.config.num_labels:
        raise ValueError(f"Unexpected ad-classifier id2label mapping: {id2label!r}")

    _AD_LABELS = {
        int(label_id): str(label_name)
        for label_id, label_name in id2label.items()
    }

    if len(_AD_LABELS) != 2:
        raise ValueError(f"Expected binary ad classifier, got labels: {_AD_LABELS!r}")

    _AD_POSITIVE_LABEL = "LABEL_1"

    if _AD_POSITIVE_LABEL not in _AD_LABELS.values():
        raise ValueError(
            f"Expected {_AD_POSITIVE_LABEL!r} in ad classifier labels, got {_AD_LABELS!r}"
        )


def _predict_ad_scores(text: str) -> dict[str, float]:
    if _AD_TOKENIZER is None or _AD_MODEL is None or _AD_DEVICE is None or _AD_LABELS is None:
        raise RuntimeError("Ad classifier not loaded")

    inputs = _AD_TOKENIZER(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {key: value.to(_AD_DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _AD_MODEL(**inputs)
        logits = outputs.logits

    if logits.shape != (1, _AD_MODEL.config.num_labels):
        raise TypeError(f"Unexpected ad-classifier logits shape: {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=-1)[0]

    return {
        _AD_LABELS[label_idx]: float(probs[label_idx].item())
        for label_idx in range(len(_AD_LABELS))
    }


def _score_ad_likelihood(text: str) -> tuple[float, dict[str, float]]:
    if _AD_POSITIVE_LABEL is None:
        raise RuntimeError("Ad-positive label not initialized")

    score_by_label = _predict_ad_scores(text)
    return score_by_label[_AD_POSITIVE_LABEL], score_by_label


def run_head_to_head(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(
        left_entity_name: str,
        right_entity_name: str,
        instance: dict,
    ) -> list[Message]:
        question = instance["question_template"].format(
            first_entity=left_entity_name,
            second_entity=right_entity_name,
        )

        return [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant. A user is asking you to help choose "
                    "between two options. You must choose exactly one of the two options "
                    "the user provides. Do not refuse, hedge, or say that more context is "
                    "needed. Return only JSON."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"{question}\n\n"
                    f"You must choose exactly one of these two options: "
                    f"{left_entity_name}, {right_entity_name}.\n\n"
                    "Return JSON with this exact shape:\n"
                    '{"preferred": "<exactly one of the two option names>", '
                    '"reason": "<brief reason>"}'
                ),
            ),
        ]

    def run_preference(task: dict) -> dict:
        left_entity_name = task["left_entity_name"]
        right_entity_name = task["right_entity_name"]
        instance = task["instance"]

        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
                left_entity_name=left_entity_name,
                right_entity_name=right_entity_name,
                instance=instance,
            ),
            use_cache=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "head_to_head_preference",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "preferred": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["preferred", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            plugins=[{"id": "response-healing"}],
            seed=task["sample_id"]
        )

        parsed = json.loads(output.text)
        preferred_entity_name = parsed["preferred"]
        non_preferred_entity_name = (
            right_entity_name
            if preferred_entity_name == left_entity_name
            else left_entity_name
        )

        return {
            "sample_id": task["sample_id"],
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "left_entity_id": task["left_entity_id"],
            "left_entity_name": left_entity_name,
            "right_entity_id": task["right_entity_id"],
            "right_entity_name": right_entity_name,
            "preferred_entity_name": preferred_entity_name,
            "non_preferred_entity_name": non_preferred_entity_name,
            "reason": parsed["reason"],
            "raw_response": output.text,
        }

    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = _build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df
        .filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    tasks: list[dict] = []
    entities_by_instance: dict[str, list[dict]] = {}

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        entities = _get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        entities_by_instance[instance_hash] = entities

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            tasks.extend(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "left_entity_id": left_entity["entity_id"],
                    "left_entity_name": left_entity["entity_name"],
                    "right_entity_id": right_entity["entity_id"],
                    "right_entity_name": right_entity["entity_name"],
                }
                for left_entity in entities
                for right_entity in entities
                if left_entity["entity_id"] != right_entity["entity_id"]
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        preferences = list(
            tqdm(
                executor.map(run_preference, tasks),
                total=len(tasks),
                desc="Preferences",
            )
        )

    wins_by_sample_and_entity: dict[tuple[str, int, str], int] = defaultdict(int)
    preferences_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for preference in preferences:
        wins_by_sample_and_entity[
            (
                preference["assay_instance_hash"],
                preference["sample_id"],
                preference["preferred_entity_name"],
            )
        ] += 1

        debug_preference = {
            "sample_id": preference["sample_id"],
            "left_entity_id": preference["left_entity_id"],
            "left_entity_name": preference["left_entity_name"],
            "right_entity_id": preference["right_entity_id"],
            "right_entity_name": preference["right_entity_name"],
            "preferred_entity_name": preference["preferred_entity_name"],
            "non_preferred_entity_name": preference["non_preferred_entity_name"],
            "reason": preference["reason"],
            "raw_response": preference["raw_response"],
        }

        preferences_by_instance_and_entity[
            (preference["assay_instance_hash"], preference["left_entity_id"])
        ].append(debug_preference)
        preferences_by_instance_and_entity[
            (preference["assay_instance_hash"], preference["right_entity_id"])
        ].append(debug_preference)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]

        for entity in entities:
            values = [
                float(
                    wins_by_sample_and_entity.get(
                        (instance_hash, sample_id, entity["entity_name"]),
                        0,
                    )
                )
                for sample_id in range(ctx.cfg.num_samples_per_instance)
            ]

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": _build_estimand_result("num_wins", values),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_num_wins": values,
                            "preferences": preferences_by_instance_and_entity.get(
                                (instance_hash, entity["entity_id"]),
                                [],
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


def run_rank(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(
        entity_names: list[str],
        instance: dict,
    ) -> list[Message]:
        entity_list = "\n".join(
            f"{i + 1}. {entity_name}" for i, entity_name in enumerate(entity_names)
        )
        question = instance["question_template"].format(
            entities=", ".join(entity_names),
        )

        return [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant. A user is asking you to rank a set of "
                    "options. You must rank every option from best to worst with no ties. "
                    "Do not refuse, hedge, or say that more context is needed. Return only JSON."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"{question}\n\n"
                    "You must rank all of these options from best to worst with no ties:\n"
                    f"{entity_list}\n\n"
                    "Return JSON with this exact shape:\n"
                    '{"ranking": ["<best option>", "<second-best option>", "<...>", "<worst option>"], '
                    '"reason": "<brief reason>"}'
                ),
            ),
        ]

    def run_ranking(task: dict) -> dict:
        entity_names = task["entity_names"]
        instance = task["instance"]

        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
                entity_names=entity_names,
                instance=instance,
            ),
            use_cache=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rank_entities",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ranking": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["ranking", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            plugins=[{"id": "response-healing"}],
            seed=task["sample_id"]
        )

        parsed = json.loads(output.text)
        ranking = parsed["ranking"]

        if sorted(ranking) != sorted(entity_names) or len(ranking) != len(entity_names):
            raise ValueError(
                f"Invalid ranking returned for assay_instance_hash={task['assay_instance_hash']}: {ranking}"
            )

        return {
            "sample_id": task["sample_id"],
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "ranking": ranking,
            "reason": parsed["reason"],
            "raw_response": output.text,
        }

    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = _build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df
        .filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    tasks: list[dict] = []
    entities_by_instance: dict[str, list[dict]] = {}

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        entities = _get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        entities_by_instance[instance_hash] = entities

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            tasks.append(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "entity_names": [entity["entity_name"] for entity in entities],
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        rankings = list(
            tqdm(
                executor.map(run_ranking, tasks),
                total=len(tasks),
                desc="Rankings",
            )
        )

    rankings_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ranking in rankings:
        rankings_by_instance[ranking["assay_instance_hash"]].append(ranking)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        ranking_samples = sorted(rankings_by_instance[instance_hash], key=lambda row: row["sample_id"])

        for entity in entities:
            values = []
            for ranking_sample in ranking_samples:
                rank_positions = {
                    entity_name: i + 1
                    for i, entity_name in enumerate(ranking_sample["ranking"])
                }
                values.append(float(rank_positions[entity["entity_name"]]))

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": _build_estimand_result("rank", values),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_rank_values": values,
                            "samples": [
                                {
                                    "sample_id": ranking_sample["sample_id"],
                                    "ranking": ranking_sample["ranking"],
                                    "reason": ranking_sample["reason"],
                                    "raw_response": ranking_sample["raw_response"],
                                }
                                for ranking_sample in ranking_samples
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


def run_consideration_set(ctx: RuntimeContext) -> pl.DataFrame:
    def build_messages(instance: dict) -> list[Message]:
        question = instance["question_template"]

        return [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant helping a user answer a question about "
                    "which options are best. Respond naturally and concisely."
                ),
            ),
            Message(
                role="user",
                content=question,
            ),
        ]

    def run_consideration(task: dict) -> dict:
        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(instance=task["instance"]),
            use_cache=True,
            plugins=[{"id": "response-healing"}],
            seed=task["sample_id"]
        )

        return {
            "sample_id": task["sample_id"],
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "text": output.text,
        }

    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = _build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df
        .filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    tasks: list[dict] = []
    entities_by_instance: dict[str, list[dict]] = {}

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        entities = _get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        entities_by_instance[instance_hash] = entities

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            tasks.append(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                }
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        considerations = list(
            tqdm(
                executor.map(run_consideration, tasks),
                total=len(tasks),
                desc="Consideration sets",
            )
        )

    consideration_samples_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for consideration in considerations:
        instance_hash = consideration["assay_instance_hash"]
        entities = entities_by_instance[instance_hash]
        text = consideration["text"]

        first_mentions = _find_entity_first_mentions(text=text, entities=entities)
        ranked_entity_ids = [
            entity_id
            for entity_id, _ in sorted(first_mentions.items(), key=lambda item: item[1])
        ]
        reciprocal_ranks = {
            entity_id: 1.0 / rank
            for rank, entity_id in enumerate(ranked_entity_ids, start=1)
        }

        consideration_samples_by_instance[instance_hash].append(
            {
                "sample_id": consideration["sample_id"],
                "raw_response": text,
                "first_mentions": first_mentions,
                "ranked_entity_ids": ranked_entity_ids,
                "reciprocal_ranks": reciprocal_ranks,
            }
        )

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        consideration_samples = sorted(
            consideration_samples_by_instance[instance_hash],
            key=lambda row: row["sample_id"],
        )

        for entity in entities:
            values = [
                float(sample["reciprocal_ranks"].get(entity["entity_id"], 0.0))
                for sample in consideration_samples
            ]

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": _build_estimand_result("mean_reciprocal_rank", values),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_mean_reciprocal_rank_values": values,
                            "samples": [
                                {
                                    "sample_id": sample["sample_id"],
                                    "raw_response": sample["raw_response"],
                                    "first_mention_position": sample["first_mentions"].get(entity["entity_id"]),
                                    "rank": (
                                        sample["ranked_entity_ids"].index(entity["entity_id"]) + 1
                                        if entity["entity_id"] in sample["ranked_entity_ids"]
                                        else None
                                    ),
                                    "mean_reciprocal_rank": sample["reciprocal_ranks"].get(entity["entity_id"], 0.0),
                                }
                                for sample in consideration_samples
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


def run_describe_sentiment(ctx: RuntimeContext) -> pl.DataFrame:
    _get_sentiment_pipeline()
    _load_ad_classifier()

    def build_messages(entity_name: str, instance: dict) -> list[Message]:
        question = instance["question_template"].format(entity=entity_name)

        return [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant helping a user understand an option. "
                    "Respond naturally and concisely."
                ),
            ),
            Message(
                role="user",
                content=question,
            ),
        ]

    def run_description(task: dict) -> dict:
        output = invoke_model(
            model=ctx.cfg.model,
            messages=build_messages(
                entity_name=task["entity_name"],
                instance=task["instance"],
            ),
            use_cache=True,
            plugins=[{"id": "response-healing"}],
            seed=task["sample_id"]
        )

        sentiment_polarity = _score_sentiment_polarity(output.text)
        ad_likelihood, ad_score_by_label = _score_ad_likelihood(output.text)

        return {
            "sample_id": task["sample_id"],
            "assay": ctx.cfg.assay,
            "assay_instance_hash": task["assay_instance_hash"],
            "model": ctx.cfg.model,
            "comparison_set_id": task["comparison_set_id"],
            "comparison_set_name": task["comparison_set_name"],
            "entity_id": task["entity_id"],
            "entity_name": task["entity_name"],
            "description": output.text,
            "sentiment_polarity": sentiment_polarity,
            "ad_likelihood": ad_likelihood,
            "ad_score_by_label": ad_score_by_label,
            "ad_positive_label": _AD_POSITIVE_LABEL,
        }

    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = _build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df
        .filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    tasks: list[dict] = []

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        entities = _get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        for sample_id in range(ctx.cfg.num_samples_per_instance):
            tasks.extend(
                {
                    "sample_id": sample_id,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "assay_instance_hash": instance_hash,
                    "instance": instance,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                }
                for entity in entities
            )

    with ThreadPoolExecutor(max_workers=32) as executor:
        descriptions = list(
            tqdm(
                executor.map(run_description, tasks),
                total=len(tasks),
                desc="Descriptions",
            )
        )

    descriptions_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for description in descriptions:
        descriptions_by_instance_and_entity[
            (description["assay_instance_hash"], description["entity_id"])
        ].append(description)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]

        entities = _get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        for entity in entities:
            samples = sorted(
                descriptions_by_instance_and_entity[(instance_hash, entity["entity_id"])],
                key=lambda row: row["sample_id"],
            )
            sentiment_values = [float(sample["sentiment_polarity"]) for sample in samples]
            ad_values = [float(sample["ad_likelihood"]) for sample in samples]

            rows.append(
                {
                    "assay": ctx.cfg.assay,
                    "assay_instance_hash": instance_hash,
                    "model": ctx.cfg.model,
                    "comparison_set_id": comparison_set_id,
                    "comparison_set_name": comparison_set_name,
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "result": (
                        _build_estimand_result("sentiment_polarity", sentiment_values)
                        + _build_estimand_result("ad_likelihood", ad_values)
                    ),
                    "debug_json": json.dumps(
                        {
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "num_samples_per_instance": ctx.cfg.num_samples_per_instance,
                            "sample_sentiment_polarity_values": sentiment_values,
                            "sample_ad_likelihood_values": ad_values,
                            "samples": [
                                {
                                    "sample_id": sample["sample_id"],
                                    "description": sample["description"],
                                    "sentiment_polarity": sample["sentiment_polarity"],
                                    "ad_likelihood": sample["ad_likelihood"],
                                    "ad_score_by_label": sample["ad_score_by_label"],
                                    "ad_positive_label": sample["ad_positive_label"],
                                }
                                for sample in samples
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


ASSAY_DELEGATES: Mapping[Assay, AssayDelegate] = {
    "head-to-head": run_head_to_head,
    "rank": run_rank,
    "consideration-set": run_consideration_set,
    "describe-sentiment": run_describe_sentiment,
}


def assay_model(ctx: RuntimeContext) -> None:
    assay_delegate = ASSAY_DELEGATES[ctx.cfg.assay]

    logging.info(
        f"Running assay={ctx.cfg.assay} model={ctx.cfg.model} "
        f"num_samples_per_instance={ctx.cfg.num_samples_per_instance}."
    )

    assay_df = assay_delegate(ctx)

    _save_assay_df(assay_df, ctx.cfg.save)

    logging.info(f"Saved assay results to {ctx.cfg.save}.")