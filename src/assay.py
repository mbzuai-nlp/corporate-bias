from dataclasses import dataclass
from typing import Literal, Mapping, Protocol
import logging
from pathlib import Path
import polars as pl
from dvclive import Live
import json
import re
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

    # Assumption: model preserves training-dataset label semantics:
    # label 1 = response contains advertisement
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
        )

        parsed = json.loads(output.text)
        preferred_entity_name = parsed["preferred"]
        non_preferred_entity_name = (
            right_entity_name
            if preferred_entity_name == left_entity_name
            else left_entity_name
        )

        return {
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

        tasks.extend(
            {
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

    wins: dict[tuple[str, str], int] = {}
    preferences_by_instance_and_entity: dict[tuple[str, str], list[dict]] = {}

    for preference in preferences:
        key = (
            preference["assay_instance_hash"],
            preference["preferred_entity_name"],
        )
        wins[key] = wins.get(key, 0) + 1

        left_key = (
            preference["assay_instance_hash"],
            preference["left_entity_id"],
        )
        right_key = (
            preference["assay_instance_hash"],
            preference["right_entity_id"],
        )

        debug_preference = {
            "left_entity_id": preference["left_entity_id"],
            "left_entity_name": preference["left_entity_name"],
            "right_entity_id": preference["right_entity_id"],
            "right_entity_name": preference["right_entity_name"],
            "preferred_entity_name": preference["preferred_entity_name"],
            "non_preferred_entity_name": preference["non_preferred_entity_name"],
            "reason": preference["reason"],
            "raw_response": preference["raw_response"],
        }

        preferences_by_instance_and_entity.setdefault(left_key, []).append(debug_preference)
        preferences_by_instance_and_entity.setdefault(right_key, []).append(debug_preference)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]

        rows.extend(
            {
                "assay": ctx.cfg.assay,
                "assay_instance_hash": instance_hash,
                "model": ctx.cfg.model,
                "comparison_set_id": comparison_set_id,
                "comparison_set_name": comparison_set_name,
                "entity_id": entity["entity_id"],
                "entity_name": entity["entity_name"],
                "result": [
                    {
                        "estimand": "num_wins",
                        "value": str(wins.get((instance_hash, entity["entity_name"]), 0)),
                    }
                ],
                "debug_json": json.dumps(
                    {
                        "entity_id": entity["entity_id"],
                        "entity_name": entity["entity_name"],
                        "preferences": preferences_by_instance_and_entity.get(
                            (instance_hash, entity["entity_id"]),
                            [],
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for entity in entities
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))

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
        )

        parsed = json.loads(output.text)
        ranking = parsed["ranking"]

        if sorted(ranking) != sorted(entity_names) or len(ranking) != len(entity_names):
            raise ValueError(
                f"Invalid ranking returned for assay_instance_hash={task['assay_instance_hash']}: {ranking}"
            )

        return {
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

        tasks.append(
            {
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

    ranking_by_instance = {
        ranking["assay_instance_hash"]: ranking["ranking"]
        for ranking in rankings
    }
    ranking_debug_by_instance = {
        ranking["assay_instance_hash"]: {
            "ranking": ranking["ranking"],
            "reason": ranking["reason"],
            "raw_response": ranking["raw_response"],
        }
        for ranking in rankings
    }

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        rank_positions = {
            entity_name: i + 1
            for i, entity_name in enumerate(ranking_by_instance[instance_hash])
        }
        ranking_debug = ranking_debug_by_instance[instance_hash]

        rows.extend(
            {
                "assay": ctx.cfg.assay,
                "assay_instance_hash": instance_hash,
                "model": ctx.cfg.model,
                "comparison_set_id": comparison_set_id,
                "comparison_set_name": comparison_set_name,
                "entity_id": entity["entity_id"],
                "entity_name": entity["entity_name"],
                "result": [
                    {
                        "estimand": "rank",
                        "value": str(rank_positions[entity["entity_name"]]),
                    }
                ],
                "debug_json": json.dumps(
                    {
                        "entity_id": entity["entity_id"],
                        "entity_name": entity["entity_name"],
                        "rank": rank_positions[entity["entity_name"]],
                        "ranking": ranking_debug["ranking"],
                        "reason": ranking_debug["reason"],
                        "raw_response": ranking_debug["raw_response"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for entity in entities
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))

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
        )

        return {
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

        tasks.append(
            {
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

    reciprocal_ranks_by_instance: dict[str, dict[str, float]] = {}
    consideration_debug_by_instance: dict[str, dict] = {}

    for consideration in considerations:
        instance_hash = consideration["assay_instance_hash"]
        entities = entities_by_instance[instance_hash]
        text = consideration["text"]

        first_mentions = _find_entity_first_mentions(text=text, entities=entities)
        ranked_entity_ids = [
            entity_id
            for entity_id, _ in sorted(first_mentions.items(), key=lambda item: item[1])
        ]

        reciprocal_ranks: dict[str, float] = {
            entity_id: 1.0 / rank
            for rank, entity_id in enumerate(ranked_entity_ids, start=1)
        }
        reciprocal_ranks_by_instance[instance_hash] = reciprocal_ranks
        consideration_debug_by_instance[instance_hash] = {
            "raw_response": text,
            "first_mentions": first_mentions,
            "ranked_entity_ids": ranked_entity_ids,
        }

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        entities = entities_by_instance[instance_hash]
        reciprocal_ranks = reciprocal_ranks_by_instance.get(instance_hash, {})
        debug = consideration_debug_by_instance.get(
            instance_hash,
            {"raw_response": "", "first_mentions": {}, "ranked_entity_ids": []},
        )

        rank_by_entity_id = {
            entity_id: rank
            for rank, entity_id in enumerate(debug["ranked_entity_ids"], start=1)
        }

        rows.extend(
            {
                "assay": ctx.cfg.assay,
                "assay_instance_hash": instance_hash,
                "model": ctx.cfg.model,
                "comparison_set_id": comparison_set_id,
                "comparison_set_name": comparison_set_name,
                "entity_id": entity["entity_id"],
                "entity_name": entity["entity_name"],
                "result": [
                    {
                        "estimand": "mean_reciprocal_rank",
                        "value": str(reciprocal_ranks.get(entity["entity_id"], 0.0)),
                    }
                ],
                "debug_json": json.dumps(
                    {
                        "entity_id": entity["entity_id"],
                        "entity_name": entity["entity_name"],
                        "raw_response": debug["raw_response"],
                        "first_mention_position": debug["first_mentions"].get(entity["entity_id"]),
                        "rank": rank_by_entity_id.get(entity["entity_id"]),
                        "mean_reciprocal_rank": reciprocal_ranks.get(entity["entity_id"], 0.0),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for entity in entities
        )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))

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
        )

        sentiment_polarity = _score_sentiment_polarity(output.text)
        ad_likelihood, ad_score_by_label = _score_ad_likelihood(output.text)

        return {
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

        tasks.extend(
            {
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

    rows = [
        {
            "assay": description["assay"],
            "assay_instance_hash": description["assay_instance_hash"],
            "model": description["model"],
            "comparison_set_id": description["comparison_set_id"],
            "comparison_set_name": description["comparison_set_name"],
            "entity_id": description["entity_id"],
            "entity_name": description["entity_name"],
            "result": [
                {
                    "estimand": "sentiment_polarity",
                    "value": str(description["sentiment_polarity"]),
                },
                {
                    "estimand": "ad_likelihood",
                    "value": str(description["ad_likelihood"]),
                },
            ],
            "debug_json": json.dumps(
                {
                    "entity_id": description["entity_id"],
                    "entity_name": description["entity_name"],
                    "description": description["description"],
                    "sentiment_polarity": description["sentiment_polarity"],
                    "ad_likelihood": description["ad_likelihood"],
                    "ad_score_by_label": description["ad_score_by_label"],
                    "ad_positive_label": description["ad_positive_label"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for description in descriptions
    ]

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)


ASSAY_DELEGATES: Mapping[Assay, AssayDelegate] = {
    "head-to-head": run_head_to_head,
    "rank": run_rank,
    "consideration-set": run_consideration_set,
    "describe-sentiment": run_describe_sentiment,
}


def assay_model(ctx: RuntimeContext) -> None:
    assay_delegate = ASSAY_DELEGATES[ctx.cfg.assay]

    logging.info(f"Running assay={ctx.cfg.assay} model={ctx.cfg.model}.")

    assay_df = assay_delegate(ctx)

    _save_assay_df(assay_df, ctx.cfg.save)

    logging.info(f"Saved assay results to {ctx.cfg.save}.")