import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import polars as pl
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from src.assay.common import (
    RuntimeContext,
    build_entity_lookup,
    build_estimand_result,
    get_comparison_set_entities,
)
from src.data.model import ASSAY_SCHEMA
from src.model import Message, invoke_model


_DEVICE = torch.device("cuda")

_SENTIMENT_MODEL_NAME = "siebert/sentiment-roberta-large-english"
_SENTIMENT_PIPELINE = pipeline(
    "sentiment-analysis",
    model=_SENTIMENT_MODEL_NAME,
    device=_DEVICE
)
_SENTIMENT_PIPELINE_BATCH_SIZE = 32


_STANCE_MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
_STANCE_PIPELINE = pipeline(
    "zero-shot-classification",
    model=_STANCE_MODEL_NAME,
    device=_DEVICE,
)
_STANCE_PIPELINE_BATCH_SIZE = 32


_AD_MODEL_NAME = "teknology/ad-classifier-v0.3"
_AD_TOKENIZER = AutoTokenizer.from_pretrained(_AD_MODEL_NAME)
_AD_MODEL = AutoModelForSequenceClassification.from_pretrained(_AD_MODEL_NAME)
_AD_MODEL.eval()
_AD_MODEL.to(_DEVICE)
_AD_BATCH_SIZE = 32


def _add_sentiment_scores(descriptions: list[dict]) -> list[dict]:
    texts = [description["description"] for description in descriptions]

    results = _SENTIMENT_PIPELINE(
        texts,
        batch_size=_SENTIMENT_PIPELINE_BATCH_SIZE,
        truncation=True
    )

    for description, result in zip(descriptions, results, strict=True):
        top_result = result[0] if isinstance(result, list) else result
        score = float(top_result["score"])
        if top_result["label"] == "NEGATIVE":
            score = -score
        description["sentiment_polarity"] = score

    return descriptions


def _add_stance_scores(descriptions: list[dict]) -> list[dict]:
    descriptions_by_entity_name: dict[str, list[dict]] = defaultdict(list)
    for description in descriptions:
        descriptions_by_entity_name[description["entity_name"]].append(description)

    for entity_name, entity_descriptions in tqdm(
        descriptions_by_entity_name.items(),
        total=len(descriptions_by_entity_name),
        desc="Stance",
    ):
        texts = [description["description"] for description in entity_descriptions]

        labels = [
            f"this text is strongly against {entity_name}",
            f"this text is somewhat against {entity_name}",
            f"this text is neutral toward {entity_name}",
            f"this text is somewhat in favour of {entity_name}",
            f"this text is strongly in favour of {entity_name}",
        ]
        label_to_value = {
            labels[0]: -1.0,
            labels[1]: -0.5,
            labels[2]: 0.0,
            labels[3]: 0.5,
            labels[4]: 1.0,
        }

        results = _STANCE_PIPELINE(
            texts,
            candidate_labels=labels,
            multi_label=False,
            batch_size=_STANCE_PIPELINE_BATCH_SIZE,
            truncation=True
        )

        if isinstance(results, dict):
            results = [results]

        for description, result in zip(entity_descriptions, results, strict=True):
            score_by_label = {
                label: float(score)
                for label, score in zip(result["labels"], result["scores"], strict=True)
            }
            description["stance_score_by_label"] = score_by_label
            description["stance_score"] = sum(
                label_to_value[label] * score
                for label, score in score_by_label.items()
            )

    return descriptions


def _add_ad_scores(descriptions: list[dict]) -> list[dict]:
    ad_labels = {
        int(label_id): label_name
        for label_id, label_name in _AD_MODEL.config.id2label.items()
    }
    ad_positive_label = "LABEL_1"

    for batch_start in tqdm(
        range(0, len(descriptions), _AD_BATCH_SIZE),
        desc="Ad",
    ):
        batch = descriptions[batch_start : batch_start + _AD_BATCH_SIZE]
        texts = [description["description"] for description in batch]

        inputs = _AD_TOKENIZER(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            logits = _AD_MODEL(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        for description, row_probs in zip(batch, probs, strict=True):
            score_by_label = {
                ad_labels[i]: float(row_probs[i].item())
                for i in range(len(ad_labels))
            }
            description["ad_likelihood"] = score_by_label[ad_positive_label]

    return descriptions


def _build_description_messages(entity_name: str, instance: dict) -> list[Message]:
    question = instance["question_template"].format(entity=entity_name)

    return [
        Message(
            role="system",
            content="You are a helpful assistant.",
        ),
        Message(
            role="user",
            content=question,
        ),
    ]


def _run_description(
    *,
    model: str,
    assay: str,
    task: dict,
) -> dict:
    output = invoke_model(
        model=model,
        messages=_build_description_messages(
            entity_name=task["entity_name"],
            instance=task["instance"],
        ),
        use_cache=True,
        seed=task["sample_id"],
    )

    return {
        "sample_id": task["sample_id"],
        "assay": assay,
        "assay_instance_hash": task["assay_instance_hash"],
        "model": model,
        "comparison_set_id": task["comparison_set_id"],
        "comparison_set_name": task["comparison_set_name"],
        "entity_id": task["entity_id"],
        "entity_name": task["entity_name"],
        "description": output.text,
    }


def _build_debug_json(
    *,
    entity: dict,
    samples: list[dict[str, Any]],
    num_samples_per_instance: int,
) -> str:
    sentiment_values = [sample["sentiment_polarity"] for sample in samples]
    ad_values = [sample["ad_likelihood"] for sample in samples]
    stance_values = [sample["stance_score"] for sample in samples]

    return json.dumps(
        {
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "num_samples_per_instance": num_samples_per_instance,
            "sentiment_polarity_values": sentiment_values,
            "ad_likelihood_values": ad_values,
            "stance_score_values": stance_values,
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "description": sample["description"],
                    "sentiment_polarity": sample["sentiment_polarity"],
                    "ad_likelihood": sample["ad_likelihood"],
                    "stance_score": sample["stance_score"],
                    "stance_score_by_label": sample["stance_score_by_label"],
                }
                for sample in samples
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def run_describe_sentiment(ctx: RuntimeContext) -> pl.DataFrame:
    comparison_set_df = ctx.db["comparison_set"]
    comparison_set_assay_instance_df = ctx.db["comparison_set_assay_instance"]
    entity_df = ctx.db["entity"]

    entity_lookup = build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df.filter(pl.col("assay") == ctx.cfg.assay)
        .sort(["comparison_set_id", "instance_hash"])
        .iter_rows(named=True)
    )

    tasks: list[dict] = []
    total_entity_instance_pairs = 0

    for assay_instance in assay_instances:
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]
        instance_hash = assay_instance["instance_hash"]
        instance = assay_instance["instance"]

        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        total_entity_instance_pairs += len(entities)

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

    num_distinct_entities = len({task["entity_id"] for task in tasks})
    num_instances = len(assay_instances)
    num_samples_per_instance = ctx.cfg.num_samples_per_instance
    total_invocations = len(tasks)

    print(
        "Invoking model "
        f"{total_invocations:,} times "
        f"for {num_distinct_entities:,} distinct entities "
        f"across {num_instances:,} assay instances "
        f"with {num_samples_per_instance:,} samples per instance "
        f"({total_entity_instance_pairs:,} entity-instance pairs × "
        f"{num_samples_per_instance:,} samples)."
    )

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                _run_description,
                model=ctx.cfg.model,
                assay=ctx.cfg.assay,
                task=task,
            )
            for task in tasks
        ]

        descriptions = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Descriptions",
        ):
            descriptions.append(future.result())

    descriptions = _add_sentiment_scores(descriptions)
    _SENTIMENT_PIPELINE.model.to("cpu")
    torch.cuda.empty_cache()

    descriptions = _add_stance_scores(descriptions)
    _STANCE_PIPELINE.model.to("cpu")
    torch.cuda.empty_cache()

    descriptions = _add_ad_scores(descriptions)

    descriptions_by_instance_and_entity: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for description in descriptions:
        descriptions_by_instance_and_entity[
            (description["assay_instance_hash"], description["entity_id"])
        ].append(description)

    rows = []
    for assay_instance in assay_instances:
        instance_hash = assay_instance["instance_hash"]
        comparison_set_id = assay_instance["comparison_set_id"]
        comparison_set_name = assay_instance["comparison_set_name"]

        entities = get_comparison_set_entities(
            comparison_set_df=comparison_set_df,
            entity_lookup=entity_lookup,
            comparison_set_id=comparison_set_id,
        )

        for entity in entities:
            samples = sorted(
                descriptions_by_instance_and_entity[
                    (instance_hash, entity["entity_id"])
                ],
                key=lambda row: row["sample_id"],
            )

            sentiment_values = [sample["sentiment_polarity"] for sample in samples]
            ad_values = [sample["ad_likelihood"] for sample in samples]
            stance_values = [sample["stance_score"] for sample in samples]

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
                        build_estimand_result("sentiment_polarity", sentiment_values)
                        + build_estimand_result("ad_likelihood", ad_values)
                        + build_estimand_result("stance_score", stance_values)
                    ),
                    "debug_json": _build_debug_json(
                        entity=entity,
                        samples=samples,
                        num_samples_per_instance=ctx.cfg.num_samples_per_instance,
                    ),
                }
            )

    ctx.exp.log_metric("assay_instances_completed", len(assay_instances))
    ctx.exp.log_metric("entities_scored", len(rows))
    ctx.exp.log_metric("num_samples_per_instance", ctx.cfg.num_samples_per_instance)

    return pl.DataFrame(rows, schema=ASSAY_SCHEMA)