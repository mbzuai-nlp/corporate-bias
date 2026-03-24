import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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


_SENTIMENT_PIPELINE = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

_AD_MODEL_NAME = "teknology/ad-classifier-v0.3"
_AD_TOKENIZER = AutoTokenizer.from_pretrained(_AD_MODEL_NAME)
_AD_MODEL = AutoModelForSequenceClassification.from_pretrained(_AD_MODEL_NAME)
_AD_MODEL.eval()

_AD_LABELS = {
    int(label_id): label_name
    for label_id, label_name in _AD_MODEL.config.id2label.items()
}
_AD_POSITIVE_LABEL = "LABEL_1"


def _score_sentiment_polarity(text: str) -> float:
    scores = _SENTIMENT_PIPELINE(text, truncation=True, top_k=None)
    score_by_label = {item["label"]: float(item["score"]) for item in scores}
    return score_by_label["POSITIVE"] - score_by_label["NEGATIVE"]


def _score_ad_likelihood(text: str) -> tuple[float, dict[str, float]]:
    inputs = _AD_TOKENIZER(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _AD_MODEL(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    score_by_label = {
        _AD_LABELS[i]: float(probs[i].item()) for i in range(len(_AD_LABELS))
    }

    return score_by_label[_AD_POSITIVE_LABEL], score_by_label


def run_describe_sentiment(ctx: RuntimeContext) -> pl.DataFrame:
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
            seed=task["sample_id"],
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

    entity_lookup = build_entity_lookup(entity_df)

    assay_instances = list(
        comparison_set_assay_instance_df.filter(pl.col("assay") == ctx.cfg.assay)
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

        entities = get_comparison_set_entities(
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
        entities = entities_by_instance[instance_hash]

        for entity in entities:
            samples = sorted(
                descriptions_by_instance_and_entity[
                    (instance_hash, entity["entity_id"])
                ],
                key=lambda row: row["sample_id"],
            )

            sentiment_values = [sample["sentiment_polarity"] for sample in samples]
            ad_values = [sample["ad_likelihood"] for sample in samples]

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
