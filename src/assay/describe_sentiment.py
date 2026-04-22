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

_STANCE_MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
_STANCE_TOKENIZER = AutoTokenizer.from_pretrained(_STANCE_MODEL_NAME)
_STANCE_MODEL = AutoModelForSequenceClassification.from_pretrained(_STANCE_MODEL_NAME)
_STANCE_MODEL.eval()
_STANCE_MODEL.to(_DEVICE)

_AD_MODEL_NAME = "teknology/ad-classifier-v0.3"
_AD_TOKENIZER = AutoTokenizer.from_pretrained(_AD_MODEL_NAME)
_AD_MODEL = AutoModelForSequenceClassification.from_pretrained(_AD_MODEL_NAME)
_AD_MODEL.eval()
_AD_MODEL.to(_DEVICE)

_AD_LABELS = {
    int(label_id): label_name
    for label_id, label_name in _AD_MODEL.config.id2label.items()
}
_AD_POSITIVE_LABEL = "LABEL_1"

_LOCAL_BATCH_SIZE = 32


T = TypeVar("T")


def _batched(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _find_entailment_label_id(model: AutoModelForSequenceClassification) -> int:
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}

    for key, value in label2id.items():
        if "entail" in key:
            return value

    raise ValueError(
        f"Could not find entailment label in label2id={model.config.label2id!r}"
    )


_STANCE_ENTAILMENT_LABEL_ID = _find_entailment_label_id(_STANCE_MODEL)


def _score_sentiment_polarity_batch(texts: list[str]) -> list[float]:
    if not texts:
        return []

    results: list[float] = []
    text_batches = _batched(texts, _LOCAL_BATCH_SIZE)

    for text_batch in tqdm(
        text_batches,
        total=len(text_batches),
        desc="Sentiment batches",
    ):
        batch_scores = _SENTIMENT_PIPELINE(
            text_batch,
            truncation=True,
            top_k=None,
            batch_size=len(text_batch),
        )

        for scores in batch_scores:
            score_by_label = {item["label"]: float(item["score"]) for item in scores}
            results.append(score_by_label["POSITIVE"] - score_by_label["NEGATIVE"])

    return results


def _score_ad_likelihood_batch(
    texts: list[str],
) -> list[tuple[float, dict[str, float]]]:
    if not texts:
        return []

    device = _get_model_device(_AD_MODEL)
    results: list[tuple[float, dict[str, float]]] = []
    text_batches = _batched(texts, _LOCAL_BATCH_SIZE)

    for text_batch in tqdm(
        text_batches,
        total=len(text_batches),
        desc="Ad batches",
    ):
        inputs = _AD_TOKENIZER(
            text_batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = _AD_MODEL(**inputs).logits

        probs = torch.softmax(logits, dim=-1)

        for row_probs in probs:
            score_by_label = {
                _AD_LABELS[i]: float(row_probs[i].item()) for i in range(len(_AD_LABELS))
            }
            results.append((score_by_label[_AD_POSITIVE_LABEL], score_by_label))

    return results


def _score_stance_batch(
    texts: list[str],
    entity_names: list[str],
) -> list[tuple[float, dict[str, float]]]:
    if not texts:
        return []

    if len(texts) != len(entity_names):
        raise ValueError("texts and entity_names must have the same length")

    device = _get_model_device(_STANCE_MODEL)

    # For each text we create 3 hypotheses:
    #   favors X, disfavors X, neutral toward X
    # Then we batch all premise/hypothesis pairs through the NLI model.
    pair_premises: list[str] = []
    pair_hypotheses: list[str] = []
    candidate_labels_by_text: list[list[str]] = []

    for text, entity_name in zip(texts, entity_names, strict=True):
        favor_label = f"favors {entity_name}"
        disfavor_label = f"disfavors {entity_name}"
        neutral_label = f"is neutral toward {entity_name}"

        candidate_labels = [favor_label, disfavor_label, neutral_label]
        candidate_labels_by_text.append(candidate_labels)

        for label in candidate_labels:
            pair_premises.append(text)
            pair_hypotheses.append(f"This text {label}.")

    pair_indices = list(range(len(pair_premises)))
    pair_index_batches = _batched(pair_indices, _LOCAL_BATCH_SIZE)

    entailment_scores: list[float] = []

    for index_batch in tqdm(
        pair_index_batches,
        total=len(pair_index_batches),
        desc="Stance batches",
    ):
        premise_batch = [pair_premises[i] for i in index_batch]
        hypothesis_batch = [pair_hypotheses[i] for i in index_batch]

        inputs = _STANCE_TOKENIZER(
            premise_batch,
            hypothesis_batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = _STANCE_MODEL(**inputs).logits

        batch_entailment_scores = logits[:, _STANCE_ENTAILMENT_LABEL_ID]
        entailment_scores.extend(batch_entailment_scores.detach().cpu().tolist())

    results: list[tuple[float, dict[str, float]]] = []

    # Reconstruct per-text scores from groups of 3 entailment logits
    for i, candidate_labels in enumerate(candidate_labels_by_text):
        start = i * 3
        group_logits = torch.tensor(entailment_scores[start : start + 3], dtype=torch.float)
        group_probs = torch.softmax(group_logits, dim=0)

        score_by_label = {
            label: float(prob.item())
            for label, prob in zip(candidate_labels, group_probs, strict=True)
        }

        favor_label, disfavor_label, _neutral_label = candidate_labels
        stance_score = score_by_label[favor_label] - score_by_label[disfavor_label]
        results.append((stance_score, score_by_label))

    return results


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
        plugins=[{"id": "response-healing"}],
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
                    "ad_score_by_label": sample["ad_score_by_label"],
                    "ad_positive_label": sample["ad_positive_label"],
                    "stance_score": sample["stance_score"],
                    "stance_score_by_label": sample["stance_score_by_label"],
                }
                for sample in samples
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _score_descriptions_in_batches(descriptions: list[dict[str, Any]]) -> None:
    if not descriptions:
        return

    texts = [description["description"] for description in descriptions]
    entity_names = [description["entity_name"] for description in descriptions]

    sentiment_scores = _score_sentiment_polarity_batch(texts)
    ad_results = _score_ad_likelihood_batch(texts)
    stance_results = _score_stance_batch(texts, entity_names)

    for (
        description,
        sentiment_polarity,
        (ad_likelihood, ad_score_by_label),
        (stance_score, stance_score_by_label),
    ) in zip(
        descriptions,
        sentiment_scores,
        ad_results,
        stance_results,
        strict=True,
    ):
        description["sentiment_polarity"] = sentiment_polarity
        description["ad_likelihood"] = ad_likelihood
        description["ad_score_by_label"] = ad_score_by_label
        description["ad_positive_label"] = _AD_POSITIVE_LABEL
        description["stance_score"] = stance_score
        description["stance_score_by_label"] = stance_score_by_label


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

    _score_descriptions_in_batches(descriptions)

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