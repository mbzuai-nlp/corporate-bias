from typing import Literal, Mapping, Sequence, Any, Protocol
from openrouter import OpenRouter
import os
from dataclasses import dataclass
import hashlib
import json
import pickle
import sqlite3
import threading
import logging


# === TYPES ===


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ModelOutput:
    text: str
    raw: Any


class ModelDelegate(Protocol):
    def __call__(self, messages: Sequence[Message], **kwargs: Any) -> ModelOutput: ...


Model = Literal["gpt5"]


# === DEPENDENCY INJECTION ===


# lazy loaded client singleton
_openrouter_client: OpenRouter | None = None


# === CACHE ===


_db_path = ".llm_response_cache.sqlite"
_tls = threading.local()
_db_lock = threading.Lock()


def _get_cache_conn() -> sqlite3.Connection:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v BLOB NOT NULL)"
        )
        conn.commit()
        _tls.conn = conn
    return conn


def _cache_get_obj(key: str) -> Any | None:
    row = (
        _get_cache_conn()
        .execute(
            "SELECT v FROM cache WHERE k=?",
            (key,),
        )
        .fetchone()
    )

    if row is None:
        return None

    return pickle.loads(row[0])


def _cache_set_obj(key: str, value: Any) -> None:
    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    with _db_lock:
        conn = _get_cache_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
            (key, sqlite3.Binary(blob)),
        )
        conn.commit()


def _cache_key_for_openrouter_call(
    model_name: str,
    messages: Sequence[Message],
    kwargs: dict[str, Any],
) -> str:
    messages_text = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    kwargs_text = json.dumps(
        kwargs,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    raw_key = f"{model_name}\n{messages_text}\n{kwargs_text}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# === HELPERS ===


def _construct_openrouter_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")

    return OpenRouter(api_key=api_key)


def _get_openrouter_client() -> OpenRouter:
    global _openrouter_client

    if _openrouter_client is None:
        _openrouter_client = _construct_openrouter_client()

    return _openrouter_client


def _serialize_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _extract_text(response: Any) -> str:
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response!r}") from e


def _invoke_openrouter_model(
    model_name: str,
    messages: Sequence[Message],
    **kwargs: Any,
) -> ModelOutput:
    client = _get_openrouter_client()

    use_cache = bool(kwargs.pop("use_cache", False))
    cache_key = None

    if use_cache:
        cache_key = _cache_key_for_openrouter_call(
            model_name=model_name,
            messages=messages,
            kwargs=kwargs,
        )
        cached = _cache_get_obj(cache_key)
        if cached is not None:
            return cached

    response = client.chat.send(
        model=model_name,
        messages=_serialize_messages(messages),
        **kwargs,
    )

    output = ModelOutput(
        text=_extract_text(response),
        raw=response,
    )

    if use_cache:
        _cache_set_obj(cache_key, output)

    return output


# === MODEL DELEGATES ===


def invoke_gpt5(messages: Sequence[Message], **kwargs: Any) -> ModelOutput:
    return _invoke_openrouter_model(
        "openai/gpt-5",
        messages,
        **kwargs,
    )


def invoke_gemini(messages: Sequence[Message], **kwargs: Any) -> ModelOutput:
    return _invoke_openrouter_model(
        "google/gemini-2.5-flash",
        messages,
        **kwargs,
    )


MODEL_DELEGATES: Mapping[Model, ModelDelegate] = {
    "gpt5": invoke_gpt5,
    "gemini": invoke_gemini
}


# === PUBLIC FUNCTIONS ===


def invoke_model(
    model: Model, 
    messages: Sequence[Message], 
    **kwargs: Any
) -> ModelOutput:
    try:
        model_delegate = MODEL_DELEGATES[model]
    except KeyError as e:
        raise ValueError(f"Unsupported model: {model}.") from e

    logging.info(f"Invoking {model}.")
    return model_delegate(messages, **kwargs)