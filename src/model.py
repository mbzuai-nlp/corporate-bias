from functools import partial
from typing import Literal, Mapping, Sequence, Any, Protocol
from openrouter import OpenRouter, errors as or_errors
import os
from dataclasses import dataclass
import hashlib
import json
import pickle
import sqlite3
import threading
import logging
import backoff
import httpx
import jsonschema


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


Model = Literal[
    "gpt-oss-120b",
    "gpt-5.4",
    "gpt-4o-mini",
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "grok-4.1-fast",
    "grok-4",
    "llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct",
    "mistral-nemo",
    "mistral-small-2603",
    "deepseek-v3.2",
    "qwen3-235b-a22b-2507",
    "qwen3.5-flash-02-23",
    "nemotron-3-super-120b-a12b:free",
    "nemotron-3-nano-30b-a3b:free",
    "phi-4",
]


# === DEPENDENCY INJECTION ===


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
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response!r}") from e

    if not isinstance(content, str):
        raise TypeError(
            f"Expected text content to be str, got {type(content).__name__}: {content!r}"
        )

    if not content.strip():
        raise TypeError("Model returned empty content")

    return content


def _send_openrouter_request(
    client: OpenRouter,
    model_name: str,
    messages: Sequence[Message],
    **kwargs: Any,
) -> Any:
    return client.chat.send(
        model=model_name,
        messages=_serialize_messages(messages),
        timeout_ms=30000,
        provider={
            # "require_parameters": True,
        },
        **kwargs,
    )


def _disable_web_plugin(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)

    raw_plugins = out.get("plugins")
    plugins: list[Any]

    if raw_plugins is None:
        plugins = []
    else:
        plugins = list(raw_plugins)

    updated = []
    saw_web = False

    for plugin in plugins:
        if isinstance(plugin, dict) and plugin.get("id") == "web":
            p = dict(plugin)
            p["enabled"] = False
            updated.append(p)
            saw_web = True
        else:
            updated.append(plugin)

    if not saw_web:
        updated.append({"id": "web", "enabled": False})

    out["plugins"] = updated
    return out


class InvalidModelOutputError(RuntimeError):
    pass


def _validate_structured_output(text: str, kwargs: dict[str, Any]) -> None:
    response_format = kwargs.get("response_format")
    if not response_format:
        return

    response_type = response_format.get("type")

    if response_type == "json_object":
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise InvalidModelOutputError(
                f"Model returned invalid JSON: {text!r}"
            ) from e
        return

    if response_type == "json_schema":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise InvalidModelOutputError(
                f"Model returned invalid JSON for json_schema response: {text!r}"
            ) from e

        json_schema = response_format.get("json_schema") or {}
        schema = json_schema.get("schema")
        schema_name = json_schema.get("name", "unnamed_schema")

        if schema is None:
            raise InvalidModelOutputError(
                "response_format.type='json_schema' but no schema was provided"
            )

        try:
            jsonschema.Draft202012Validator(schema).validate(parsed)
        except jsonschema.ValidationError as e:
            raise InvalidModelOutputError(
                f"Model returned JSON that failed schema validation for {schema_name}: {e.message}"
            ) from e


@backoff.on_exception(
    backoff.constant,
    (
        httpx.HTTPError,
        httpx.TransportError,
        or_errors.TooManyRequestsResponseError,
        or_errors.InternalServerResponseError,
        or_errors.BadGatewayResponseError,
        or_errors.ServiceUnavailableResponseError,
        or_errors.EdgeNetworkTimeoutResponseError,
        or_errors.ProviderOverloadedResponseError,
        or_errors.ResponseValidationError,
        TypeError,
        InvalidModelOutputError,
    ),
    interval=30,
    max_tries=3,
    jitter=None,
)
def _invoke_openrouter_model(
    model_name: str,
    messages: Sequence[Message],
    **kwargs: Any,
) -> ModelOutput:
    client = _get_openrouter_client()

    use_cache = bool(kwargs.pop("use_cache", False))
    kwargs = _disable_web_plugin(kwargs)

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

    response = _send_openrouter_request(
        client=client,
        model_name=model_name,
        messages=messages,
        **kwargs,
    )

    output = ModelOutput(
        text=_extract_text(response),
        raw=response,
    )

    _validate_structured_output(output.text, kwargs)

    if use_cache:
        _cache_set_obj(cache_key, output)

    return output


# === MODEL DELEGATES ===


MODEL_DELEGATES: Mapping[Model, ModelDelegate] = {
    "gpt-oss-120b": partial(_invoke_openrouter_model, "openai/gpt-oss-120b"),
    "gpt-5.4": partial(_invoke_openrouter_model, "openai/gpt-5.4"),
    "gpt-4o-mini": partial(_invoke_openrouter_model, "openai/gpt-4o-mini"),
    "claude-sonnet-4.6": partial(
        _invoke_openrouter_model, "anthropic/claude-sonnet-4.6"
    ),
    "claude-opus-4.6": partial(_invoke_openrouter_model, "anthropic/claude-opus-4.6"),
    "gemini-2.5-flash": partial(_invoke_openrouter_model, "google/gemini-2.5-flash"),
    "gemini-2.5-pro": partial(_invoke_openrouter_model, "google/gemini-2.5-pro"),
    "grok-4.1-fast": partial(_invoke_openrouter_model, "x-ai/grok-4.1-fast"),
    "grok-4": partial(_invoke_openrouter_model, "x-ai/grok-4"),
    "llama-3.1-8b-instruct": partial(
        _invoke_openrouter_model, "meta-llama/llama-3.1-8b-instruct"
    ),
    "llama-3.1-70b-instruct": partial(
        _invoke_openrouter_model, "meta-llama/llama-3.1-70b-instruct"
    ),
    "mistral-nemo": partial(_invoke_openrouter_model, "mistralai/mistral-nemo"),
    "mistral-small-2603": partial(
        _invoke_openrouter_model, "mistralai/mistral-small-2603"
    ),
    "deepseek-v3.2": partial(_invoke_openrouter_model, "deepseek/deepseek-v3.2"),
    "qwen3-235b-a22b-2507": partial(
        _invoke_openrouter_model, "qwen/qwen3-235b-a22b-2507"
    ),
    "qwen3.5-flash-02-23": partial(
        _invoke_openrouter_model, "qwen/qwen3.5-flash-02-23"
    ),
    "nemotron-3-super-120b-a12b": partial(
        _invoke_openrouter_model, "nvidia/nemotron-3-super-120b-a12b"
    ),
    "nemotron-3-nano-30b-a3b": partial(
        _invoke_openrouter_model, "nvidia/nemotron-3-nano-30b-a3b"
    ),
    "phi-4": partial(_invoke_openrouter_model, "microsoft/phi-4"),
}


# === PUBLIC FUNCTIONS ===


def invoke_model(
    model: Model, messages: Sequence[Message], **kwargs: Any
) -> ModelOutput:
    try:
        model_delegate = MODEL_DELEGATES[model]
    except KeyError as e:
        raise ValueError(f"Unsupported model: {model}.") from e

    logging.info(f"Invoking {model}.")
    return model_delegate(messages, **kwargs)
