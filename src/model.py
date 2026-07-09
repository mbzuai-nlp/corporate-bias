from copy import deepcopy
from functools import partial
from typing import Literal, Mapping, Sequence, Any, Protocol, Callable, Optional
from openrouter import OpenRouter, errors as or_errors
import os
from dataclasses import dataclass
import hashlib
import json
import pickle
import sqlite3
import threading
import logging
from tenacity import retry, stop_after_attempt, retry_if_exception_type
import httpx
import jsonschema
from func_timeout import func_timeout, FunctionTimedOut


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


class InvalidModelOutputError(RuntimeError):
    pass


class RetryableNetworkError(RuntimeError):
    def __init__(self, message: str, seconds: Optional[int] = None):
        super().__init__(message)
        self.seconds = seconds


RETRYABLE_NETWORK_ERRORS = (
    httpx.HTTPError,
    httpx.TransportError,
    or_errors.TooManyRequestsResponseError,
    or_errors.InternalServerResponseError,
    or_errors.BadGatewayResponseError,
    or_errors.ServiceUnavailableResponseError,
    or_errors.EdgeNetworkTimeoutResponseError,
    or_errors.ProviderOverloadedResponseError,
    or_errors.ResponseValidationError,
)


# === DEPENDENCY INJECTION ===


_openrouter_client: OpenRouter | None = None


_openrouter_vertex_client: OpenRouter | None = None


def _get_openrouter_client() -> OpenRouter:
    global _openrouter_client

    if _openrouter_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")
        _openrouter_client = OpenRouter(api_key=api_key)

    return _openrouter_client


def _get_openrouter_vertex_client() -> OpenRouter:
    global _openrouter_vertex_client

    if _openrouter_vertex_client is None:
        api_key = os.environ.get("OPENROUTER_VERTEX_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")
        _openrouter_vertex_client = OpenRouter(api_key=api_key)

    return _openrouter_vertex_client


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


def _cache_key(
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


def _serialize_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _extract_text_from_model_output(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response!r}") from e

    if not isinstance(content, str):
        raise InvalidModelOutputError(
            f"Expected text content to be str, got {response}"
        )

    if not content.strip():
        raise InvalidModelOutputError("Model returned empty content")

    return content


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


CANONICAL_OPENROUTER_JSON_SCHEMA_RULES: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "strip_keywords": {"minItems", "maxItems", "uniqueItems"},
    },
    "google/gemma-4-31b-it": {
        "strip_keywords": {"uniqueItems",},
    },
    "anthropic/claude-sonnet-5": {
        "strip_keywords": {"minItems", "maxItems", "uniqueItems", "minimum", "maximum"},
    },
}


def _strip_schema_keywords(node: Any, keywords_to_strip: set[str]) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_schema_keywords(v, keywords_to_strip)
            for k, v in node.items()
            if k not in keywords_to_strip
        }
    if isinstance(node, list):
        return [_strip_schema_keywords(item, keywords_to_strip) for item in node]
    return node


def canonicalise_openrouter_json_schema(
    *,
    model_name: str,
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Return the provider-facing response_format for an OpenRouter request. Some
    OpenRouter providers/models reject otherwise-valid JSON Schema keywords
    even though we still want to validate against the original schema locally.
    """
    if response_format is None:
        return None

    out = deepcopy(response_format)

    if out.get("type") != "json_schema":
        return out

    json_schema_payload = out.get("json_schema")
    if not isinstance(json_schema_payload, dict):
        return out

    schema = json_schema_payload.get("schema")
    if schema is None:
        return out

    rules = CANONICAL_OPENROUTER_JSON_SCHEMA_RULES.get(model_name, {})
    strip_keywords = set(rules.get("strip_keywords", set()))

    if strip_keywords:
        json_schema_payload["schema"] = _strip_schema_keywords(schema, strip_keywords)

    return out


def _validate_structured_output(
    text: str,
    response_format: dict[str, Any] | None,
) -> None:
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

        json_schema_payload = response_format.get("json_schema") or {}
        schema = json_schema_payload.get("schema")
        schema_name = json_schema_payload.get("name", "unnamed_schema")

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
        

def _invoke_openrouter_model(
    client_getter: Callable[[], OpenRouter],
    model_name: str,
    messages: Sequence[Message],
    **kwargs: Any,
) -> ModelOutput:
    client = client_getter()

    original_response_format = kwargs.get("response_format")

    request_response_format = canonicalise_openrouter_json_schema(
        model_name=model_name,
        response_format=original_response_format,
    )

    request_kwargs = dict(kwargs)

    if original_response_format is not None:
        request_kwargs["response_format"] = request_response_format

    request_kwargs = _disable_web_plugin(request_kwargs)

    try:
        # If fully determinstic, adding timeout will invariably cause fatality
        response = client.chat.send(
            model=model_name,
            messages=_serialize_messages(messages),
            **request_kwargs,
        )
    except or_errors.TooManyRequestsResponseError as e:
        wait = int(e.headers.get("retry-after")) + 1 if "retry-after" in e.headers else None
        err_str = json.dumps({"message": e.message, "headers": dict(e.headers), 
                              "body": e.body, "request_kwargs": request_kwargs, 
                              "should_wait_seconds": wait})
        raise RetryableNetworkError(err_str, wait) from e
    except RETRYABLE_NETWORK_ERRORS as e:
        err_str = json.dumps({"message": e.message, "headers": dict(e.headers), 
                              "body": e.body, "request_kwargs": request_kwargs})
        raise RetryableNetworkError(err_str) from e
    except Exception as e:
        err_str = json.dumps({"message": e.message, "headers": dict(e.headers), 
                              "body": e.body, "request_kwargs": request_kwargs})
        raise RuntimeError(err_str) from e

    output = ModelOutput(
        text=_extract_text_from_model_output(response),
        raw={
            "input_messages": messages,
            "request_kwargs": request_kwargs,
            "response": response
        }
    )

    return output


# === MODEL DELEGATES ===


DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0,
    "seed": 0
}


MODEL_DELEGATES: Mapping[str, ModelDelegate] = {
    "gpt-oss-120b": partial(
        _invoke_openrouter_model,
        _get_openrouter_client,
        "openai/gpt-oss-120b",
        provider={"only": ["cerebras"], "quantizations": ["fp16"]},
        reasoning={"effort": "minimal"},
        **DEFAULT_SAMPLING_PARAMS
    ),
    "gpt-5.4": partial(
        _invoke_openrouter_model,
        _get_openrouter_client,
        "openai/gpt-5.4",
        provider={"only": ["openai"]},
        reasoning={"effort": "none"},
        **DEFAULT_SAMPLING_PARAMS
    ),
    # "gpt-4o-mini": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "openai/gpt-4o-mini",
    #     provider={"only": ["openai"]},
    #     reasoning={"effort": "none"},
    #     **DEFAULT_SAMPLING_PARAMS
    # ),
    "claude-sonnet-5": partial(
        _invoke_openrouter_model,
        _get_openrouter_client,
        "anthropic/claude-sonnet-5",
        provider={"only": ["anthropic"]},
        reasoning={"effort": "none"},
        **DEFAULT_SAMPLING_PARAMS
    ),
    # "claude-opus-4.6": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "anthropic/claude-opus-4.6",
    #     provider={"only": ["anthropic"]},
    #     reasoning={"effort": "none"},
    #     **DEFAULT_SAMPLING_PARAMS
    # ),
    # "gemma-4-31b-it": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_vertex_client,
    #     "google/gemma-4-31b-it",
    #     provider={"only": ["venice"], "quantizations": ["bf16"]},
    #     reasoning={"effort": "none"},
    #     **DEFAULT_SAMPLING_PARAMS
    # ),
    "gemini-3.5-flash": partial(
        _invoke_openrouter_model,
        _get_openrouter_vertex_client,
        "google/gemini-3.5-flash",
        provider={"only": ["google-vertex"], "regions": ["global"]},
        reasoning={"effort": "minimal"},
        **DEFAULT_SAMPLING_PARAMS
    ),
    # "gemini-2.5-pro": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_vertex_client,
    #     "google/gemini-2.5-pro",
    #     provider={"only": ["google-vertex"], "regions": ["global"]},
    #     reasoning={"effort": "minimal"},
    #     **DEFAULT_SAMPLING_PARAMS
    # ),
    # "grok-4.20": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "x-ai/grok-4.20",
    #     reasoning={"effort": "none"},
    # ),
    # "grok-4.3": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "x-ai/grok-4.3",
    #     reasoning={"effort": "none"},
    # ),
    # "llama-3.1-8b-instruct": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "meta-llama/llama-3.1-8b-instruct",
    #     reasoning={"effort": "none"},
    # ),
    # "llama-3.1-70b-instruct": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "meta-llama/llama-3.1-70b-instruct",
    #     reasoning={"effort": "none"},
    # ),
    # "mistral-nemo": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "mistralai/mistral-nemo",
    #     reasoning={"effort": "none"},
    # ),
    # "mistral-small-2603": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "mistralai/mistral-small-2603",
    #     reasoning={"effort": "none"},
    # ),
    # "deepseek-v3.2": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "deepseek/deepseek-v3.2",
    #     reasoning={"effort": "none"},
    # ),
    # "qwen3-235b-a22b-2507": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "qwen/qwen3-235b-a22b-2507",
    #     reasoning={"effort": "none"},
    # ),
    # "qwen3.5-flash-02-23": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "qwen/qwen3.5-flash-02-23",
    #     reasoning={"effort": "none"},
    # ),
    # "nemotron-3-super-120b-a12b": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "nvidia/nemotron-3-super-120b-a12b",
    #     reasoning={"effort": "none"},
    # ),
    # "phi-4": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "microsoft/phi-4",
    #     reasoning={"effort": "none"},
    #     provider={"ignore": ["nextbit"]},
    # ),
    # "hy3-preview": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "tencent/hy3-preview",
    #     reasoning={"effort": "none"},
    # ),
    # "mimo-v2.5": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "xiaomi/mimo-v2.5",
    #     reasoning={"effort": "none"},
    # ),
    # "glm-5.2": partial(
    #     _invoke_openrouter_model,
    #     _get_openrouter_client,
    #     "z-ai/glm-5.2",
    #     reasoning={"effort": "none"},
    # ),
}


# === PUBLIC FUNCTIONS ===


def _wait(retry_state):
    exception = retry_state.outcome.exception()
    logging.error(f"Encountered `{exception}`, retrying.")
    if isinstance(exception, RetryableNetworkError) and exception.seconds:
        return exception.seconds
    return 30


@retry(
    stop=stop_after_attempt(10),
    wait=_wait,
    retry=retry_if_exception_type((RetryableNetworkError, InvalidModelOutputError)),
    reraise=True,
)
def invoke_model(
    model: str, messages: Sequence[Message], use_cache: bool, **kwargs: Any
) -> ModelOutput:
    try:
        model_delegate = MODEL_DELEGATES[model]
    except KeyError as e:
        raise ValueError(f"Unsupported model: {model}.") from e

    logging.info(f"Invoking {model}.")

    response_format = deepcopy(kwargs.get("response_format"))

    if use_cache:
        delegate_kwargs = dict(model_delegate.keywords or {})

        # Match functools.partial behavior: delegate kwargs take precedence over caller.
        effective_kwargs = {
            **kwargs,
            **delegate_kwargs, # Ensures kwargs like `provider` cannot be overridden.
        }

        cache_key = _cache_key(
            model_name=model,
            messages=messages,
            kwargs=effective_kwargs,
        )

        cached = _cache_get_obj(cache_key)
        if cached is not None:
            try:
                _validate_structured_output(cached.text, response_format)
            except InvalidModelOutputError as e:
                raise InvalidModelOutputError(
                    "Tried returning invalid result from cache."
                ) from e
            return cached

    output = model_delegate(messages, **kwargs)

    # Validate against the original, stronger contract.
    _validate_structured_output(output.text, response_format)

    if use_cache:
        _cache_set_obj(cache_key, output)

    return output
