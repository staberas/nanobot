from __future__ import annotations

from nanobot.config.schema import Config, ProviderCapabilitiesConfig
from nanobot.providers.factory import make_provider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


def _tool() -> dict:
    return {"type": "function", "function": {"name": "web_search", "description": "search"}}


def test_provider_capabilities_strip_unsupported_chat_fields() -> None:
    provider = OpenAICompatProvider(
        capabilities=ProviderCapabilitiesConfig(
            tools=False,
            toolChoice=False,
            parallelToolCalls=False,
            responseFormat=False,
        ),
        extra_body={"response_format": {"type": "json_object"}, "parallel_tool_calls": True},
    )
    kwargs = provider._build_kwargs(
        [{"role": "user", "content": "search"}],
        [_tool()],
        model="local-model",
        max_tokens=32,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice="auto",
    )
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "parallel_tool_calls" not in kwargs
    assert "response_format" not in kwargs
    assert "extra_body" not in kwargs or "response_format" not in kwargs["extra_body"]


def test_provider_capabilities_prefer_max_tokens_over_max_completion_tokens() -> None:
    provider = OpenAICompatProvider(
        spec=find_by_name("volcengine"),
        capabilities=ProviderCapabilitiesConfig(maxCompletionTokens=False, preferMaxTokens=True),
    )
    kwargs = provider._build_kwargs(
        [{"role": "user", "content": "hi"}],
        None,
        model="ep-test",
        max_tokens=32,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert kwargs["max_tokens"] == 32
    assert "max_completion_tokens" not in kwargs


def test_multiple_custom_providers_and_presets_resolve_independently() -> None:
    config = Config.model_validate({
        "providers": {
            "rkllama": {
                "apiKey": None,
                "apiBase": "http://192.168.100.23:30082/v1",
                "capabilities": {"tools": False, "preferMaxTokens": True},
            },
            "local-big": {
                "apiKey": None,
                "apiBase": "http://192.168.100.50:8000/v1",
                "capabilities": {"tools": True},
            },
            "openrouter": {
                "apiKey": "sk-or-test",
                "apiBase": "https://openrouter.ai/api/v1",
                "capabilities": {"tools": True, "maxCompletionTokens": True},
            },
        },
        "agents": {
            "defaults": {"provider": "rkllama", "model": "Qwen3-4B-w8a8-npu"},
        },
        "modelPresets": {
            "powerful": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"},
            "local-big-agent": {"provider": "local-big", "model": "qwen2.5-14b-instruct"},
        },
    })

    assert config.get_provider_name() == "rkllama"
    assert config.get_provider().api_base == "http://192.168.100.23:30082/v1"
    assert config.resolve_preset("powerful").provider == "openrouter"
    assert config.get_provider(preset=config.resolve_preset("local-big-agent")).api_base == "http://192.168.100.50:8000/v1"

    provider = make_provider(config)
    assert provider.get_default_model() == "Qwen3-4B-w8a8-npu"
    assert provider._capabilities.tools is False


def test_plain_chat_capability_request_uses_max_tokens_without_tool_fields() -> None:
    provider = OpenAICompatProvider(
        spec=find_by_name("volcengine"),
        capabilities=ProviderCapabilitiesConfig(
            tools=False,
            toolChoice=False,
            parallelToolCalls=False,
            responseFormat=False,
            maxCompletionTokens=False,
            preferMaxTokens=True,
        ),
        extra_body={"parallel_tool_calls": True, "response_format": {"type": "json_object"}},
    )

    kwargs = provider._build_kwargs(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "say exactly ok"},
        ],
        [_tool()],
        model="ep-test",
        max_tokens=32,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice="auto",
    )

    assert kwargs["max_tokens"] == 32
    assert "max_completion_tokens" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "parallel_tool_calls" not in kwargs
    assert "response_format" not in kwargs
    assert "extra_body" not in kwargs
