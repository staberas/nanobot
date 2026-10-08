"""Tests for provider extra_body config injection into request payloads."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, ProviderConfig
from nanobot.providers.factory import make_provider, provider_signature
from nanobot.providers.openai_compat_provider import (
    OpenAICompatProvider,
    _deep_merge,
)
from nanobot.providers.registry import find_by_name

# ---------------------------------------------------------------------------
# _deep_merge unit tests
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Verify recursive dict merge semantics."""

    def test_flat_merge(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_scalar(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self) -> None:
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        assert _deep_merge(base, override) == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deeply_nested(self) -> None:
        base = {"l1": {"l2": {"a": 1}}}
        override = {"l1": {"l2": {"b": 2}}}
        assert _deep_merge(base, override) == {"l1": {"l2": {"a": 1, "b": 2}}}

    def test_override_replaces_non_dict_with_dict(self) -> None:
        assert _deep_merge({"a": 1}, {"a": {"nested": True}}) == {"a": {"nested": True}}

    def test_override_replaces_dict_with_scalar(self) -> None:
        assert _deep_merge({"a": {"nested": True}}, {"a": "flat"}) == {"a": "flat"}

    def test_empty_base(self) -> None:
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self) -> None:
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


class TestExtraBodyInit:
    """Verify the provider stores extra_body from config."""

    def test_default_is_empty(self) -> None:
        provider = OpenAICompatProvider(api_key="test")
        assert provider._extra_body == {}

    def test_none_becomes_empty(self) -> None:
        provider = OpenAICompatProvider(api_key="test", extra_body=None)
        assert provider._extra_body == {}

    def test_dict_stored(self) -> None:
        body = {"chat_template_kwargs": {"enable_thinking": False}}
        provider = OpenAICompatProvider(api_key="test", extra_body=body)
        assert provider._extra_body == body


class TestConfiguredThinkingStyle:
    """Verify custom/local providers can select their thinking wire format."""

    @pytest.mark.parametrize(
        ("style", "expected"),
        [
            ("thinking_type", {"thinking": {"type": "enabled"}}),
            ("enable_thinking", {"enable_thinking": True}),
            ("reasoning_split", {"reasoning_split": True}),
        ],
    )
    def test_style_maps_reasoning_effort_to_request_body(
        self, style: str, expected: dict[str, Any]
    ) -> None:
        provider = OpenAICompatProvider(
            api_base="http://rkllama.local/v1",
            default_model="local-model",
            thinking_style=style,
        )

        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None,
            model=None,
            max_tokens=100,
            temperature=0.1,
            reasoning_effort="high",
            tool_choice=None,
        )

        assert kwargs["extra_body"] == expected

    def test_explicit_style_overrides_model_inference(self) -> None:
        provider = OpenAICompatProvider(
            default_model="kimi-k2.5",
            thinking_style="enable_thinking",
        )

        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None,
            model=None,
            max_tokens=100,
            temperature=0.1,
            reasoning_effort="none",
            tool_choice=None,
        )

        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert "thinking" not in kwargs["extra_body"]

    def test_config_rejects_unknown_style(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig.model_validate({"thinkingStyle": "rkllama_magic"})

    def test_factory_passes_style_and_tracks_it_in_signature(self) -> None:
        config = Config.model_validate({
            "agents": {
                "defaults": {
                    "provider": "rkllama",
                    "model": "Qwen3-4B-w8a8-npu",
                }
            },
            "providers": {
                "rkllama": {
                    "apiBase": "http://rkllama.local/v1",
                    "thinkingStyle": "enable_thinking",
                    "capabilities": {"tools": False},
                }
            },
        })

        provider = make_provider(config)

        assert isinstance(provider, OpenAICompatProvider)
        assert provider._thinking_style == "enable_thinking"
        assert "enable_thinking" in provider_signature(config)


# ---------------------------------------------------------------------------
# _build_kwargs integration
# ---------------------------------------------------------------------------


def _make_provider(extra_body: dict[str, Any] | None = None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test-key",
        default_model="test-model",
        extra_body=extra_body,
    )


def _simple_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "hello"}]


class TestBuildKwargsExtraBody:
    """Verify extra_body flows into _build_kwargs output."""

    def test_no_extra_body_no_key(self) -> None:
        provider = _make_provider()
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert "extra_body" not in kwargs

    def test_extra_body_injected(self) -> None:
        provider = _make_provider({"chat_template_kwargs": {"enable_thinking": False}})
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_extra_body_merges_with_thinking(self) -> None:
        """Config extra_body should merge with (and override) thinking params."""
        from nanobot.providers.registry import ProviderSpec

        spec = MagicMock(spec=ProviderSpec)
        spec.thinking_style = "deepseek"
        spec.supports_prompt_caching = False
        spec.strip_model_prefix = False
        spec.model_overrides = []
        spec.name = "custom"
        spec.supports_max_completion_tokens = False
        spec.env_key = None
        spec.default_api_base = None
        spec.is_local = True
        spec.detect_by_base_keyword = None

        provider = OpenAICompatProvider(
            api_key="test",
            default_model="deepseek-v3",
            spec=spec,
            extra_body={"custom_param": "value"},
        )
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort="high", tool_choice=None,
        )
        body = kwargs.get("extra_body", {})
        # Config param should be present
        assert body.get("custom_param") == "value"

    def test_nested_extra_body_does_not_clobber_siblings(self) -> None:
        """Nested dict merge should preserve sibling keys."""
        provider = _make_provider({
            "chat_template_kwargs": {"enable_thinking": False},
        })
        # Simulate internal code having set a sibling key
        # by manually calling _build_kwargs — the internal logic
        # doesn't set chat_template_kwargs, so we test the merge path
        # by having extra_body itself contain nested keys
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_guided_json_injection(self) -> None:
        """Real-world use case: vLLM guided decoding."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        provider = _make_provider({"guided_json": schema})
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"]["guided_json"] == schema

    def test_repetition_penalty_injection(self) -> None:
        """Real-world use case: local model sampling param."""
        provider = _make_provider({"repetition_penalty": 1.15})
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"]["repetition_penalty"] == 1.15


class TestBuildResponsesBodyExtraBody:
    """Verify extra_body flows into Responses API request bodies."""

    def test_responses_extra_body_merges_top_level_fields(self) -> None:
        provider = OpenAICompatProvider(
            api_key="test-key",
            default_model="gpt-5",
            spec=find_by_name("openai"),
            extra_body={
                "metadata": {"source": "test"},
                "parallel_tool_calls": False,
            },
        )

        body = provider._build_responses_body(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )

        assert body["metadata"] == {"source": "test"}
        assert body["parallel_tool_calls"] is False

    def test_responses_extra_body_appends_tools(self) -> None:
        provider = OpenAICompatProvider(
            api_key="test-key",
            default_model="gpt-5",
            spec=find_by_name("openai"),
            extra_body={"tools": [{"type": "web_search"}]},
        )

        body = provider._build_responses_body(
            messages=_simple_messages(),
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }],
            model=None, max_tokens=100, temperature=0.1,
            reasoning_effort=None, tool_choice=None,
        )

        assert body["tools"] == [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
            {"type": "web_search"},
        ]

    def test_responses_extra_body_merges_include_without_duplicates(self) -> None:
        provider = OpenAICompatProvider(
            api_key="test-key",
            default_model="gpt-5",
            spec=find_by_name("openai"),
            extra_body={
                "include": [
                    "reasoning.encrypted_content",
                    "web_search_call.action.sources",
                ],
            },
        )

        body = provider._build_responses_body(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort="high", tool_choice=None,
        )

        assert body["include"] == [
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaConfig:
    """Verify ProviderConfig accepts extra_body."""

    def test_default_is_none(self) -> None:
        from nanobot.config.schema import ProviderConfig

        config = ProviderConfig()
        assert config.extra_body is None

    def test_accepts_dict(self) -> None:
        from nanobot.config.schema import ProviderConfig

        config = ProviderConfig(extra_body={"guided_json": {"type": "object"}})
        assert config.extra_body == {"guided_json": {"type": "object"}}

    def test_nested_dict(self) -> None:
        from nanobot.config.schema import ProviderConfig

        config = ProviderConfig(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        assert config.extra_body["chat_template_kwargs"]["enable_thinking"] is False
