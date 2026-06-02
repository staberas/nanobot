from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tool_selection import (
    ToolSelectionPolicy,
    heuristic_tool_names,
    select_tools_for_request,
)
from nanobot.config.schema import Config, ProviderCapabilitiesConfig, ToolSelectionConfig
from nanobot.providers.base import LLMResponse
from nanobot.providers.factory import make_provider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Use {name} when relevant.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


ALL_TOOL_NAMES = [
    "apply_patch",
    "run_cli_app",
    "complete_goal",
    "cron",
    "edit_file",
    "find_files",
    "grep",
    "list_dir",
    "long_task",
    "message",
    "read_file",
    "spawn",
    "web_fetch",
    "web_search",
    "write_file",
    "my",
]


class _Provider:
    async def chat_with_retry(self, **kwargs):
        self.kwargs = kwargs
        return LLMResponse(content="ok")

    def estimate_prompt_tokens(self, messages, tools=None, model=None):
        return 100 + 10 * len(tools or []), "fake"


def test_default_tool_selection_preserves_all_tools() -> None:
    provider = _Provider()
    runner = AgentRunner(provider)  # type: ignore[arg-type]
    tools = MagicMock()
    tools.get_definitions.return_value = [_schema(name) for name in ALL_TOOL_NAMES]
    asyncio.run(runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        workspace=Path("."),
    )))
    sent = provider.kwargs["tools"]
    assert [tool["function"]["name"] for tool in sent] == ALL_TOOL_NAMES


def test_tool_selection_allow_deny_filters_dynamic_tools() -> None:
    result = select_tools_for_request(
        all_tools=[_schema(name) for name in ALL_TOOL_NAMES],
        policy=ToolSelectionPolicy(
            enabled=True,
            max_tools=2,
            allow=["web_search", "web_fetch", "cron", "spawn", "message", "complete_goal"],
            deny=["web_fetch", "message"],
        ),
        messages=[{"role": "user", "content": "search the web and fetch the page"}],
        provider=_Provider(),
        model="x",
    )
    assert result.selected_names == ["web_search"]


def test_deny_wins_over_allow_and_always_include() -> None:
    result = select_tools_for_request(
        all_tools=[_schema(name) for name in ALL_TOOL_NAMES],
        policy=ToolSelectionPolicy(
            enabled=True,
            max_tools=2,
            always_include=["message", "complete_goal"],
            allow=["message", "complete_goal", "web_search"],
            deny=["message", "web_search"],
        ),
        messages=[{"role": "user", "content": "latest news"}],
        provider=_Provider(),
        model="x",
    )
    assert result.selected_names == ["complete_goal"]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("hi, say exactly ok", []),
        ("search the web for today's date", ["web_search"]),
        ("open and read https://example.com/page", ["web_search", "web_fetch"]),
        ("remind me every day at 9", ["cron"]),
        ("delegate this to a subagent in the background", ["spawn"]),
    ],
)
def test_heuristic_tool_names(prompt: str, expected: list[str]) -> None:
    assert heuristic_tool_names(prompt)[: len(expected)] == expected


def test_config_accepts_multiple_custom_providers_and_agent_presets() -> None:
    cfg = Config(
        agents={
            "defaults": {"provider": "rkllama", "model": "small", "maxTokens": 32},
            "powerful": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"},
            "local-big-agent": {"provider": "local-big", "model": "qwen2.5-14b-instruct"},
        },
        providers={
            "rkllama": {
                "apiKey": None,
                "apiBase": "http://192.168.100.23:30082/v1",
                "capabilities": {"tools": False, "preferMaxTokens": True},
            },
            "local-big": {"apiKey": None, "apiBase": "http://192.168.100.50:8000/v1"},
            "openrouter": {"apiKey": "sk-or-test", "apiBase": "https://openrouter.ai/api/v1"},
        },
    )
    assert cfg.get_provider_name(preset=cfg.resolve_preset()) == "rkllama"
    assert cfg.get_api_base(preset=cfg.resolve_preset()) == "http://192.168.100.23:30082/v1"
    assert cfg.resolve_preset("powerful").provider == "openrouter"
    assert cfg.resolve_preset("local-big-agent").provider == "local-big"
    provider = make_provider(cfg)
    assert provider.get_default_model() == "small"


def test_openai_compat_capabilities_strip_tool_fields_and_prefer_max_tokens() -> None:
    provider = OpenAICompatProvider(
        api_base="http://local/v1",
        default_model="local-small",
        capabilities=ProviderCapabilitiesConfig(
            tools=False,
            tool_choice=False,
            parallel_tool_calls=False,
            response_format=False,
            max_completion_tokens=False,
            prefer_max_tokens=True,
        ),
    )
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "search web"}],
        tools=[_schema("web_search")],
        model=None,
        max_tokens=32,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice="auto",
    )
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "parallel_tool_calls" not in kwargs
    assert "response_format" not in kwargs
    assert kwargs["max_tokens"] == 32
    assert "max_completion_tokens" not in kwargs


def test_openai_compat_capabilities_can_disable_tool_choice_only() -> None:
    provider = OpenAICompatProvider(
        api_base="http://local/v1",
        default_model="local-big",
        capabilities=ProviderCapabilitiesConfig(tools=True, tool_choice=False),
    )
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "search web"}],
        tools=[_schema("web_search")],
        model=None,
        max_tokens=32,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice="auto",
    )
    assert kwargs["tools"]
    assert "tool_choice" not in kwargs


def test_openai_compat_default_max_completion_behavior_preserved_for_registry_specs() -> None:
    provider = OpenAICompatProvider(
        api_key="test",
        api_base="https://api.openai.com/v1",
        default_model="gpt-5.1",
        spec=find_by_name("openai"),
    )
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=123,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in kwargs


def test_provider_without_tool_support_receives_plain_chat() -> None:
    provider = _Provider()
    provider.supports_configured_tool_calls = False
    runner = AgentRunner(provider)  # type: ignore[arg-type]
    tools = MagicMock()
    tools.get_definitions.return_value = [_schema("web_search")]
    asyncio.run(runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "search the web"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        workspace=Path("."),
        tool_selection=ToolSelectionConfig(enabled=True, allow=["web_search"], maxTools=2),
    )))
    assert provider.kwargs["tools"] is None


def test_tools_web_enable_false_removes_web_tools(tmp_path: Path) -> None:
    from nanobot.agent.tools.context import ToolContext
    from nanobot.agent.tools.loader import ToolLoader
    from nanobot.agent.tools.registry import ToolRegistry

    cfg = Config(tools={"web": {"enable": False}})
    registry = ToolRegistry()
    ToolLoader().load(ToolContext(config=cfg.tools, workspace=str(tmp_path)), registry)
    assert "web_search" not in registry.tool_names
    assert "web_fetch" not in registry.tool_names


def test_tools_exec_enable_false_removes_exec_tool(tmp_path: Path) -> None:
    from nanobot.agent.tools.context import ToolContext
    from nanobot.agent.tools.loader import ToolLoader
    from nanobot.agent.tools.registry import ToolRegistry

    cfg = Config(tools={"exec": {"enable": False}})
    registry = ToolRegistry()
    ToolLoader().load(ToolContext(config=cfg.tools, workspace=str(tmp_path)), registry)
    assert "run_cli_app" in registry.tool_names
    assert "exec" not in registry.tool_names


def test_openai_compat_capability_overrides_registry_max_completion_tokens() -> None:
    provider = OpenAICompatProvider(
        api_key="test",
        api_base="https://api.openai.com/v1",
        default_model="gpt-5.1",
        spec=find_by_name("openai"),
        capabilities=ProviderCapabilitiesConfig(maxCompletionTokens=False, preferMaxTokens=True),
    )
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=77,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert kwargs["max_tokens"] == 77
    assert "max_completion_tokens" not in kwargs
