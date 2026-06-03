from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.utils.helpers import estimate_prompt_tokens_chain


class PlainFakeProvider(LLMProvider):
    supports_progress_deltas = False

    def __init__(self, *, supports_tools: bool = False) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=32, temperature=0.2)
        self._supports_tools = supports_tools
        self.calls: list[dict[str, Any]] = []

    def get_default_model(self) -> str:
        return "fake-small"

    def supports_tools(self) -> bool:
        return self._supports_tools

    async def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        return await self.chat_with_retry(**kwargs)

    async def chat_with_retry(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        return LLMResponse(content="ok", usage={"prompt_tokens": 12, "completion_tokens": 1})


def test_openai_compat_capabilities_tools_false_disables_supports_tools() -> None:
    provider = OpenAICompatProvider(capabilities={"tools": False})

    assert provider.supports_tools() is False
    assert provider.supports_configured_tool_calls is False


def test_plain_chat_config_accepts_camel_case_aliases() -> None:
    defaults = AgentDefaults.model_validate({
        "plainChatWhenToolsUnsupported": True,
        "plainChatSystemPrompt": "plain only",
        "toolExecutionMode": "prompt_injection",
        "toolResultInjectionMaxChars": 1000,
    })

    assert defaults.plain_chat_when_tools_unsupported is True
    assert defaults.plain_chat_system_prompt == "plain only"
    assert defaults.tool_execution_mode == "prompt_injection"
    assert defaults.tool_result_injection_max_chars == 1000


def test_plain_chat_bypasses_tool_loop_when_provider_lacks_tools(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(supports_tools=False)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            max_iterations=0,
            context_window_tokens=1024,
            plain_chat_when_tools_unsupported=True,
            plain_chat_system_prompt="You are a concise assistant. Reply in plain text only.",
        )
        loop.context.build_messages = AsyncMock(side_effect=AssertionError("tool prompt built"))
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
        loop.tools.get_definitions = MagicMock(side_effect=AssertionError("tools requested"))

        result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="llo")
        )

        assert result is not None
        assert result.content == "ok"
        assert "maximum number of tool call iterations" not in result.content
        assert len(provider.calls) == 1
        call = provider.calls[0]
        assert call["tools"] is None
        assert call["model"] == "fake-small"
        assert "tool_choice" not in call
        assert "parallel_tool_calls" not in call
        assert "response_format" not in call
        assert call["messages"] == [
            {"role": "system", "content": "You are a concise assistant. Reply in plain text only."},
            {"role": "user", "content": "llo"},
        ]
        prompt_tokens, _ = estimate_prompt_tokens_chain(provider, "fake-small", call["messages"], None)
        assert prompt_tokens < 1024
        prompt_text = "\n".join(str(message.get("content", "")) for message in call["messages"])
        assert "tool" not in prompt_text.lower()
        assert "function" not in prompt_text.lower()
    asyncio.run(run())

def test_prompt_injection_web_search_runs_internally_without_tool_schema(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(supports_tools=False)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            max_iterations=0,
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="prompt_injection",
            tool_result_injection_max_chars=1200,
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {
                    "enabled": True,
                    "mode": "heuristic",
                    "maxTools": 1,
                    "allow": ["web_search"],
                }
            }).tool_selection,
        )
        loop.context.build_messages = AsyncMock(side_effect=AssertionError("tool prompt built"))
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
        loop.tools.get_definitions = MagicMock(return_value=[{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }])
        loop.tools.execute = AsyncMock(
            return_value='1. Jason Kolios — example snippet — https://example.com/jason'
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="room",
                content="can you search jason kolios?",
            )
        )

        assert result is not None
        assert result.content == "ok"
        loop.tools.execute.assert_awaited_once_with("web_search", {"query": "jason kolios", "count": 3})
        assert len(provider.calls) == 1
        call = provider.calls[0]
        assert call["tools"] is None
        assert "tool_choice" not in call
        assert "response_format" not in call
        assert all("_prompt_injection" not in message for message in call["messages"])
        prompt_text = "\n".join(str(message.get("content", "")) for message in call["messages"])
        assert 'Search results for "jason kolios"' in prompt_text
        assert "Jason Kolios" in prompt_text
        assert "Answer the user using only these search results" in prompt_text
        prompt_tokens, _ = estimate_prompt_tokens_chain(provider, "fake-small", call["messages"], None)
        assert prompt_tokens < 2048
    asyncio.run(run())

def test_plain_chat_new_clears_stale_history(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(supports_tools=False)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            max_iterations=0,
            context_window_tokens=1024,
            plain_chat_when_tools_unsupported=True,
        )
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

        stale = loop.sessions.get_or_create("matrix:room")
        stale.add_message("user", "old prompt that must be cleared")
        stale.add_message("assistant", "old answer that must be cleared")
        loop.sessions.save(stale)

        new_result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="/new")
        )
        assert new_result is not None

        result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="llo")
        )

        assert result is not None
        assert result.content == "ok"
        assert len(provider.calls) == 1
        contents = [message["content"] for message in provider.calls[0]["messages"]]
        assert "old prompt that must be cleared" not in contents
        assert "old answer that must be cleared" not in contents
        assert contents[-1] == "llo"
    asyncio.run(run())

def test_tool_capable_provider_still_uses_agent_runner(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(supports_tools=True)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-tool-model",
            max_iterations=1,
            context_window_tokens=1024,
            plain_chat_when_tools_unsupported=True,
        )
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

        result = await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="test", content="Hi")
        )

        assert result is not None
        assert result.content == "ok"
        assert len(provider.calls) == 1
        assert len(provider.calls[0]["messages"]) > 2
    asyncio.run(run())
