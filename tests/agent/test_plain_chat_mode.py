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

    def __init__(
        self,
        *,
        supports_tools: bool = False,
        responses: list[str | LLMResponse] | None = None,
    ) -> None:
        super().__init__()
        self.generation = GenerationSettings(max_tokens=32, temperature=0.2)
        self._supports_tools = supports_tools
        self.calls: list[dict[str, Any]] = []
        self.responses = list(responses or [])

    def get_default_model(self) -> str:
        return "fake-small"

    def supports_tools(self) -> bool:
        return self._supports_tools

    async def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        return await self.chat_with_retry(**kwargs)

    async def chat_with_retry(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        self.calls.append(kwargs)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, LLMResponse):
                return response
            return LLMResponse(content=response, usage={"prompt_tokens": 12, "completion_tokens": 1})
        return LLMResponse(content="ok", usage={"prompt_tokens": 12, "completion_tokens": 1})


def test_openai_compat_capabilities_tools_false_disables_supports_tools() -> None:
    provider = OpenAICompatProvider(capabilities={"tools": False})

    assert provider.supports_tools() is False
    assert provider.supports_configured_tool_calls is False


def test_plain_chat_config_accepts_camel_case_aliases() -> None:
    defaults = AgentDefaults.model_validate({
        "plainChatWhenToolsUnsupported": True,
        "plainChatSystemPrompt": "plain only",
        "toolExecutionMode": "context_pipeline",
        "toolResultInjectionMaxChars": 1000,
        "contextPipeline": {"maxPlannerTokens": 48, "maxRelevantResults": 2},
    })

    assert defaults.plain_chat_when_tools_unsupported is True
    assert defaults.plain_chat_system_prompt == "plain only"
    assert defaults.tool_execution_mode == "context_pipeline"
    assert defaults.tool_result_injection_max_chars == 1000
    assert defaults.context_pipeline.max_planner_tokens == 48
    assert defaults.context_pipeline.max_relevant_results == 2


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


def test_context_pipeline_planner_actions_and_fallback(tmp_path) -> None:
    async def run() -> None:
        search_provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"web_search","query":"Jason Kolios Greece","reason":"external info"}'],
        )
        search_loop = AgentLoop(
            bus=MessageBus(),
            provider=search_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        plan = await search_loop._context_pipeline_plan("search Jason Kolios Greece")
        assert plan["action"] == "web_search"
        assert plan["query"] == "Jason Kolios Greece"
        assert search_provider.calls[0]["max_tokens"] == 64
        planner_prompt = search_provider.calls[0]["messages"][0]["content"]
        assert "old prompt" not in planner_prompt
        assert "old answer" not in planner_prompt
        assert "function" not in planner_prompt.lower()

        direct_provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"answer_directly","query":"","reason":"greeting"}'],
        )
        direct_loop = AgentLoop(
            bus=MessageBus(),
            provider=direct_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        direct_plan = await direct_loop._context_pipeline_plan("llo")
        assert direct_plan["action"] == "answer_directly"

        fallback_provider = PlainFakeProvider(supports_tools=False, responses=["not json"])
        fallback_loop = AgentLoop(
            bus=MessageBus(),
            provider=fallback_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        fallback_plan = await fallback_loop._context_pipeline_plan("search Jason Kolios Greece")
        assert fallback_plan["action"] == "web_search"
        assert fallback_plan["query"] == "Jason Kolios Greece"
    asyncio.run(run())


def test_context_pipeline_direct_answer_uses_tiny_no_history_prompt(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"answer_directly","query":"","reason":"greeting"}',
                "hello",
            ],
        )
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            max_iterations=0,
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        loop.context.build_messages = AsyncMock(side_effect=AssertionError("tool prompt built"))
        loop.tools.get_definitions = MagicMock(side_effect=AssertionError("tools requested"))
        session = loop.sessions.get_or_create("matrix:room")
        session.add_message("user", "old prompt should not be included")
        loop.sessions.save(session)

        result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="llo")
        )

        assert result is not None
        assert result.content == "hello"
        assert len(provider.calls) == 2
        final_prompt = "\n".join(str(message.get("content", "")) for message in provider.calls[-1]["messages"])
        assert "old prompt should not be included" not in final_prompt
        assert provider.calls[-1]["tools"] is None
        assert "tool_choice" not in provider.calls[-1]
    asyncio.run(run())


def test_context_pipeline_web_search_reducer_and_final_prompt(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"web_search","query":"Jason Kolios Greece","reason":"external info"}',
                '{"relevant":true,"score":3,"summary":"Jason Kolios is associated with Greece.","url":"https://example.com/jason"}',
                '{"relevant":false,"score":0,"summary":"irrelevant","url":"https://example.com/other"}',
                "Jason Kolios appears connected to Greece based on the search evidence.",
            ],
        )
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            max_iterations=0,
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["web_search"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {
                    "maxSearchResults": 2,
                    "maxRelevantResults": 1,
                    "maxFinalEvidenceChars": 500,
                }
            }).context_pipeline,
        )
        loop.context.build_messages = AsyncMock(side_effect=AssertionError("tool prompt built"))
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
        loop.tools.execute = AsyncMock(return_value=(
            "Results for: Jason Kolios Greece\n\n"
            "1. Jason Kolios Greece profile\n"
            "   https://example.com/jason\n"
            "   Jason Kolios Greece snippet\n"
            "2. Unrelated result\n"
            "   https://example.com/other\n"
            "   Something else entirely\n"
        ))

        session = loop.sessions.get_or_create("matrix:room")
        session.add_message("user", "old prompt that must not enter planner or reducer")
        session.add_message("assistant", "old answer that must not enter planner or reducer")
        loop.sessions.save(session)

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="room",
                content="search Jason Kolios Greece",
            )
        )

        assert result is not None
        assert "Jason Kolios" in result.content
        loop.tools.execute.assert_awaited_once_with(
            "web_search", {"query": "Jason Kolios Greece", "count": 2}
        )
        assert len(provider.calls) == 4
        planner_prompt = provider.calls[0]["messages"][0]["content"]
        reducer_prompts = [provider.calls[1]["messages"][0]["content"], provider.calls[2]["messages"][0]["content"]]
        assert "old prompt" not in planner_prompt
        assert all("old prompt" not in prompt for prompt in reducer_prompts)
        final_call = provider.calls[-1]
        assert final_call["tools"] is None
        assert "tool_choice" not in final_call
        assert "response_format" not in final_call
        final_prompt = "\n".join(str(message.get("content", "")) for message in final_call["messages"])
        assert "Original user request:" in final_prompt
        assert "search Jason Kolios Greece" in final_prompt
        assert "Jason Kolios is associated with Greece." in final_prompt
        assert "Unrelated result" not in final_prompt
        assert "old prompt that must not enter planner or reducer" not in final_prompt
        prompt_tokens, _ = estimate_prompt_tokens_chain(provider, "fake-small", final_call["messages"], None)
        assert prompt_tokens < 2048
        assert loop.context_pipeline is not None
    asyncio.run(run())


def test_context_pipeline_reducer_json_fallback_keeps_query_matches(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(supports_tools=False, responses=["not json"])
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        kept = await loop._context_pipeline_reduce_result(
            "search Jason Kolios Greece",
            "Jason Kolios Greece",
            {
                "title": "Jason Kolios Greece profile",
                "url": "https://example.com/jason",
                "snippet": "Jason Kolios appears in Greek search result snippets.",
            },
        )
        dropped = await loop._context_pipeline_reduce_result(
            "search Jason Kolios Greece",
            "Jason Kolios Greece",
            {
                "title": "Completely unrelated",
                "url": "https://example.com/other",
                "snippet": "No overlapping terms here.",
            },
        )
        assert kept is not None
        assert kept["score"] >= 1
        assert dropped is None
    asyncio.run(run())
