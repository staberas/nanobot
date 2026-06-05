from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults
from nanobot.cron.service import CronService
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.providers.factory import ProviderSnapshot
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
        "contextPipeline": {
            "maxPlannerTokens": 48,
            "maxRelevantResults": 2,
            "enableCron": True,
            "defaultReminderTime": "08:30",
            "timezone": "Europe/Athens",
            "enableCloudEscalation": True,
            "cloudEscalation": {
                "enabled": True,
                "providerPreset": "cloud-agent",
                "requireExplicitTrigger": True,
                "maxSummaryChars": 700,
                "maxReportChars": 1200,
                "appendFullReport": True,
                "returnCloudDirectly": False,
                "timeoutSeconds": 30,
            },
            "chatHistory": {
                "enabled": True,
                "maxTurns": 3,
                "maxChars": 900,
                "includeAssistant": False,
                "includeOnlyWhenAction": ["answer_directly"],
            },
        },
    })

    assert defaults.plain_chat_when_tools_unsupported is True
    assert defaults.plain_chat_system_prompt == "plain only"
    assert defaults.tool_execution_mode == "context_pipeline"
    assert defaults.tool_result_injection_max_chars == 1000
    assert defaults.context_pipeline.max_planner_tokens == 48
    assert defaults.context_pipeline.max_relevant_results == 2
    assert defaults.context_pipeline.enable_cron is True
    assert defaults.context_pipeline.default_reminder_time == "08:30"
    assert defaults.context_pipeline.timezone == "Europe/Athens"
    assert defaults.context_pipeline.enable_cloud_escalation is True
    assert defaults.context_pipeline.cloud_escalation.enabled is True
    assert defaults.context_pipeline.cloud_escalation.provider_preset == "cloud-agent"
    assert defaults.context_pipeline.cloud_escalation.require_explicit_trigger is True
    assert defaults.context_pipeline.cloud_escalation.max_summary_chars == 700
    assert defaults.context_pipeline.cloud_escalation.max_report_chars == 1200
    assert defaults.context_pipeline.cloud_escalation.timeout_seconds == 30
    assert defaults.context_pipeline.chat_history.enabled is True
    assert defaults.context_pipeline.chat_history.max_turns == 3
    assert defaults.context_pipeline.chat_history.max_chars == 900
    assert defaults.context_pipeline.chat_history.include_assistant is False
    assert defaults.context_pipeline.chat_history.include_only_when_action == ["answer_directly"]


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


def test_context_pipeline_planner_uses_history_for_reference_messages(tmp_path) -> None:
    async def run() -> None:
        reference_provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"ask_clarifying","query":"What are you referring to?","reason":"pronoun"}',
                '{"action":"ask_clarifying","query":"Which previous message?","reason":"history"}',
            ],
        )
        reference_loop = AgentLoop(
            bus=MessageBus(),
            provider=reference_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )

        that_plan = await reference_loop._context_pipeline_plan(
            "do you know what that means?",
            recent_chat_history_available=True,
        )
        previous_plan = await reference_loop._context_pipeline_plan(
            "what was my previous message?",
            recent_chat_history_available=True,
        )

        assert that_plan["action"] == "answer_directly"
        assert previous_plan["action"] == "answer_directly"
        planner_prompt = reference_provider.calls[0]["messages"][0]["content"]
        assert "If the user refers to previous messages" in planner_prompt
        assert "Recent chat history available: yes" in planner_prompt

        empty_provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"ask_clarifying","query":"What are you referring to?","reason":"pronoun"}'],
        )
        empty_loop = AgentLoop(
            bus=MessageBus(),
            provider=empty_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        empty_plan = await empty_loop._context_pipeline_plan(
            "do you know what that means?",
            recent_chat_history_available=False,
        )
        assert empty_plan["action"] == "ask_clarifying"
        empty_prompt = empty_provider.calls[0]["messages"][0]["content"]
        assert "Recent chat history available: no" in empty_prompt

        search_provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"ask_clarifying","query":"What should I search?","reason":"ambiguous"}'],
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
        search_plan = await search_loop._context_pipeline_plan(
            "search Jason Kolios Greece",
            recent_chat_history_available=True,
        )
        assert search_plan["action"] == "web_search"

        cron_provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"ask_clarifying","query":"When?","reason":"ambiguous"}'],
        )
        cron_loop = AgentLoop(
            bus=MessageBus(),
            provider=cron_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
        )
        cron_plan = await cron_loop._context_pipeline_plan(
            "remind me in 2 min to check the cluster",
            recent_chat_history_available=True,
        )
        assert cron_plan["action"] == "cron"
    asyncio.run(run())


def test_context_pipeline_cloud_escalation_planning_rules(tmp_path) -> None:
    async def run() -> None:
        cfg = AgentDefaults.model_validate({
            "contextPipeline": {
                "enableCloudEscalation": True,
                "cloudEscalation": {"enabled": True, "requireExplicitTrigger": False},
            }
        }).context_pipeline
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"answer_directly","query":"","reason":"casual"}',
                '{"action":"escalate_cloud_agent","query":"Jason Kolios Greece","reason":"too eager"}',
                '{"action":"escalate_cloud_agent","query":"deep research Jason Kolios Greece","reason":"research"}',
            ],
        )
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
            context_pipeline=cfg,
        )

        hello = await loop._context_pipeline_plan("hello")
        search = await loop._context_pipeline_plan("search Jason Kolios Greece")
        deep = await loop._context_pipeline_plan("deep research Jason Kolios Greece")

        assert hello["action"] == "answer_directly"
        assert search["action"] == "web_search"
        assert deep["action"] == "escalate_cloud_agent"

        explicit_cfg = AgentDefaults.model_validate({
            "contextPipeline": {
                "enableCloudEscalation": True,
                "cloudEscalation": {"enabled": True, "requireExplicitTrigger": True},
            }
        }).context_pipeline
        explicit_provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"escalate_cloud_agent","query":"architecture review","reason":"complex"}',
                '{"action":"escalate_cloud_agent","query":"use cloud architecture review","reason":"explicit"}',
            ],
        )
        explicit_loop = AgentLoop(
            bus=MessageBus(),
            provider=explicit_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
            context_pipeline=explicit_cfg,
        )

        blocked = await explicit_loop._context_pipeline_plan("architecture review this design")
        allowed = await explicit_loop._context_pipeline_plan("use cloud architecture review this design")

        assert blocked["action"] == "answer_directly"
        assert allowed["action"] == "escalate_cloud_agent"
    asyncio.run(run())


def test_context_pipeline_cloud_escalation_feeds_summary_to_local_and_appends_report(tmp_path) -> None:
    async def run() -> None:
        local_provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"escalate_cloud_agent","query":"deep research Jason Kolios Greece","reason":"explicit"}',
                "detailsFor Jason Kolios has Greece-related evidence in the cloud report.",
            ],
        )
        cloud_provider = PlainFakeProvider(
            supports_tools=True,
            responses=[
                "summary: Jason Kolios is mentioned in Greece-related results.\n"
                "key_points:\n- Greece evidence item\n"
                "citations/evidence:\n- Example — https://example.com\n"
                "full_report: Detailed cloud report about Jason Kolios Greece with citations.\n"
                "limitations: Some uncertainty remains."
            ],
        )
        cfg = AgentDefaults.model_validate({
            "contextPipeline": {
                "enableCloudEscalation": True,
                "cloudEscalation": {
                    "enabled": True,
                    "providerPreset": "cloud-agent",
                    "maxSummaryChars": 500,
                    "maxReportChars": 80,
                    "appendFullReport": True,
                },
            }
        }).context_pipeline
        loop = AgentLoop(
            bus=MessageBus(),
            provider=local_provider,
            workspace=tmp_path,
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
            context_pipeline=cfg,
            preset_snapshot_loader=lambda name: ProviderSnapshot(
                provider=cloud_provider,
                model="openai/gpt-5-mini",
                context_window_tokens=64000,
                signature=("cloud", name),
            ),
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="room",
                content="deep research Jason Kolios Greece",
            )
        )

        assert result is not None
        assert result.content.startswith("Short answer:\nFor Jason Kolios has Greece-related evidence")
        assert "detailsFor" not in result.content
        assert "Cloud report:\nDetailed cloud report about Jason Kolios Greece" in result.content
        report_section = result.content.split("Cloud report:\n", 1)[1].split("\n\nSources:", 1)[0]
        assert len(report_section) <= 81
        assert "Sources:\n1. Example — https://example.com" in result.content
        assert len(cloud_provider.calls) == 1
        assert cloud_provider.calls[0]["tools"] is None
        local_final_prompt = "\n".join(
            str(message.get("content", "")) for message in local_provider.calls[-1]["messages"]
        )
        assert "Cloud specialist summary:" in local_final_prompt
        assert "Jason Kolios is mentioned" in local_final_prompt
        assert "Key points/evidence:" in local_final_prompt
        assert "Detailed cloud report about Jason Kolios" not in local_final_prompt
        assert local_provider.calls[-1]["tools"] is None
    asyncio.run(run())


def test_context_pipeline_cloud_parser_sources_truncation_and_chunks(tmp_path) -> None:
    sample = (
        "Summary:\nFor RK3588, Ollama is mature.\n\n"
        "Key points:\n- Ollama uses llama.cpp.\n\n"
        "Full report:\n"
        "This is a detailed report with enough text to split across Matrix chunks. "
        "It preserves whitespace and should not merge headings into detailsFor artifacts.\n\n"
        "Sources:\n"
        "- Ollama docs — https://ollama.com/docs\n"
        "- RKLLAMA project — https://example.com/rkllama"
    )
    sections = AgentLoop._extract_cloud_sections(sample)

    assert sections["summary"] == "For RK3588, Ollama is mature."
    assert sections["full_report"].startswith("This is a detailed report")
    assert "detailsFor" not in sections["summary"]
    sources = AgentLoop._extract_cloud_sources(sections)
    assert "Ollama docs — https://ollama.com/docs" in sources
    assert "RKLLAMA project — https://example.com/rkllama" in sources

    truncated = AgentLoop._truncate_cloud_report("A" * 120, 40)
    assert truncated.endswith("[report truncated]")
    assert len(truncated) <= 40

    cfg = AgentDefaults.model_validate({
        "contextPipeline": {
            "enableCloudEscalation": True,
            "cloudEscalation": {
                "enabled": True,
                "splitLongReports": True,
                "matrixChunkChars": 500,
            },
        }
    }).context_pipeline
    loop = AgentLoop(
        bus=MessageBus(),
        provider=PlainFakeProvider(supports_tools=False),
        workspace=tmp_path,
        model="fake-small",
        context_window_tokens=2048,
        plain_chat_when_tools_unsupported=True,
        tool_execution_mode="context_pipeline",
        context_pipeline=cfg,
    )
    long_report = "Paragraph.\n\n" + ("Long cloud report text. " * 80)
    final = f"Short answer:\nLocal answer.\n\nCloud report:\n{long_report}\n\nSources:\n1. https://example.com"
    chunks = loop._cloud_escalation_outbound_chunks(final)

    assert len(chunks) > 1
    assert chunks[0] == "Short answer:\nLocal answer."
    assert any(chunk.startswith("Cloud report (part") for chunk in chunks[1:])
    assert chunks[-1].startswith("Sources:")
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_context_pipeline_cloud_missing_key_and_return_direct(tmp_path) -> None:
    async def run() -> None:
        cfg = AgentDefaults.model_validate({
            "contextPipeline": {
                "enableCloudEscalation": True,
                "cloudEscalation": {"enabled": True, "providerPreset": "cloud-agent"},
            }
        }).context_pipeline
        missing_provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"escalate_cloud_agent","query":"deep research x","reason":"explicit"}'],
        )
        missing_loop = AgentLoop(
            bus=MessageBus(),
            provider=missing_provider,
            workspace=tmp_path / "missing",
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
            context_pipeline=cfg,
            preset_snapshot_loader=lambda name: (_ for _ in ()).throw(
                ValueError("No API key configured for provider 'openrouter'.")
            ),
        )
        missing_result = await missing_loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="deep research x")
        )
        assert missing_result is not None
        assert missing_result.content == "Cloud escalation is configured, but the cloud provider API key is missing."

        direct_cfg = AgentDefaults.model_validate({
            "contextPipeline": {
                "enableCloudEscalation": True,
                "cloudEscalation": {
                    "enabled": True,
                    "returnCloudDirectly": True,
                    "appendFullReport": False,
                },
            }
        }).context_pipeline
        direct_local = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"escalate_cloud_agent","query":"deep research x","reason":"explicit"}'],
        )
        direct_cloud = PlainFakeProvider(supports_tools=True, responses=["summary: cloud answer\nfull_report: cloud report"])
        direct_loop = AgentLoop(
            bus=MessageBus(),
            provider=direct_local,
            workspace=tmp_path / "direct",
            model="fake-small",
            context_window_tokens=2048,
            plain_chat_when_tools_unsupported=True,
            tool_execution_mode="context_pipeline",
            context_pipeline=direct_cfg,
            preset_snapshot_loader=lambda name: ProviderSnapshot(
                provider=direct_cloud,
                model="openai/gpt-5-mini",
                context_window_tokens=64000,
                signature=("cloud", name),
            ),
        )
        direct_result = await direct_loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room2", content="deep research x")
        )
        assert direct_result is not None
        assert direct_result.content == "summary: cloud answer\nfull_report: cloud report"
        assert len(direct_local.calls) == 1
        assert len(direct_cloud.calls) == 1
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
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"chatHistory": {"enabled": False}}
            }).context_pipeline,
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


def test_context_pipeline_direct_chat_history_answers_previous_message(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"answer_directly","query":"","reason":"chat"}',
                'I\'m not sure what you mean by "my clanker."',
                '{"action":"answer_directly","query":"","reason":"chat"}',
                "okay",
                '{"action":"ask_clarifying","query":"Could you clarify?","reason":"history"}',
                'Your previous message was: "dont worry bout it"',
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
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {
                    "chatHistory": {
                        "enabled": True,
                        "maxTurns": 4,
                        "maxChars": 1500,
                        "includeAssistant": True,
                    }
                }
            }).context_pipeline,
        )
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="room",
                content="what's up my clanker?",
            )
        )
        await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="room",
                content="dont worry bout it",
            )
        )
        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="room",
                content="what was my previous message",
            )
        )

        assert result is not None
        assert result.content == 'Your previous message was: "dont worry bout it"'
        final_prompt = "\n".join(
            str(message.get("content", "")) for message in provider.calls[-1]["messages"]
        )
        assert "Recent conversation:" in final_prompt
        assert "User: what's up my clanker?" in final_prompt
        assert 'Assistant: I\'m not sure what you mean by "my clanker."' in final_prompt
        assert "User: dont worry bout it" in final_prompt
        assert "Current user:\nwhat was my previous message" in final_prompt
        assert provider.calls[-1]["tools"] is None
        assert "tool_choice" not in provider.calls[-1]
    asyncio.run(run())


def test_context_pipeline_direct_chat_history_respects_turn_and_char_limits(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"answer_directly","query":"","reason":"chat"}',
                "done",
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
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {
                    "chatHistory": {"enabled": True, "maxTurns": 1, "maxChars": 120}
                }
            }).context_pipeline,
        )
        session = loop.sessions.get_or_create("matrix:room")
        session.add_message("user", "very old message that should be dropped")
        session.add_message("assistant", "very old answer that should be dropped")
        session.add_message("user", "recent short message")
        session.add_message("assistant", "recent short answer")
        loop.sessions.save(session)

        result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="continue")
        )

        assert result is not None
        final_prompt = "\n".join(
            str(message.get("content", "")) for message in provider.calls[-1]["messages"]
        )
        assert "recent short message" in final_prompt
        assert "recent short answer" in final_prompt
        assert "very old message" not in final_prompt
        prompt_tokens, _ = estimate_prompt_tokens_chain(provider, "fake-small", provider.calls[-1]["messages"], None)
        assert prompt_tokens < 2048
    asyncio.run(run())


def test_context_pipeline_chat_history_strips_attachment_paths(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"answer_directly","query":"","reason":"chat"}',
                "ok",
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
        image_path = tmp_path / "private-image.png"
        image_path.write_text("fake", encoding="utf-8")
        session = loop.sessions.get_or_create("matrix:room")
        session.add_message("user", "see attached", media=[str(image_path)])
        session.add_message("assistant", "I can only process text in this mode.")
        loop.sessions.save(session)

        result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="what did I just say")
        )

        assert result is not None
        final_prompt = "\n".join(
            str(message.get("content", "")) for message in provider.calls[-1]["messages"]
        )
        assert "see attached" in final_prompt
        assert str(image_path) not in final_prompt
        assert "[image:" not in final_prompt
    asyncio.run(run())


def test_context_pipeline_new_clears_direct_chat_history(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=[
                '{"action":"answer_directly","query":"","reason":"chat"}',
                "fresh",
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
        session = loop.sessions.get_or_create("matrix:room")
        session.add_message("user", "stale message")
        session.add_message("assistant", "stale answer")
        loop.sessions.save(session)

        await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="/new")
        )
        result = await loop._process_message(
            InboundMessage(channel="matrix", sender_id="@u:s", chat_id="room", content="hello again")
        )

        assert result is not None
        final_prompt = "\n".join(
            str(message.get("content", "")) for message in provider.calls[-1]["messages"]
        )
        assert "stale message" not in final_prompt
        assert "stale answer" not in final_prompt
        assert "hello again" in final_prompt
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



def test_context_pipeline_cron_reminder_creates_job(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": True, "timezone": "Europe/Athens"}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="remind me in 2 min to check the cluster",
            )
        )

        assert result is not None
        assert "Reminder set for" in result.content
        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.payload.message == "check the cluster"
        assert job.payload.channel == "matrix"
        assert job.payload.to == "!room:s"
        assert job.payload.session_key == "matrix:!room:s"
        assert job.delete_after_run is True
        assert job.schedule.kind == "at"
        assert job.state.next_run_at_ms is not None
        assert job.payload.channel_meta["created_by"] == "@u:s"
        assert job.payload.channel_meta["reminder_text"] == "check the cluster"
    asyncio.run(run())


def test_context_pipeline_cron_denied_does_not_claim_success(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "deny": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": True}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="remind me tomorrow at 9 to check the cluster",
            )
        )

        assert result is not None
        assert result.content == "I cannot schedule reminders yet."
        assert cron_service.list_jobs(include_disabled=True) == []
    asyncio.run(run())


def test_context_pipeline_cron_ambiguous_time_asks_clarification(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": True, "timezone": "Europe/Athens"}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="remind me to check the cluster",
            )
        )

        assert result is not None
        assert result.content == "When should I remind you?"
        assert cron_service.list_jobs(include_disabled=True) == []
    asyncio.run(run())


def test_context_pipeline_cron_timezone_is_respected(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": True, "timezone": "Europe/Athens"}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="remind me every day at 9 to check the cluster in America/Denver",
            )
        )

        assert result is not None
        jobs = cron_service.list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert jobs[0].schedule.kind == "cron"
        assert jobs[0].schedule.expr == "0 9 * * *"
        assert jobs[0].schedule.tz == "America/Denver"
        assert jobs[0].payload.message == "check the cluster"
        assert jobs[0].payload.to == "!room:s"
    asyncio.run(run())


def test_context_pipeline_cron_tomorrow_at_nine_creates_one_shot(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": True, "timezone": "Europe/Athens"}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="remind me tomorrow at 9 to check the cluster",
            )
        )

        assert result is not None
        assert "Reminder set for" in result.content
        job = cron_service.list_jobs(include_disabled=True)[0]
        assert job.schedule.kind == "at"
        assert job.delete_after_run is True
        assert job.payload.message == "check the cluster"
    asyncio.run(run())


def test_context_pipeline_cron_enable_false_does_not_schedule(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": False}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="remind me in 2 min to check the cluster",
            )
        )

        assert result is not None
        assert result.content == "I cannot schedule reminders yet."
        assert cron_service.list_jobs(include_disabled=True) == []
    asyncio.run(run())


def test_context_pipeline_cron_every_morning_uses_default_time(tmp_path) -> None:
    async def run() -> None:
        cron_service = CronService(tmp_path / "cron" / "jobs.json")
        provider = PlainFakeProvider(
            supports_tools=False,
            responses=['{"action":"cron","query":"check the cluster","reason":"reminder"}'],
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
            cron_service=cron_service,
            timezone="UTC",
            tool_selection=AgentDefaults.model_validate({
                "toolSelection": {"enabled": True, "mode": "heuristic", "maxTools": 1, "allow": ["cron"]}
            }).tool_selection,
            context_pipeline=AgentDefaults.model_validate({
                "contextPipeline": {"enableCron": True, "defaultReminderTime": "09:00", "timezone": "Europe/Athens"}
            }).context_pipeline,
        )

        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="every morning remind me to check the cluster",
            )
        )

        assert result is not None
        job = cron_service.list_jobs(include_disabled=True)[0]
        assert job.schedule.kind == "cron"
        assert job.schedule.expr == "0 9 * * *"
        assert job.schedule.tz == "Europe/Athens"
        assert job.payload.message == "check the cluster"
    asyncio.run(run())


def test_plain_chat_text_only_guard_rejects_media_without_llm(tmp_path) -> None:
    async def run() -> None:
        provider = PlainFakeProvider(supports_tools=False, responses=["should not be used"])
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
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

        image_path = tmp_path / "image.png"
        image_path.write_bytes(b"not really an image")
        result = await loop._process_message(
            InboundMessage(
                channel="matrix",
                sender_id="@u:s",
                chat_id="!room:s",
                content="what is in this image?",
                media=[str(image_path)],
            )
        )

        assert result is not None
        assert "I can only process text in this mode" in result.content
        assert provider.calls == []
    asyncio.run(run())


def test_agent_loop_from_config_uses_top_level_memory_dream(tmp_path) -> None:
    from nanobot.config.schema import Config

    config = Config.model_validate({
        "memory": {
            "dream": {
                "enabled": True,
                "toolsRequired": False,
                "skipWhenToolsUnsupported": True,
                "plainChatFallback": False,
            }
        },
        "agents": {
            "defaults": {
                "workspace": str(tmp_path),
                "provider": "rkllama",
                "model": "Qwen3-4B-w8a8-npu",
            }
        },
        "providers": {
            "rkllama": {
                "apiKey": None,
                "apiBase": "http://192.168.100.23:30082/v1",
                "capabilities": {"tools": False, "preferMaxTokens": True},
            }
        },
    })

    loop = AgentLoop.from_config(config, bus=MessageBus(), provider=PlainFakeProvider(supports_tools=False))

    assert loop.dream.skip_when_tools_unsupported is True
    assert loop.dream.plain_chat_fallback is False
