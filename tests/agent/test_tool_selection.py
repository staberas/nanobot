from __future__ import annotations

import asyncio

from nanobot.agent.tool_selection import ToolSelectionConfig, select_tool_definitions
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool"}}


def _select(prompt: str, cfg: ToolSelectionConfig, names: list[str] | None = None) -> list[str]:
    definitions = [_schema(name) for name in (names or [
        "web_search", "web_fetch", "cron", "spawn", "long_task",
        "message", "complete_goal", "read_file", "grep", "find_files", "list_dir",
        "edit_file", "write_file", "apply_patch",
    ])]
    selected = select_tool_definitions(definitions, [{"role": "user", "content": prompt}], cfg)
    return [ToolRegistry._schema_name(schema) for schema in selected]


def test_disabled_tool_selection_preserves_current_behavior() -> None:
    definitions = [_schema("web_search"), _schema("cron")]
    selected = select_tool_definitions(
        definitions,
        [{"role": "user", "content": "hi"}],
        ToolSelectionConfig(enabled=False),
    )
    assert selected == definitions


def test_allow_and_deny_filter_selected_tools() -> None:
    cfg = ToolSelectionConfig(
        enabled=True,
        max_tools=3,
        allow=("web_search", "web_fetch", "cron"),
        deny=("web_fetch",),
    )
    assert _select("search the web and fetch the page", cfg) == ["web_search"]


def test_deny_wins_over_allow_and_always_include() -> None:
    cfg = ToolSelectionConfig(
        enabled=True,
        always_include=("message", "complete_goal"),
        allow=("message", "complete_goal", "web_search"),
        deny=("message", "web_search"),
    )
    assert _select("search latest news", cfg) == ["complete_goal"]


def test_ordinary_chat_selects_no_external_tools() -> None:
    cfg = ToolSelectionConfig(enabled=True, max_tools=2)
    assert _select("hi, say exactly ok", cfg) == []


def test_web_query_selects_search_and_fetch_only_when_implied() -> None:
    cfg = ToolSelectionConfig(enabled=True, max_tools=2)
    assert _select("search the web for today's date", cfg) == ["web_search"]
    assert _select("search the web and open the first result", cfg) == ["web_search", "web_fetch"]


def test_cron_and_spawn_prompts_select_allowed_tools() -> None:
    cfg = ToolSelectionConfig(enabled=True, max_tools=2, allow=("cron", "spawn"))
    assert _select("remind me tomorrow morning", cfg) == ["cron"]
    assert _select("delegate this to a subagent in the background", cfg) == ["spawn"]


def test_file_tools_are_conservative_and_respect_allow() -> None:
    cfg = ToolSelectionConfig(enabled=True, max_tools=3, allow=("read_file", "grep"))
    assert _select("read the config file in the workspace", cfg) == ["read_file", "grep"]
    assert _select("edit the config file", cfg) == []


def test_tools_web_disabled_still_removes_web_tools_from_configured_loop(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMProvider

    class DummyProvider(LLMProvider):
        async def chat(self, *args, **kwargs):
            raise NotImplementedError

        def get_default_model(self) -> str:
            return "test-model"

    provider = DummyProvider()
    config = Config.model_validate({"tools": {"web": {"enable": False}}})
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        tools_config=config.tools,
    )
    assert "web_search" not in loop.tool_names
    assert "web_fetch" not in loop.tool_names


def test_tools_exec_disabled_still_removes_exec_tools_from_configured_loop(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMProvider

    class DummyProvider(LLMProvider):
        async def chat(self, *args, **kwargs):
            raise NotImplementedError

        def get_default_model(self) -> str:
            return "test-model"

    provider = DummyProvider()
    config = Config.model_validate({"tools": {"exec": {"enable": False}}})
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        tools_config=config.tools,
    )
    assert "exec" not in loop.tool_names
    assert "write_stdin" not in loop.tool_names

def test_new_command_clears_current_session_history(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.command.builtin import cmd_new
    from nanobot.command.router import CommandContext
    from nanobot.providers.base import LLMProvider

    class DummyProvider(LLMProvider):
        async def chat(self, *args, **kwargs):
            raise NotImplementedError

        def get_default_model(self) -> str:
            return "test-model"

    loop = AgentLoop(bus=MessageBus(), provider=DummyProvider(), workspace=tmp_path)
    key = "matrix:!room:example.org"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old message")
    session.add_message("assistant", "old response")
    loop.sessions.save(session)

    msg = InboundMessage(
        channel="matrix",
        chat_id="!room:example.org",
        sender_id="@user:example.org",
        content="/new",
    )
    outbound = asyncio.run(cmd_new(CommandContext(msg=msg, session=session, key=key, raw="/new", loop=loop)))

    cleared = loop.sessions.get_or_create(key)
    assert outbound.content == "New session started."
    assert cleared.get_history(max_messages=0) == []
