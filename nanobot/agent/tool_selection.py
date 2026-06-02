"""Heuristic tool selection for small-context model requests."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.utils.helpers import estimate_prompt_tokens_chain

_TOOL_FETCH_HINTS = re.compile(r"\b(fetch|open|read|summari[sz]e|page|url|link|article|website|webpage|http[s]?://)\b", re.I)
_TOOL_WEB_HINTS = re.compile(r"\b(web|search|internet|current|latest|today|date|news|recent|lookup|look up|find online|google)\b", re.I)
_TOOL_CRON_HINTS = re.compile(r"\b(schedule|remind|reminder|repeat|recurring|cron|timer|alarm|every\s+\d|tomorrow|daily|weekly|monthly)\b", re.I)
_TOOL_SPAWN_HINTS = re.compile(r"\b(delegate|subagent|sub-agent|spawn|background|parallel|another agent|worker agent)\b", re.I)
_TOOL_LONG_TASK_HINTS = re.compile(r"\b(long[- ]?task|sustained goal|keep working|autonomously|over time)\b", re.I)
_TOOL_FILE_READ_HINTS = re.compile(r"\b(read|show|cat|list|find|grep|search files?|inspect)\b", re.I)
_TOOL_FILE_EDIT_HINTS = re.compile(r"\b(edit|write|create|patch|modify|replace|delete|fix|implement|change)\b", re.I)
_FILE_OBJECT_HINTS = re.compile(r"\b(file|directory|folder|repo|repository|code|path|workspace)\b|[/\\][\w.-]+", re.I)


@dataclass(frozen=True)
class ToolSelectionResult:
    """Tool definitions selected for one model request."""

    tools: list[dict[str, Any]]
    selected_names: list[str]
    registered_count: int
    prompt_tokens: int = 0
    budget_tokens: int | None = None
    source: str = "none"


@dataclass
class ToolSelectionPolicy:
    """Runtime copy of config-driven tool selection policy."""

    enabled: bool = False
    mode: str = "heuristic"
    max_tools: int = 4
    always_include: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Any | None) -> "ToolSelectionPolicy":
        if config is None:
            return cls()
        return cls(
            enabled=bool(getattr(config, "enabled", False)),
            mode=str(getattr(config, "mode", "heuristic")),
            max_tools=int(getattr(config, "max_tools", 4)),
            always_include=list(getattr(config, "always_include", []) or []),
            allow=list(getattr(config, "allow", []) or []),
            deny=list(getattr(config, "deny", []) or []),
        )


def _tool_name(schema: dict[str, Any]) -> str:
    fn = schema.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    name = schema.get("name")
    return name if isinstance(name, str) else ""


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(p for p in parts if p)
    return ""


def _dedupe(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def heuristic_tool_names(text: str) -> list[str]:
    """Conservatively infer relevant tool names from the user's latest request."""
    selected: list[str] = []
    if not text or not text.strip():
        return selected
    if _TOOL_WEB_HINTS.search(text) or re.search(r"http[s]?://", text, re.I):
        selected.append("web_search")
        if _TOOL_FETCH_HINTS.search(text):
            selected.append("web_fetch")
    if _TOOL_CRON_HINTS.search(text):
        selected.append("cron")
    if _TOOL_SPAWN_HINTS.search(text):
        selected.append("spawn")
    if _TOOL_LONG_TASK_HINTS.search(text):
        selected.append("long_task")
    if _FILE_OBJECT_HINTS.search(text):
        if _TOOL_FILE_EDIT_HINTS.search(text):
            selected.extend(["apply_patch", "edit_file", "write_file"])
        elif _TOOL_FILE_READ_HINTS.search(text):
            selected.extend(["read_file", "grep", "find_files", "list_dir"])
    return _dedupe(selected)


def _truncate_tool_description(schema: dict[str, Any], limit: int) -> dict[str, Any]:
    clean = copy.deepcopy(schema)
    targets: list[dict[str, Any]] = [clean]
    fn = clean.get("function")
    if isinstance(fn, dict):
        targets.append(fn)
    for target in targets:
        desc = target.get("description")
        if isinstance(desc, str) and len(desc) > limit:
            target["description"] = desc[: max(0, limit - 1)].rstrip() + "…"
    return clean


def select_tools_for_request(
    *,
    all_tools: list[dict[str, Any]],
    policy: ToolSelectionPolicy | Any | None,
    messages: list[dict[str, Any]],
    provider: Any,
    model: str | None,
    context_window_tokens: int | None = None,
    description_limit: int = 80,
    session_key: str | None = None,
) -> ToolSelectionResult:
    """Return the model-visible tool schemas for this request."""
    runtime_policy = policy if isinstance(policy, ToolSelectionPolicy) else ToolSelectionPolicy.from_config(policy)
    registered_count = len(all_tools)
    if not runtime_policy.enabled:
        prompt_tokens, source = estimate_prompt_tokens_chain(provider, model, messages, all_tools)
        return ToolSelectionResult(all_tools, [_tool_name(t) for t in all_tools], registered_count, prompt_tokens, context_window_tokens, source)

    by_name = {_tool_name(schema): schema for schema in all_tools if _tool_name(schema)}
    deny = set(runtime_policy.deny)
    allow = set(runtime_policy.allow)

    always = [n for n in runtime_policy.always_include if n in by_name and n not in deny]
    if allow:
        always = [n for n in always if n in allow]

    dynamic_candidates = heuristic_tool_names(_latest_user_text(messages))
    dynamic: list[str] = []
    for name in dynamic_candidates:
        if name in deny or name in always or name not in by_name:
            continue
        if allow and name not in allow:
            continue
        dynamic.append(name)
        if len(dynamic) >= runtime_policy.max_tools:
            break

    selected_names = _dedupe(always + dynamic)
    selected = [by_name[name] for name in selected_names]
    prompt_tokens, source = estimate_prompt_tokens_chain(provider, model, messages, selected)

    if context_window_tokens and prompt_tokens > context_window_tokens and selected:
        selected = [_truncate_tool_description(tool, description_limit) for tool in selected]
        prompt_tokens, source = estimate_prompt_tokens_chain(provider, model, messages, selected)

    if context_window_tokens and prompt_tokens > context_window_tokens and dynamic:
        selected_names = always
        selected = [by_name[name] for name in selected_names]
        prompt_tokens, source = estimate_prompt_tokens_chain(provider, model, messages, selected)

    logger.info(
        "Selected tools for session {}: {} from {} registered tools; estimated prompt tokens {}/{} ({})",
        session_key or "<unknown>",
        selected_names,
        registered_count,
        prompt_tokens,
        context_window_tokens or "?",
        source,
    )
    return ToolSelectionResult(selected, selected_names, registered_count, prompt_tokens, context_window_tokens, source)
