"""Heuristic tool selection for per-turn LLM tool budgets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class ToolSelectionConfig:
    """Runtime copy of agent tool-selection settings."""

    enabled: bool = False
    mode: str = "heuristic"
    max_tools: int = 4
    always_include: tuple[str, ...] = field(default_factory=tuple)
    allow: tuple[str, ...] = field(default_factory=tuple)
    deny: tuple[str, ...] = field(default_factory=tuple)


_WEB_RE = re.compile(
    r"\b(search|web|internet|online|current|latest|today|news|recent|now|date|weather|price|lookup|look up)\b",
    re.IGNORECASE,
)
_FETCH_RE = re.compile(
    r"\b(fetch|open|read|summari[sz]e|extract|visit|page|url|link|article|website|http[s]?://)\b",
    re.IGNORECASE,
)
_CRON_RE = re.compile(
    r"\b(schedule|remind|reminder|repeat|recurring|cron|timer|alarm|every|tomorrow|daily|weekly|monthly)\b",
    re.IGNORECASE,
)
_DELEGATE_RE = re.compile(
    r"\b(delegate|subagent|spawn|background|parallel|hand off|long[- ]?task|sustained goal|keep working)\b",
    re.IGNORECASE,
)
_FILE_READ_RE = re.compile(
    r"\b(read|show|cat|list|find|grep|search files?|inspect|open)\b.*\b(file|dir|directory|repo|workspace|path|code)\b|\b(grep|find files?|list dir)\b",
    re.IGNORECASE,
)
_FILE_WRITE_RE = re.compile(
    r"\b(edit|write|create|modify|patch|apply patch|change|fix|update)\b.*\b(file|code|repo|workspace|path)\b|\b(apply_patch|write_file|edit_file)\b",
    re.IGNORECASE,
)


def _message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
    return ""


def _schema_name(schema: dict[str, Any]) -> str:
    fn = schema.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return schema.get("name") if isinstance(schema.get("name"), str) else ""


def _pick_existing(candidates: list[str], available: set[str]) -> list[str]:
    return [name for name in candidates if name in available]


def heuristic_tool_names(prompt: str, available: set[str]) -> list[str]:
    """Return conservative dynamic tool candidates for *prompt*."""
    selected: list[str] = []

    def add(names: list[str]) -> None:
        for name in _pick_existing(names, available):
            if name not in selected:
                selected.append(name)

    if _WEB_RE.search(prompt):
        add(["web_search"])
        if _FETCH_RE.search(prompt):
            add(["web_fetch"])
    elif _FETCH_RE.search(prompt) and "web_fetch" in available:
        add(["web_fetch"])

    if _CRON_RE.search(prompt):
        add(["cron"])
    if _DELEGATE_RE.search(prompt):
        add(["spawn", "long_task"])
    if _FILE_WRITE_RE.search(prompt):
        add(["edit_file", "write_file", "apply_patch"])
    elif _FILE_READ_RE.search(prompt):
        add(["read_file", "grep", "find_files", "list_dir"])

    return selected


def select_tool_definitions(
    definitions: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    config: ToolSelectionConfig | None,
    *,
    session_key: str | None = None,
    context_window_tokens: int | None = None,
    estimate_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Filter tool schemas according to runtime tool-selection settings."""
    if config is None or not config.enabled:
        return definitions

    by_name = {_schema_name(schema): schema for schema in definitions if _schema_name(schema)}
    available = set(by_name)
    deny = set(config.deny)
    allow = set(config.allow)

    def permitted(name: str) -> bool:
        if name in deny:
            return False
        return not allow or name in allow

    selected_names: list[str] = []
    for name in config.always_include:
        if name in by_name and permitted(name) and name not in selected_names:
            selected_names.append(name)

    prompt = _message_text(messages)
    dynamic_candidates = heuristic_tool_names(prompt, available)
    dynamic_count = 0
    max_dynamic = max(0, config.max_tools)
    for name in dynamic_candidates:
        if dynamic_count >= max_dynamic:
            break
        if not permitted(name) or name in selected_names:
            continue
        selected_names.append(name)
        dynamic_count += 1

    selected = [by_name[name] for name in selected_names]
    logger.info(
        "Selected tools for session {}: {} from {} registered tools; estimated prompt tokens {}/{}",
        session_key or "default",
        selected_names,
        len(definitions),
        estimate_tokens if estimate_tokens is not None else "?",
        context_window_tokens if context_window_tokens is not None else "?",
    )
    return selected
