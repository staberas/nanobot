# Changelog

## 0.2.2 - 2026-10-08

### Added

- Configurable `thinkingStyle` translation for RKLLAMA and other OpenAI-compatible
  endpoints using `thinking_type`, `enable_thinking`, or `reasoning_split` payloads.
- Matrix attachment downloads with local media paths and attachment metadata passed to
  the agent.
- Matrix reply/thread relations, media uploads, SAS verification improvements, and safer
  Markdown/HTML rendering that preserves trusted `mxc://` image sources.

### Changed

- Matrix group-room addressing continues to support configured text prefixes and aliases,
  including the Hermes and Atlas deployment patterns, while ignoring unaddressed messages.
- Context-pipeline attachment responses now report the received local path and state clearly
  when PDF/file text extraction is unavailable.
- Local attachment breadcrumbs are removed from replayed prompts on POSIX and Windows.

### Compatibility

- RKLLAMA/OpenAI-compatible capability overrides, `plainChatWhenToolsUnsupported`, direct
  URL `web_fetch`, context-pipeline behavior, Matrix `allowFrom`, and both `mention` and
  `mentions` group policies remain supported.
