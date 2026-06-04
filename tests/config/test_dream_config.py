from nanobot.config.schema import DreamConfig


def test_dream_config_defaults_to_interval_hours() -> None:
    cfg = DreamConfig()

    assert cfg.interval_h == 2
    assert cfg.cron is None


def test_dream_config_builds_every_schedule_from_interval() -> None:
    cfg = DreamConfig(interval_h=3)

    schedule = cfg.build_schedule("UTC")

    assert schedule.kind == "every"
    assert schedule.every_ms == 3 * 3_600_000
    assert schedule.expr is None


def test_dream_config_honors_legacy_cron_override() -> None:
    cfg = DreamConfig.model_validate({"cron": "0 */4 * * *"})

    schedule = cfg.build_schedule("UTC")

    assert schedule.kind == "cron"
    assert schedule.expr == "0 */4 * * *"
    assert schedule.tz == "UTC"
    assert cfg.describe_schedule() == "cron 0 */4 * * * (legacy)"


def test_dream_config_dump_uses_interval_h_and_hides_legacy_cron() -> None:
    cfg = DreamConfig.model_validate({"intervalH": 5, "cron": "0 */4 * * *"})

    dumped = cfg.model_dump(by_alias=True)

    assert dumped["intervalH"] == 5
    assert "cron" not in dumped


def test_dream_config_uses_model_override_name_and_accepts_legacy_model() -> None:
    cfg = DreamConfig.model_validate({"model": "openrouter/sonnet"})

    dumped = cfg.model_dump(by_alias=True)

    assert cfg.model_override == "openrouter/sonnet"
    assert dumped["modelOverride"] == "openrouter/sonnet"
    assert "model" not in dumped


def test_dream_config_accepts_toolless_fallback_flags() -> None:
    cfg = DreamConfig.model_validate({
        "toolsRequired": False,
        "skipWhenToolsUnsupported": True,
        "plainChatFallback": False,
    })

    dumped = cfg.model_dump(by_alias=True)

    assert cfg.tools_required is False
    assert cfg.skip_when_tools_unsupported is True
    assert cfg.plain_chat_fallback is False
    assert dumped["toolsRequired"] is False
    assert dumped["skipWhenToolsUnsupported"] is True
    assert dumped["plainChatFallback"] is False


def test_top_level_memory_dream_exact_live_config_parses() -> None:
    from nanobot.config.schema import Config

    config = Config.model_validate({
        "memory": {
            "dream": {
                "enabled": True,
                "toolsRequired": False,
                "skipWhenToolsUnsupported": False,
                "plainChatFallback": True,
            }
        },
        "providers": {
            "rkllama": {
                "apiKey": None,
                "apiBase": "http://192.168.100.23:30082/v1",
                "capabilities": {"tools": False, "preferMaxTokens": True},
            }
        },
        "agents": {
            "defaults": {
                "provider": "rkllama",
                "model": "Qwen3-4B-w8a8-npu",
            }
        },
    })

    assert config.memory.dream.enabled is True
    assert config.memory.dream.tools_required is False
    assert config.memory.dream.skip_when_tools_unsupported is False
    assert config.memory.dream.plain_chat_fallback is True
    assert config.effective_dream_config is config.memory.dream
    assert config.agents.defaults.provider == "rkllama"
    assert config.providers.__pydantic_extra__["rkllama"].api_base == "http://192.168.100.23:30082/v1"


def test_load_config_does_not_fallback_when_top_level_memory_dream_present(tmp_path) -> None:
    import json

    from nanobot.config.loader import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "memory": {
            "dream": {
                "enabled": True,
                "toolsRequired": False,
                "skipWhenToolsUnsupported": False,
                "plainChatFallback": True,
            }
        },
        "providers": {
            "rkllama": {
                "apiKey": None,
                "apiBase": "http://192.168.100.23:30082/v1",
                "capabilities": {"tools": False, "preferMaxTokens": True},
            }
        },
        "agents": {
            "defaults": {
                "provider": "rkllama",
                "model": "Qwen3-4B-w8a8-npu",
            }
        },
    }), encoding="utf-8")

    config = load_config(path)

    assert config.agents.defaults.provider == "rkllama"
    assert config.agents.defaults.model == "Qwen3-4B-w8a8-npu"
    assert config.effective_dream_config.plain_chat_fallback is True
