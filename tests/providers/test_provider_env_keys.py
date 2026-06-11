from nanobot.config.schema import Config
from nanobot.providers.factory import make_provider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


def test_perplexity_provider_spec_and_env_key(monkeypatch) -> None:
    spec = find_by_name("perplexity")

    assert spec is not None
    assert spec.env_key == "PERPLEXITY_API_KEY"
    assert spec.backend == "openai_compat"
    assert spec.default_api_base == "https://api.perplexity.ai"

    config = Config.model_validate({
        "agents": {"defaults": {"provider": "perplexity", "model": "sonar-pro"}}
    })

    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    try:
        make_provider(config)
    except ValueError as exc:
        assert "No API key configured" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected missing Perplexity API key to fail")

    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test-key")
    provider = make_provider(config)

    assert isinstance(provider, OpenAICompatProvider)
    assert provider._api_key_for_client == "pplx-test-key"
