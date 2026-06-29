from kvoptbench.engines.profiles import (
    get_engine_profile,
    list_engine_profiles,
    render_command_preview,
)


def test_engine_registry_includes_vllm_and_sglang_with_matching_core_strategies() -> None:
    profiles = {profile.engine: profile for profile in list_engine_profiles()}

    assert {"vllm", "sglang"}.issubset(profiles)
    for engine in ["vllm", "sglang"]:
        strategy_names = set(profiles[engine].strategies)
        assert {
            "baseline",
            "cache_off",
            "cache_on",
            "kv_fp8",
            "kv_offload",
            "speculative_decoding",
            "prefill_decode_disaggregation",
        }.issubset(strategy_names)


def test_get_engine_profile_rejects_unknown_engine() -> None:
    try:
        get_engine_profile("unknown-engine")
    except ValueError as exc:
        assert "Unknown engine profile" in str(exc)
    else:
        raise AssertionError("Expected unknown engine profile to raise")


def test_render_command_preview_returns_documented_non_launching_command() -> None:
    preview = render_command_preview(
        engine="vllm",
        strategy="cache_on",
        model_id="example/model",
        host="0.0.0.0",
        port=8000,
    )

    assert preview.engine == "vllm"
    assert preview.strategy == "cache_on"
    assert preview.launches_server is False
    assert "example/model" in preview.command
    assert "--enable-prefix-caching" in preview.command
    assert preview.endpoint.base_url == "http://0.0.0.0:8000/v1"
    assert "server command preview only" in preview.notes.lower()


def test_render_sglang_cache_command_uses_radix_cache_language() -> None:
    preview = render_command_preview(
        engine="sglang",
        strategy="cache_on",
        model_id="example/model",
        host="127.0.0.1",
        port=30000,
    )

    assert preview.engine == "sglang"
    assert "radix" in preview.description.lower()
    assert "--disable-radix-cache" not in preview.command
    assert preview.endpoint.healthcheck_path == "/v1/models"


def test_render_kv_offload_command_is_marked_as_placeholder() -> None:
    for engine in ["vllm", "sglang"]:
        preview = render_command_preview(
            engine=engine,
            strategy="kv_offload",
            model_id="example/model",
        )

        assert preview.strategy == "kv_offload"
        assert "<kv-offload-flags>" in preview.command
        assert "engine-specific validation" in preview.notes


def test_render_advanced_inference_commands_are_marked_as_placeholders() -> None:
    speculative = render_command_preview(
        engine="vllm",
        strategy="speculative_decoding",
        model_id="example/model",
    )
    assert speculative.strategy == "speculative_decoding"
    assert "<draft-model>" in speculative.command
    assert "engine-specific validation" in speculative.notes

    for engine in ["vllm", "sglang"]:
        disaggregation = render_command_preview(
            engine=engine,
            strategy="prefill_decode_disaggregation",
            model_id="example/model",
        )

        assert disaggregation.strategy == "prefill_decode_disaggregation"
        assert "<prefill-decode-disaggregation-flags>" in disaggregation.command
        assert "engine-specific validation" in disaggregation.notes

