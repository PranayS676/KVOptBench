"""Engine strategy profiles and command previews.

These profiles describe how an operator may start a compatible server. KVOptBench
does not execute these commands or manage server lifecycle here.
"""

from __future__ import annotations

from kvoptbench.schemas import (
    EngineCommandPreview,
    EngineEndpoint,
    EngineProfile,
    EngineStrategyProfile,
)


def _vllm_profile() -> EngineProfile:
    base = "python -m vllm.entrypoints.openai.api_server --model {model_id} --host {host} --port {port}"
    return EngineProfile(
        engine="vllm",
        display_name="vLLM",
        default_port=8000,
        strategies={
            "baseline": EngineStrategyProfile(
                name="baseline",
                description="Baseline OpenAI-compatible vLLM serving preview.",
                command_template=base,
            ),
            "cache_off": EngineStrategyProfile(
                name="cache_off",
                description="vLLM prefix-cache-off control preview.",
                command_template=base,
                notes=[
                    "Use this as the cache-disabled control and verify exact flags for the installed vLLM version.",
                ],
            ),
            "cache_on": EngineStrategyProfile(
                name="cache_on",
                description="vLLM prefix caching preview for shared-prefix workloads.",
                command_template=base + " --enable-prefix-caching",
            ),
            "kv_fp8": EngineStrategyProfile(
                name="kv_fp8",
                description="vLLM FP8 KV cache preview.",
                command_template=base + " --kv-cache-dtype fp8",
                notes=["Validate model and engine support before treating results as official."],
            ),
            "kv_offload": EngineStrategyProfile(
                name="kv_offload",
                description="vLLM KV offload preview.",
                command_template=base + " <kv-offload-flags>",
                notes=[
                    "Replace <kv-offload-flags> with supported offload or cache-transfer flags "
                    "for the installed vLLM version before official runs.",
                ],
                placeholder=True,
            ),
            "speculative_decoding": EngineStrategyProfile(
                name="speculative_decoding",
                description="vLLM speculative decoding preview.",
                command_template=base + " --speculative-model <draft-model>",
                notes=["Replace <draft-model> with a compatible draft model before use."],
                placeholder=True,
            ),
            "prefill_decode_disaggregation": EngineStrategyProfile(
                name="prefill_decode_disaggregation",
                description="vLLM prefill/decode disaggregation preview.",
                command_template=base + " <prefill-decode-disaggregation-flags>",
                notes=[
                    "Replace <prefill-decode-disaggregation-flags> with the supported "
                    "multi-process or disaggregated serving setup for the installed vLLM version.",
                ],
                placeholder=True,
            ),
        },
        notes=["Command previews assume an OpenAI-compatible server endpoint."],
    )


def _sglang_profile() -> EngineProfile:
    base = (
        "python -m sglang.launch_server --model-path {model_id} --host {host} --port {port}"
    )
    return EngineProfile(
        engine="sglang",
        display_name="SGLang",
        default_port=30000,
        strategies={
            "baseline": EngineStrategyProfile(
                name="baseline",
                description="Baseline OpenAI-compatible SGLang serving preview.",
                command_template=base,
            ),
            "cache_off": EngineStrategyProfile(
                name="cache_off",
                description="SGLang radix-cache-off control preview.",
                command_template=base + " --disable-radix-cache",
            ),
            "cache_on": EngineStrategyProfile(
                name="cache_on",
                description="SGLang radix cache preview for shared-prefix workloads.",
                command_template=base,
                notes=["Radix caching is the cache-on condition for this profile."],
            ),
            "kv_fp8": EngineStrategyProfile(
                name="kv_fp8",
                description="SGLang FP8 KV cache preview.",
                command_template=base + " --kv-cache-dtype fp8_e5m2",
                notes=["Validate supported KV cache dtype values for the installed SGLang version."],
            ),
            "kv_offload": EngineStrategyProfile(
                name="kv_offload",
                description="SGLang KV offload preview.",
                command_template=base + " <kv-offload-flags>",
                notes=[
                    "Replace <kv-offload-flags> with supported offload or cache-transfer flags "
                    "for the installed SGLang version before official runs.",
                ],
                placeholder=True,
            ),
            "speculative_decoding": EngineStrategyProfile(
                name="speculative_decoding",
                description="SGLang speculative decoding preview.",
                command_template=base + " --speculative-algorithm EAGLE --speculative-draft-model-path <draft-model>",
                notes=["Replace <draft-model> and algorithm flags with a supported SGLang setup."],
                placeholder=True,
            ),
            "prefill_decode_disaggregation": EngineStrategyProfile(
                name="prefill_decode_disaggregation",
                description="SGLang prefill/decode disaggregation preview.",
                command_template=base + " <prefill-decode-disaggregation-flags>",
                notes=[
                    "Replace <prefill-decode-disaggregation-flags> with the supported "
                    "multi-process or disaggregated serving setup for the installed SGLang version.",
                ],
                placeholder=True,
            ),
        },
        notes=["Command previews assume an OpenAI-compatible server endpoint."],
    )


_PROFILES = {
    "vllm": _vllm_profile(),
    "sglang": _sglang_profile(),
}


def list_engine_profiles() -> list[EngineProfile]:
    """Return all built-in engine profiles."""
    return list(_PROFILES.values())


def get_engine_profile(engine: str) -> EngineProfile:
    """Look up an engine profile by name."""
    normalized = engine.strip().lower()
    if normalized not in _PROFILES:
        valid = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown engine profile '{engine}'. Valid profiles: {valid}")
    return _PROFILES[normalized]


def render_command_preview(
    *,
    engine: str,
    strategy: str,
    model_id: str,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> EngineCommandPreview:
    """Render a command preview for a strategy without launching anything."""
    profile = get_engine_profile(engine)
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in profile.strategies:
        valid = ", ".join(sorted(profile.strategies))
        raise ValueError(
            f"Unknown strategy '{strategy}' for engine '{profile.engine}'. Valid strategies: {valid}"
        )
    selected = profile.strategies[normalized_strategy]
    selected_port = port if port is not None else profile.default_port
    command = selected.command_template.format(model_id=model_id, host=host, port=selected_port)
    notes = ["Server command preview only; KVOptBench does not launch or manage this server."]
    notes.extend(profile.notes)
    notes.extend(selected.notes)
    if selected.placeholder:
        notes.append("This strategy requires engine-specific validation before an official run.")
    return EngineCommandPreview(
        engine=profile.engine,
        strategy=selected.name,
        description=selected.description,
        command=command,
        endpoint=EngineEndpoint(base_url=f"http://{host}:{selected_port}/v1"),
        launches_server=False,
        notes=" ".join(notes),
    )

