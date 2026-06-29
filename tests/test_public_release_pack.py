import json
from pathlib import Path

from kvoptbench.reports.generate import generate_report
from kvoptbench.strategy.advisor import (
    build_strategy_advisor_report,
    write_strategy_advisor_outputs,
)


PUBLIC_RELEASE_DIR = Path("examples/public_release")


def test_public_release_fixture_pack_renders_report_and_advisor(tmp_path: Path) -> None:
    required_files = [
        "summary.csv",
        "cache_summary.csv",
        "prefix_sweep.csv",
        "prefill_decode.csv",
        "long_context.csv",
        "kv_quantization.csv",
        "kv_offload.csv",
        "speculative_decoding.csv",
        "disaggregation.csv",
        "mock_benchmark_report.md",
        "strategy_advisor.md",
        "strategy_advisor.json",
    ]
    for filename in required_files:
        assert (PUBLIC_RELEASE_DIR / filename).exists(), f"Missing {filename}"

    rendered_strategy_json = tmp_path / "strategy_advisor.json"
    rendered_strategy_md = tmp_path / "strategy_advisor.md"
    rendered_report = tmp_path / "mock_benchmark_report.md"

    advisor = build_strategy_advisor_report(
        summary_path=PUBLIC_RELEASE_DIR / "summary.csv",
        cache_input_path=PUBLIC_RELEASE_DIR / "cache_summary.csv",
        prefix_sweep_input_path=PUBLIC_RELEASE_DIR / "prefix_sweep.csv",
        prefill_decode_input_path=PUBLIC_RELEASE_DIR / "prefill_decode.csv",
        long_context_input_path=PUBLIC_RELEASE_DIR / "long_context.csv",
        kv_quant_input_path=PUBLIC_RELEASE_DIR / "kv_quantization.csv",
        kv_offload_input_path=PUBLIC_RELEASE_DIR / "kv_offload.csv",
        spec_decoding_input_path=PUBLIC_RELEASE_DIR / "speculative_decoding.csv",
        disagg_input_path=PUBLIC_RELEASE_DIR / "disaggregation.csv",
    )
    write_strategy_advisor_outputs(
        report=advisor,
        json_output_path=rendered_strategy_json,
        markdown_output_path=rendered_strategy_md,
    )
    generate_report(
        input_path=PUBLIC_RELEASE_DIR / "summary.csv",
        output_path=rendered_report,
        cache_input_path=PUBLIC_RELEASE_DIR / "cache_summary.csv",
        prefix_sweep_input_path=PUBLIC_RELEASE_DIR / "prefix_sweep.csv",
        prefill_decode_input_path=PUBLIC_RELEASE_DIR / "prefill_decode.csv",
        long_context_input_path=PUBLIC_RELEASE_DIR / "long_context.csv",
        kv_quant_input_path=PUBLIC_RELEASE_DIR / "kv_quantization.csv",
        kv_offload_input_path=PUBLIC_RELEASE_DIR / "kv_offload.csv",
        spec_decoding_input_path=PUBLIC_RELEASE_DIR / "speculative_decoding.csv",
        disagg_input_path=PUBLIC_RELEASE_DIR / "disaggregation.csv",
        strategy_input_path=rendered_strategy_json,
    )

    payload = json.loads(rendered_strategy_json.read_text(encoding="utf-8"))
    report = rendered_report.read_text(encoding="utf-8")
    advisor_markdown = rendered_strategy_md.read_text(encoding="utf-8")

    assert payload["overall_recommendation"] == "prefix_caching"
    assert payload["recommendations"][0]["strategy"] == "prefix_caching"
    assert "# KVOptBench Mock Benchmark Report" in report
    assert "## Strategy Advisor" in report
    assert "Do not treat mock metrics as real engine benchmark results." in report
    assert "# Strategy Advisor" in advisor_markdown
    assert "Milestone" not in report


def test_reproducibility_guide_documents_public_release_workflow() -> None:
    guide = Path("guides/reproducibility.md")

    text = guide.read_text(encoding="utf-8")

    assert "Public Example Bundle" in text
    assert "kvoptbench strategy-recommend" in text
    assert "kvoptbench report" in text
    assert "--strategy-input" in text
    assert "missing_metrics" in text
    assert "RunPod" in text
    assert "mock metrics" in text


def test_public_release_guides_document_real_endpoint_and_runpod_workflows() -> None:
    real_endpoint = Path("guides/real_endpoint_vllm_sglang.md")
    runpod = Path("guides/runpod.md")
    readme = Path("README.md")

    real_text = real_endpoint.read_text(encoding="utf-8")
    runpod_text = runpod.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")

    assert "examples/vllm_openai_compatible_config.yaml" in real_text
    assert "examples/sglang_openai_compatible_config.yaml" in real_text
    assert "examples/runpod_vllm_openai_compatible_config.yaml" in real_text
    assert "examples/lambda_cloud_vllm_openai_compatible_config.yaml" in real_text
    assert "examples/generic_openai_compatible_config.yaml" in real_text
    assert "kvoptbench endpoint-check" in real_text
    assert "kvoptbench engine-command" in real_text
    assert "--enable-prefix-caching" in real_text
    assert "--disable-radix-cache" in real_text
    assert "missing_metrics" in real_text
    assert "https://docs.vllm.ai" in real_text
    assert "https://docs.sglang.io" in real_text

    assert "https://docs.runpod.io" in runpod_text
    assert "examples/runpod_vllm_openai_compatible_config.yaml" in runpod_text
    assert "examples/runpod_sglang_openai_compatible_config.yaml" in runpod_text
    assert "proxy.runpod.net" in runpod_text
    assert "/workspace" in runpod_text
    assert "/runpod-volume" in runpod_text
    assert "100-second timeout" in runpod_text
    assert "kvoptbench endpoint-check" in runpod_text
    assert "Never commit secrets" in runpod_text

    assert "guides/real_endpoint_vllm_sglang.md" in readme_text
    assert "guides/runpod.md" in readme_text
    assert "guides/first_real_benchmark.md" in readme_text
    assert "Lambda Cloud" in readme_text

    public_text = "\n".join([real_text, runpod_text])
    assert "Milestone" not in public_text
    assert "C:\\Users" not in public_text
    assert "OneDrive" not in public_text


def test_public_release_templates_have_required_publication_sections() -> None:
    result_template = Path("examples/public_release/result_template.md")
    blog_template = Path("examples/public_release/blog_report_template.md")

    result_text = result_template.read_text(encoding="utf-8")
    blog_text = blog_template.read_text(encoding="utf-8")

    for required in [
        "Run Identity",
        "Environment",
        "Backend Launch",
        "Workloads",
        "Artifacts",
        "Strategy Advisor",
        "missing_metrics",
        "Model Revision",
        "Workload Hash",
    ]:
        assert required in result_text

    for required in [
        "Headline",
        "TL;DR",
        "Setup",
        "Findings",
        "Quality And Caveats",
        "Reproduction",
        "Appendix",
    ]:
        assert required in blog_text

    combined = result_text + "\n" + blog_text
    assert "Do not publish mock metrics as real endpoint results" in combined
    assert "Milestone" not in combined
    assert "C:\\Users" not in combined
    assert "OneDrive" not in combined


def test_public_readiness_files_do_not_expose_internal_placeholders() -> None:
    checked_files = [
        Path("README.md"),
        Path("ROADMAP.md"),
        Path("AGENTS.md"),
        Path("CITATION.cff"),
        Path("guides/reproducibility.md"),
        Path("guides/real_endpoint_vllm_sglang.md"),
        Path("guides/runpod.md"),
        Path("guides/first_real_benchmark.md"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_files)

    assert "Milestone" not in combined
    assert "YOUR_USERNAME" not in combined
    assert "C:\\Users" not in combined
    assert "OneDrive" not in combined
    assert "https://github.com/PranayS676/KVOptBench" in combined
