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
