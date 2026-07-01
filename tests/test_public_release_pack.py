import json
from pathlib import Path

from kvoptbench.reports.generate import generate_report
from kvoptbench.strategy.advisor import (
    build_strategy_advisor_report,
    write_strategy_advisor_outputs,
)


PUBLIC_RELEASE_DIR = Path("examples/public_release")

ARCHITECTURE_DOCS = [
    Path("docs/architecture/README.md"),
    Path("docs/architecture/telemetry_lifecycle.md"),
    Path("docs/architecture/environment_capture.md"),
    Path("docs/architecture/import_adapters.md"),
    Path("docs/architecture/strategy_plan_run.md"),
    Path("docs/architecture/advisor_confidence.md"),
    Path("docs/architecture/lmcache_scbench_extensions.md"),
]


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
    assert "guides/datasets.md" in readme_text
    assert "guides/dataset_adapter_contract.md" in readme_text
    assert "guides/frontier_dataset_pack.md" in readme_text
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
        "Benchmark Methodology",
        "Workloads",
        "Artifacts",
        "Metric Provenance",
        "Strategy Advisor",
        "missing_metrics",
        "Model Revision",
        "Workload Hash",
        "Workload sample",
        "Dataset source URL",
        "License review status",
        "Redistribution policy",
        "Dataset adapter version",
        "Dataset manifest hash",
        "Prompt template hash",
        "Tokenizer id",
        "Token count method",
        "Run manifest JSON",
        "Missing metrics JSON",
        "Known limitations",
        "Engine-reported cache hit rate",
        "Cache hit proxy",
        "source_type",
        "measurement_method",
        "Run order",
        "Randomization seed",
        "Repetition count",
        "Confidence interval",
        "Effect size",
        "Advisor confidence",
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
        Path("guides/benchmark_validity.md"),
        Path("guides/metric_provenance.md"),
        Path("guides/reproducibility.md"),
        Path("guides/real_endpoint_vllm_sglang.md"),
        Path("guides/runpod.md"),
        Path("guides/first_real_benchmark.md"),
        Path("guides/datasets.md"),
        Path("guides/dataset_adapter_contract.md"),
        Path("guides/frontier_dataset_pack.md"),
        *ARCHITECTURE_DOCS,
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_files)

    assert "Milestone" not in combined
    assert "YOUR_USERNAME" not in combined
    assert "C:\\Users" not in combined
    assert "OneDrive" not in combined
    assert "KVOptBench_Strategic_Direction_Memo.docx" not in combined
    assert "https://github.com/PranayS676/KVOptBench" in combined


def test_architecture_docs_cover_non_gpu_design_backlog() -> None:
    for path in ARCHITECTURE_DOCS:
        assert path.exists(), f"Missing architecture doc: {path}"

    readme = Path("README.md").read_text(encoding="utf-8")
    telemetry = Path("docs/architecture/telemetry_lifecycle.md").read_text(encoding="utf-8")
    environment = Path("docs/architecture/environment_capture.md").read_text(encoding="utf-8")
    imports = Path("docs/architecture/import_adapters.md").read_text(encoding="utf-8")
    strategy = Path("docs/architecture/strategy_plan_run.md").read_text(encoding="utf-8")
    advisor = Path("docs/architecture/advisor_confidence.md").read_text(encoding="utf-8")
    lmcache = Path("docs/architecture/lmcache_scbench_extensions.md").read_text(encoding="utf-8")

    assert "docs/architecture/README.md" in readme

    for required in [
        "Prometheus",
        "GPU sampler",
        "missing_metrics",
        "result package",
        "No fabricated metrics",
    ]:
        assert required in telemetry

    for required in [
        "engine_version",
        "model_revision",
        "cuda_version",
        "gpu_type",
        "gpu_count",
        "backend_launch_command",
        "config_sha256",
        "workload_sha256",
    ]:
        assert required in environment

    for required in [
        "vLLM bench",
        "GenAI-Perf",
        "AIPerf",
        "metric mapping registry",
        "source_type=imported",
        "absolute local paths",
    ]:
        assert required in imports

    for required in [
        "strategy-plan",
        "strategy-run",
        "randomization",
        "repetitions",
        "confidence intervals",
        "effect size",
    ]:
        assert required in strategy

    for required in [
        "workload-aware",
        "quality gates",
        "missing telemetry",
        "Next-Experiment Command Generation",
    ]:
        assert required in advisor

    for required in [
        "LMCache",
        "SCBench",
        "KV Generation",
        "Compression",
        "Retrieval",
        "Loading",
        "does not implement",
    ]:
        assert required in lmcache


def test_benchmark_methodology_guides_define_validity_and_metric_provenance() -> None:
    validity = Path("guides/benchmark_validity.md").read_text(encoding="utf-8")
    provenance = Path("guides/metric_provenance.md").read_text(encoding="utf-8")
    first_real = Path("guides/first_real_benchmark.md").read_text(encoding="utf-8")
    real_endpoint = Path("guides/real_endpoint_vllm_sglang.md").read_text(encoding="utf-8")
    reproducibility = Path("guides/reproducibility.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    for required in [
        "What KVOptBench Can Claim",
        "What KVOptBench Refuses To Claim",
        "Exploratory vs Publishable Results",
        "Mock results validate the pipeline",
        "Synthetic workloads are smoke tests",
        "Failed requests stay in the result package",
        "random-prefix controls",
        "randomized condition order",
        "repeated trials",
        "confidence intervals",
    ]:
        assert required in validity

    for required in [
        "client_observed",
        "provider_reported",
        "engine_reported",
        "gpu_reported",
        "imported",
        "derived",
        "estimated",
        "estimated_output_tokens",
        "provider_completion_tokens",
        "cache_hit_proxy",
        "engine_reported_cache_hit_rate",
        "metric_provenance",
    ]:
        assert required in provenance

    assert "guides/benchmark_validity.md" in readme
    assert "guides/metric_provenance.md" in readme
    assert "randomized condition order" in first_real
    assert "repeated trials" in first_real
    assert "Prometheus" in real_endpoint
    assert "nvidia-smi" in real_endpoint
    assert "official or exploratory" in reproducibility


def test_dataset_docs_define_public_workload_pack_and_adapter_contract() -> None:
    datasets = Path("guides/datasets.md").read_text(encoding="utf-8")
    contract = Path("guides/dataset_adapter_contract.md").read_text(encoding="utf-8")
    frontier_pack = Path("guides/frontier_dataset_pack.md").read_text(encoding="utf-8")

    for required_source in [
        "QASPER",
        "https://huggingface.co/datasets/allenai/qasper",
        "Project Gutenberg",
        "https://www.gutenberg.org/",
        "LongBench",
        "https://huggingface.co/datasets/zai-org/LongBench",
        "BEIR",
        "https://github.com/beir-cellar/beir",
        "beir_scifact",
        "Natural Questions",
        "https://github.com/google-research-datasets/natural-questions",
        "SWE-bench",
        "CodeSearchNet",
        "BFCL",
        "bfcl: tool_calling",
    ]:
        assert required_source in datasets

    for required_field in [
        "CLI Usage",
        "--download",
        "--cache-dir",
        "--dataset-revision",
        "--subset",
        "--force",
        "qasper.py",
        "gutenberg.py",
        "longbench.py",
        "beir.py",
        "bfcl.py",
        "KVOPTBENCH_DATASET_DOWNLOAD",
        "dataset_source_url",
        "source_license",
        "prefix_group_id",
        "shared_prefix_tokens",
        "prefix_hash",
        "prompt_hash",
        "tokenizer_id",
        "tokenizer_revision",
        "measured_input_tokens",
        "measured_shared_prefix_tokens",
        "truncation_policy",
        "redistributable_prompt",
        "license_review_status",
        "redistribution_policy",
        "kvoptbench_version",
        "git_commit",
        "prompt_template_hash",
        "workload_sha256",
        "generation_command",
        "prompt_template",
        "engine_reported_cache_hit_rate",
        "cache_hit_proxy",
        "Comparability Rules",
    ]:
        assert required_field in contract

    for required_workload in [
        "qasper_shared_prefix_8k.jsonl",
        "qasper_shared_prefix_32k.jsonl",
        "qasper_random_prefix_8k.jsonl",
        "qasper_random_prefix_32k.jsonl",
        "QASPER prefix overlap sweep",
        "gutenberg_needle_8k_128k.jsonl",
        "LongBench Core Subset",
        "Small Public RAG Pack",
        "BFCL Tool-Calling Pack",
        "one self-hosted vLLM or SGLang endpoint",
        "50-100 shared-prefix",
        "README_result.md",
        "run_manifest.json",
        "dataset_manifest_shared.json",
        "missing_metrics.json",
        "Direct comparison is not credible",
    ]:
        assert required_workload in frontier_pack

    for required_dataset_guidance in [
        "Implemented adapters",
        "qasper: shared_prefix, random_prefix, partial_prefix_sweep",
        "gutenberg: needle, no_needle_control, multi_needle, conflicting_needle",
        "longbench: long_context_qa, long_context_retrieval, code_context",
        "partial-prefix sweep",
        "0%, 25%, 50%, 75%, and 90%",
        "engine_reported_cache_hit_rate",
        "cache_hit_proxy",
        "redistribution policy",
        "license review status",
    ]:
        assert required_dataset_guidance in datasets
