# RunPod Runbook

This runbook shows how to run KVOptBench against an OpenAI-compatible vLLM or SGLang server hosted on RunPod. KVOptBench still runs outside the serving engine: it sends requests, records metrics, writes JSONL/CSV/Markdown artifacts, and recommends strategies from measured comparison outputs.

Primary upstream references:

- RunPod Pods: https://docs.runpod.io/pods
- RunPod exposed HTTP ports: https://docs.runpod.io/pods/configuration/expose-ports
- RunPod Network Volumes: https://docs.runpod.io/storage/network-volumes
- RunPod Serverless endpoints: https://docs.runpod.io/serverless/endpoints

## Recommended Path

Start with a RunPod Pod, not a queue-style Serverless handler. KVOptBench expects a long-running OpenAI-compatible HTTP endpoint with `/v1/models` and `/v1/chat/completions`.

Use Serverless only when the endpoint exposes an HTTP/TCP port that behaves like a normal OpenAI-compatible server. Queue-only handlers are not a drop-in replacement for KVOptBench's `base_url` workflow.

## RunPod Pod Checklist

1. Choose a GPU large enough for the served model and context window.
2. Attach storage if model downloads or result artifacts need to survive Pod restarts.
3. Use `/workspace` for normal Pod workspace files.
4. Use `/runpod-volume` when a RunPod network volume is attached.
5. Expose the serving HTTP port:
   - vLLM default in this repo: `8000`
   - SGLang default in this repo: `30000`
6. Start the server with `--host 0.0.0.0` so it is reachable through the Pod networking layer.
7. Build the public proxy URL in the form `https://<pod-id>-<port>.proxy.runpod.net`.
8. Keep benchmark clients aware of RunPod's HTTP proxy behavior, including the 100-second timeout for proxied HTTP requests.
9. Stop or terminate the Pod after the run if it is no longer needed.

Never commit secrets. Keep provider tokens, Hugging Face tokens, RunPod API tokens, and endpoint credentials in environment variables or the RunPod secret/config surface.

## vLLM On RunPod

Inside the Pod:

```bash
vllm serve your/model --host 0.0.0.0 --port 8000
```

From the KVOptBench machine, set the config:

```yaml
provider: runpod
engine: vllm
endpoint_type: vllm
base_url: https://<pod-id>-8000.proxy.runpod.net/v1
healthcheck_path: /v1/models
model_id: your/model
strategy: baseline
api_key_env: null
```

Then run:

```bash
kvoptbench validate-config --config examples/vllm_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/vllm_openai_compatible_config.yaml
kvoptbench run --config examples/vllm_openai_compatible_config.yaml
```

For strategy runs, use the command previews in `guides/real_endpoint_vllm_sglang.md` and replace placeholders with flags validated against the installed vLLM version.

## SGLang On RunPod

Inside the Pod:

```bash
python -m sglang.launch_server --model-path your/model --host 0.0.0.0 --port 30000
```

From the KVOptBench machine, set the config:

```yaml
provider: runpod
engine: sglang
endpoint_type: sglang
base_url: https://<pod-id>-30000.proxy.runpod.net/v1
healthcheck_path: /v1/models
model_id: your/model
strategy: baseline
api_key_env: null
```

Then run:

```bash
kvoptbench validate-config --config examples/sglang_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/sglang_openai_compatible_config.yaml
kvoptbench run --config examples/sglang_openai_compatible_config.yaml
```

For a cache-disabled SGLang control, validate `--disable-radix-cache` against the installed SGLang version before official publication.

## Workload And Artifact Layout

Suggested Pod-local layout:

```text
/workspace/kvoptbench/
/workspace/models/
/workspace/results/
/runpod-volume/model-cache/
/runpod-volume/kvoptbench-artifacts/
```

Suggested local artifact layout after runs:

```text
results/raw/
results/summary.csv
results/cache_summary.csv
results/prefix_sweep.csv
results/prefill_decode.csv
results/long_context.csv
results/kv_quantization.csv
results/kv_offload.csv
results/speculative_decoding.csv
results/disaggregation.csv
reports/outputs/strategy_advisor.json
reports/outputs/strategy_advisor.md
reports/outputs/real_endpoint_report.md
```

Generated artifacts remain ignored by git unless intentionally copied into `examples/public_release/` as small deterministic public examples.

## Run Sequence

1. Generate or copy the workload.
2. Start the vLLM or SGLang server in the Pod.
3. Confirm `/v1/models` through the RunPod proxy URL.
4. Run `kvoptbench endpoint-check`.
5. Run the benchmark config.
6. Summarize raw JSONL.
7. Run the relevant comparison command.
8. Generate the Strategy Advisor JSON/Markdown.
9. Generate the combined report with `--strategy-input`.
10. Fill `examples/public_release/result_template.md`.
11. Fill `examples/public_release/blog_report_template.md` only after deciding the result is publishable.

## RunPod Serverless Note

RunPod Serverless can be useful for production serving experiments, but only use it with KVOptBench when the endpoint exposes a normal HTTP/TCP serving port. If the deployment is request-queue based, add a dedicated adapter later rather than forcing it through the OpenAI-compatible `base_url` runner.

## Failure Checklist

- `endpoint-check` fails: verify the Pod is running, the port is exposed, and the server is bound to `0.0.0.0`.
- `/v1/models` works but benchmark requests fail: verify `model_id`, auth, request timeout, and max context length.
- Long-context requests fail: lower `context_buckets`, increase request timeout, or use a larger GPU.
- GPU memory is absent: keep `gpu_memory_peak_gb` null and list it in `missing_metrics`.
- Proxy timeout appears: lower per-request work or use a direct networking path appropriate for the deployment.

## Publication Rule

Do not publish RunPod results unless the result template includes the Pod image, GPU type/count, engine version, model revision, launch command, workload hash, config hash, and `missing_metrics`.
