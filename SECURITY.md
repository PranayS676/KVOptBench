# Security Policy

## Supported Versions

KVOptBench is currently pre-1.0. Security fixes will target the latest main branch until versioned releases exist.

## Reporting a Vulnerability

If you find a vulnerability, please open a private security advisory if GitHub security advisories are enabled. If not, contact the maintainer directly.

Do not open a public issue for vulnerabilities involving secrets, token leakage, private endpoints, or private benchmark data.

## Sensitive Data Policy

Do not commit:

- API keys
- Hugging Face tokens
- RunPod tokens
- private endpoint URLs
- private benchmark data
- private model outputs
- `.env` files

Generated reports and results should redact credentials.

## Benchmark Data Warning

KVOptBench can run prompts against private models and endpoints. Users are responsible for ensuring that prompts, outputs, logs, and results do not contain confidential data before publishing.
