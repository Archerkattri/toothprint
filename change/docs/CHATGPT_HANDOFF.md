# ChatGPT Handoff - DentalChangeCert

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop DentalChangeCert.

Read README.md, AGENTS.md, docs/DATASETS.md, docs/dataset-risk-note.md, dcc/cli.py,
and the tests before making changes.

Goal: turn this scaffold into a real benchmark/method for certified intraoral
dental radiograph change detection under acquisition uncertainty.

Rules:
- Do not commit raw data, extracted data, tokens, .env, LOCAL_SECRETS.md, or
  generated outputs.
- Treat this as the GPU development path; expose CUDA device 0 by default.
- Add tests for every parser, manifest writer, certificate rule, and metric.
- Keep claims conservative: this is calibrated longitudinal change/abstention,
  not clinical diagnosis.

First useful tasks:
1. run the existing tests with CUDA visible;
2. add dataset extraction manifests for one dataset only;
3. add a tiny fixture-based parser test;
4. add checksum and split-leakage checks;
5. wire CUDA model outputs into the benchmark.
```
