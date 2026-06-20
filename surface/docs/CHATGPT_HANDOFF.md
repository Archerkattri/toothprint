# ChatGPT Handoff - DentalMapCert

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop DentalMapCert.

Read README.md, AGENTS.md, docs/DATASETS.md, docs/GPU_BASELINE_CONTRACT.md,
src/dentalmapcert/cli.py, and the tests before making changes.

Goal: build a benchmark/method for smartphone oral visible-surface mapping with
coverage certificates, uncertainty intervals, recapture guidance, and future
fusion with DentalChangeCert.

Rules:
- Do not commit raw data, patient captures, meshes, tokens, .env,
  LOCAL_SECRETS.md, or generated outputs.
- Treat this as the GPU development path; expose CUDA device 0 by default.
- This is not hidden/root/subgingival reconstruction. Keep claims to visible
  tooth/gingiva surfaces.
- Add tests before adapters and keep every dataset parser fixture-sized first.

First useful tasks:
1. run tests with CUDA visible;
2. add dataset manifest validation;
3. create a 5-case fixture protocol;
4. add one CUDA reconstruction baseline adapter;
5. evaluate coverage and recapture guidance before visual quality.
```
