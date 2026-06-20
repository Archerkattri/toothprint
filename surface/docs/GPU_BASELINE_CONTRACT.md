# GPU Baseline Contract

This package is the project-owned DentalMapCert scaffold. It is safe to edit and
inspect while GPU reconstruction work is being added because the current code
contains only dataclasses, decision rules, manifest writers, and fixture tests.

It should next:

- expose CUDA device 0 for baseline adapters;
- add torch/CUDA, OpenCV, COLMAP, DUSt3R, nerfstudio, gsplat, or Meshroom one
  adapter at a time;
- extract only explicit fixture-sized subsets before full dataset parsing;
- train or run reconstruction models only after manifests and tests pass.

The first compute step should be a minimal, explicit subset extraction and
manifest validation pass. The first model step should be one CUDA baseline
adapter with fixture tests.
