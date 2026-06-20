<div align="center">

# ToothPrint

**Certified dental-imaging intelligence — recognise a person by their teeth,
and certify what changed.**

`identity` · `change` · `surface` — three reads of one durable signal, each
returning a certificate instead of a guess.

</div>

---

A face can be lost; the teeth remain. ToothPrint reads the dentition three ways
and attaches a statistical guarantee to every verdict:

- **Who** is this? — dental biometric identification from a 3D scan or a 2D radiograph.
- **Whether** it changed — certified longitudinal bone-level change detection.
- **What** its surface is — certified 3D surface-change mapping.

The certification core depends only on `numpy`, `scipy`, `opencv`, and `open3d`.
Learned front-ends (tooth detection, Gaussian-splatting reconstruction) are
pluggable and optional, so the guarantees run without a GPU.

## Results (measured on real public data)

| Capability | What it answers | Result |
|---|---|---|
| **Identity — 3D scans** | Who is this arch? | **Rank-1 1.000**, genuine 0.10 mm vs impostor 0.55 mm, d′ = 50 |
| **Identity — 2D radiographs** | Who is this X-ray? | **Rank-1 1.000**, genuine 4 px vs impostor 102 px, d′ = 4.2 |
| **Change certificate** | Did the bone level change? | recall **0.97 @ 0% false-progression** (0.72 end-to-end) |
| **Surface certificate** | Did the 3D surface change? | stable ≤0.2 mm / change ≥1.0 mm at **0% false-change** |

![Genuine vs impostor — both modalities](docs/identification_separation.png)

Every certificate is conformal: it fires only when the interval around the
measurement lies entirely past the threshold, so the false-alarm rate is bounded
by α in finite samples — no distributional assumptions.

## Evidence on real data

These are the system's own outputs on the public Poseidon3D and DenPAR datasets.

**Identification** — a query arch registers tightly onto its own enrolment
(0.10 mm) but cannot match a stranger's (0.54 mm). Recognition by teeth:

![Genuine vs impostor registration overlay](docs/identity_registration.png)

**Photos to geometry** — no scanner? **3D Gaussian Splatting** rebuilds a real
arch from shaded photos to within **0.84 mm** of the ground-truth scan (on an
8 GB GPU). Shading turns the textureless surface into the photometric signal that
photogrammetry can't find:

![Gaussian Splatting reconstruction](docs/gaussian_splatting_recon.png)

**Tooth localization** — ViTPose coarsely localises CEJ and bone crest on real
radiographs (GT green, prediction red). Pinpoint accuracy isn't required: the
change certificate measures the shift by sub-pixel registration, not by these
points:

![Landmark overlays on real radiographs](docs/landmark_overlays.png)

## How it works

One stack, three certificates:

```
scan / radiograph ─▶ detect ─▶ register ─▶ certify
                     teeth +    2D/3D ICP ·  conformal interval ─▶ identity
                     landmarks  FPFH ·        ─▶ change
                     or cloud   template      ─▶ surface
                                matching
```

- **Identity (3D):** FPFH descriptors → coarse RANSAC → fine ICP → the gallery
  arch with the smallest registration RMSE is the person.
- **Identity (2D):** the per-tooth landmark constellation, scale-normalised so
  magnification cancels, aligned by rigid ICP.
- **Change:** the bone-level shift is measured *differentially* — sub-pixel
  template matching of the margin between timepoints, referenced to a stationary
  crown so acquisition motion cancels — then certified conformally.
- **Surface:** scale-aware ICP + screened-Poisson refinement give a surface
  error that a conformal certificate decides against the reconstruction's own noise.

## Use it

```python
import numpy as np
from toothprint.identity import enroll, identify_scan, identification_metrics
from toothprint.change import ConformalCertifier, certify_change, bone_vector
from toothprint.surface import surface_error, certify_surface_change

# Identify a person from a 3D arch against a gallery
gallery = [enroll(points, voxel_size=0.5) for points in enrolled_scans]
rmse_row = identify_scan(query_points, gallery, voxel_size=0.5)
person = labels[int(np.argmin(rmse_row))]

# Certify a surface change against calibrated reconstruction noise
certifier = ConformalCertifier.fit(measured_stable, true_stable, alpha=0.1)
verdict = certify_surface_change(measured_mm=1.2, certifier=certifier)   # -> "changed"
```

## Run the app

A web console for the three certificates, plus a JSON API.

```bash
pip install -e ".[api]"
uvicorn api.main:app --reload      # http://localhost:8000
```

| Endpoint | Does |
|---|---|
| `POST /api/identify/radiograph` | Match a landmark constellation against a gallery |
| `POST /api/certify/change` | Certify a radiograph bone-level change |
| `POST /api/certify/surface` | Certify a 3D surface change |

The frontend (`web/`) is a static, dependency-free single page — open it directly
or let the API serve it.

## Layout

```
toothprint/
  toothprint/        the library — identity · change · surface (100% covered)
  api/               FastAPI service
  web/               the console (HTML/CSS/JS, no build step)
  docs/              result figures
  tests/             58 tests, 100% coverage
```

## Test

```bash
pip install -e ".[dev]"
pytest --cov=toothprint --cov=api      # 100%
```

## Provenance & limits

Numbers are measured on the public Poseidon3D (intraoral scans) and DenPAR
(radiographs) datasets; reproduction scripts and the underlying research live in
the companion repositories. Identification is validated closed-set on real
anatomy; end-to-end change *sensitivity* is bounded today by tooth-localisation
precision, not by the measurement, and is reported honestly as such. Datasets and
model checkpoints are never committed.

## License

MIT.
