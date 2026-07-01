# External datasets for the remaining pillars (#4 multimodal, #7 longitudinal)

The two pillars that are **data-blocked, not code-blocked** need datasets that don't ship
with this repo. They were located by literature search (June 2026). Medical dental data is
mostly privacy-gated, so most require a signed agreement you must complete yourself.

## #4 — multimodal fusion (same-subject 3D scan + 2D radiograph)

**3D multimodal dental dataset based on CBCT and oral scan** — Figshare, **CC BY 4.0, open,
direct download** (the one that needs no agreement). 300 patients, paired CBCT (3D volume →
derive a panoramic/periapical 2D via DRR) + intraoral **oral scan** (3D surface, same modality
as our Poseidon3D pipeline). 15 zips, ~59 GB total.

```bash
# all 15 parts (~59 GB) — or just teeth_data1.zip for a working sample
for f in $(curl -s https://api.figshare.com/v2/articles/26965903 \
            | python3 -c "import json,sys;[print(x['download_url'],x['name']) for x in json.load(sys.stdin)['files']]"); do
  url=${f%% *}; name=${f##* }
  [ "$name" != "${name%.zip}" ] && curl -L "$url" -o "data/cbct_ios/$name"
done
```
Fusion path: IOS mesh → existing identity embedding/GICP; CBCT → DRR panoramic → existing
2D landmark constellation; **score-level fuse the two distances under one conformal
calibration** (the same conformal machinery already in `analyze_identity.py`).

- DOI 10.6084/m9.figshare.26965903 · `https://figshare.com/articles/dataset/26965903`

Alternatives (richer, but gated): **MMDental** (Nature Sci Data 2025, CBCT + panoramic +
intraoral, 389–660 patients) and **FDTooth** (intraoral photos + CBCT).

## #7 — real longitudinal / cross-session (the binding constraint)

Both best fits are **access-restricted** — they need *your* registration + a data-use
agreement; they cannot be auto-downloaded.

1. **Multimodal Dental Dataset (PhysioNet v1.1.0)** — *credentialed*. 329 CBCT from 169
   patients + 16,203 periapical + 8 panoramic, **with visit timestamps for multiple visits
   per patient** → serves BOTH #7 (longitudinal) and #4 (multimodal). Register on PhysioNet,
   sign the *PhysioNet Restricted Health Data License 1.5.0*, then:
   ```bash
   wget -r -N -c -np --user YOUR_USER --ask-password \
     https://physionet.org/files/multimodal-dental-dataset/1.1.0/
   ```
   `https://physionet.org/content/multimodal-dental-dataset/`

2. **3D pre/post-orthodontic dental models (Zenodo 11392406)** — *restricted, request form*.
   1,060 pre/post **3D intraoral** model pairs from 435 patients → real longitudinal change
   on our exact modality. Request access on the Zenodo record; download once granted.
   `https://zenodo.org/records/11392406`
   → **concrete DUA application checklist + gate status in `evaluation/DATA_GATE.md`** (this is the
   primary #7 lead; nothing published in 2025–26 closes the gate).

Once either is in `data/`, the existing identity (GICP + embedding) and surface/change
certificates run on **real re-scans / real timepoints** instead of synthetic perturbations —
turning every "optimistic ceiling" headline into a real-world number.
