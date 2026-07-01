# Real-data gate tracker (#7 longitudinal / cross-session)

The binding constraint on ToothPrint remains **real longitudinal intraoral data**: genuine
pre/post re-scans of the same patient on our exact modality, so every "optimistic ceiling"
identity/change/surface number becomes a real-world number instead of a synthetic perturbation.

**2026-07-01 status:** nothing published in 2025–26 closes this gate. The only real longitudinal
lead is still **Zenodo 11392406** (DUA-gated); PhysioNet Multimodal Dental v1.1.0 (credentialed) is
the second. Both require *your* application — neither auto-downloads. See also
`evaluation/EXTERNAL_DATA.md` for the multimodal (#4) datasets.

---

## Lead #1 (primary) — Zenodo 11392406: 3D pre/post-orthodontic dental models

- **What:** 1,060 pre/post **3D intraoral (IOS)** model pairs from **435 patients** — real
  longitudinal change on our exact modality (the closest public match to what #7 needs).
- **Access:** Zenodo **Restricted** record — access is granted per-request by the record owner
  under a Data Use Agreement; there is no direct download URL.
- **Record:** https://zenodo.org/records/11392406

### DUA application checklist

1. **Create/confirm a Zenodo account** (ORCID-linked) using an institutional identity
   (`kattri@snu.ac.kr`, SNU) — institutional affiliation materially helps restricted approvals.
2. **Open the record** and read its *Access conditions* / *Terms* block. Record verbatim:
   - the exact **contact / record owner** named on the page (the person or lab who approves),
   - any **required use statement** or license they cite (e.g. research-only, no redistribution).
   *(Do this live — the approver and terms are set by the depositor and can change; do not assume.)*
3. **Click "Request access"** on the record and attach a short request letter (below). If the
   record instead lists a direct **contact email**, send the same letter there.
4. **Request letter must contain:**
   - Requester name, role, institution, and **institutional email** (SNU / `kattri@snu.ac.kr`).
   - **Purpose:** non-commercial academic research on *certified* longitudinal dental change /
     identity from intraoral scans (ToothPrint; PolyForm Noncommercial).
   - **Specific use:** validate conformal change/surface certificates on real pre/post IOS pairs;
     report aggregate metrics only.
   - **Data handling:** stored on encrypted institutional storage, not redistributed, not used to
     re-identify patients, deleted on project end.
   - Explicit agreement to the record's stated **terms/DUA** and any citation requirement.
   - Offer to sign a formal DUA and, if asked, route through the **SNU IRB / institutional
     signatory**.
5. **If the DUA needs an institutional signature**, start the SNU IRB / research-office signature
   process in parallel — this is usually the long pole.
6. **On approval:** download into `data/` and point `TP_POSEIDON3D` (or a new
   `TP_LONGITUDINAL` path in `evaluation/scripts/paths.py`) at the pre/post pairs; the existing
   identity (GICP + embedding) and surface/change certificate pipelines run unchanged on real
   timepoints.
7. **Track:** log request date, approver, and outcome here so the gate status stays current.

**Application log:**

| date | action | approver / contact | status |
| --- | --- | --- | --- |
| _pending_ | request access on Zenodo 11392406 | _read from record_ | not yet submitted |

---

## Lead #2 (secondary) — PhysioNet Multimodal Dental Dataset v1.1.0

- **What:** 329 CBCT from 169 patients + 16,203 periapical + 8 panoramic, **with visit timestamps
  for multiple visits per patient** → serves BOTH #7 (longitudinal) and #4 (multimodal). Not
  intraoral-surface, so it is a weaker fit for our IOS pipeline than Zenodo 11392406, but it is a
  credentialed (not bespoke-DUA) path.
- **Access:** PhysioNet **credentialed** access.
  1. Register on PhysioNet and complete **credentialing** (CITI "Data or Specimens Only Research"
     training + institutional reference).
  2. Sign the **PhysioNet Restricted Health Data License 1.5.0** on the project page.
  3. Download:
     ```bash
     wget -r -N -c -np --user YOUR_USER --ask-password \
       https://physionet.org/files/multimodal-dental-dataset/1.1.0/
     ```
- **Record:** https://physionet.org/content/multimodal-dental-dataset/

---

## What does NOT close the gate (2025–26 scan)

- The dental-recon systems catalogued in `docs/RECON_UPGRADES_2026.md` (DentalSplat, Dental3R,
  DentalGS, TeethDreamer, DentalMonitoring) are **reconstruction**, not longitudinal identity data.
- The multimodal open dataset in `EXTERNAL_DATA.md` (Figshare CBCT+IOS, 300 patients) is
  **cross-sectional** — same-visit multimodal, no second timepoint — so it feeds #4, not #7.
- No new public **longitudinal intraoral** corpus appeared in 2025–26. Zenodo 11392406 remains the
  only real lead; the gate stays **DUA-blocked, not code-blocked**.
