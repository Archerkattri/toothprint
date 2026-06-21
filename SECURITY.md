# Security

ToothPrint ingests **untrusted medical files** (DICOM, intraoral-scan meshes, NIfTI
volumes, exported images) and exposes an HTTP API. Medical parsers are a documented
attack surface — DICOM alone has a long CVE history (malformed length fields,
decompression bombs in the JPEG2000/RLE codecs, a 128-byte preamble that can carry a
polyglot executable), and mesh/zip formats can declare billions of elements or
decompress to enormous sizes. This document is the threat model and the controls.

## Threat model

| Asset | Threat | Control |
|---|---|---|
| Server memory / CPU | Decompression bomb (DICOM codec, gzip NIfTI, 3MF zip), billion-element mesh, giant image | Hard caps before decode: file size (1 GiB), decoded pixels (120 MP), mesh vertices/faces (25 M / 50 M), volume voxels (1.5 G), decompressed bytes (2 GiB). gzip ISIZE + zip directory checked *before* inflation. (`toothprint/io/_limits.py`) |
| File-type confusion | Hostile content with a benign extension (e.g. an executable named `.dcm`) | Detection is by **magic bytes**, not extension (`toothprint/io/detect.py`); the extension is only a fallback for genuinely text/ambiguous formats (ASCII STL, OBJ). |
| Path traversal / SSRF | OBJ `mtllib`, GLB URIs, or other external references pointing at arbitrary local paths/URLs | Materials and textures are **not** loaded (`skip_materials=True`); only geometry is read. No loader follows an external reference. |
| Process integrity | Parser exception on malformed input crashing the request worker | Every loader raises a `ValueError` subclass (`UnsupportedFormat` / `CorruptFile` / `FileTooLarge`); the API maps these to a clean **422**, never a 500 or an uncaught crash. |
| API availability | Unbounded request body, oversized landmark arrays, NaN/Inf poisoning a certificate | Pydantic bounds every list (`≤ MAX_POINTS`, gallery `≤ MAX_GALLERY`) and scalar, rejects non-finite values, requires `0 < α < 1` and `q_lo ≤ q_hi`; a middleware caps JSON bodies (16 MiB) and uploads stream to disk under a 1 GiB cap. |
| Information leak | Validation errors echoing the (possibly hostile) input back | A custom handler returns a generic 422 without the input; security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`) are set on every response. |
| Data at rest | Patient files / PHI committed to the repo | `/data`, `/outputs`, and credentials are git-ignored and never committed; uploads are written to a temp file and deleted in a `finally` block. |

## What is NOT covered (deployment responsibilities)

Authentication, authorization, rate limiting, TLS, and HIPAA/GDPR audit retention are
**deployment concerns**, not library defaults. The `toothprint.clinical` layer adds an
append-only audit trail and site recalibration; a real deployment must add an auth
proxy, per-tenant rate limits, and encrypted storage. ToothPrint is a research
prototype, not a cleared medical device — see `CLINICAL_READINESS.md`.

## Reporting

This is a personal research project; open a GitHub issue for security concerns.
