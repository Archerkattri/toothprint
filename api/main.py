"""ToothPrint API — identification and certificate endpoints, safe medical-file
ingest, and the web app.

Hardened for untrusted input: every list and scalar is bounded and checked finite
(an unbounded landmark array or a NaN would otherwise be a denial-of-service or a
garbage certificate), request bodies are size-capped, uploads stream to disk under a
hard cap and are parsed by the guarded :mod:`toothprint.io` loaders, and basic
security headers are set. Run: ``uvicorn api.main:app --reload`` -> http://localhost:8000
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from toothprint import __version__
from toothprint.identity.constellation import icp_residual
from toothprint.identity.metrics import rank1_match
from toothprint.change.conformal import ConformalCertifier
from toothprint.surface.certificate import certify_surface_change

WEB = Path(__file__).resolve().parents[1] / "web"

MAX_POINTS = 5_000            # a landmark constellation is tens of points; this is generous
MAX_GALLERY = 5_000           # gallery entries per request
MAX_REQUEST_BYTES = 16 * 1024 * 1024     # 16 MiB JSON body cap (constellations are tiny)
MAX_UPLOAD_BYTES = 1024 ** 3             # 1 GiB upload cap (io layer re-checks)

app = FastAPI(title="ToothPrint API", version=__version__,
              description="Certified dental identity, change, and surface — with safe medical-file ingest.")


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # Clean 422 that never echoes the (possibly non-finite / huge) input back.
    return JSONResponse(status_code=422, content={"detail": "invalid or out-of-range request"})


@app.middleware("http")
async def _guard(request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_REQUEST_BYTES and not request.url.path.startswith("/api/inspect"):
        return JSONResponse(status_code=413, content={"detail": "request body too large"})
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


# --- schemas (bounded + finite) --------------------------------------------

def _finite_points(v, dim=2):
    a = np.asarray(v, dtype=float)
    if a.ndim != 2 or a.shape[1] != dim:
        raise ValueError(f"points must be (M, {dim})")
    if not (0 < len(a) <= MAX_POINTS):
        raise ValueError(f"need 1..{MAX_POINTS} points, got {len(a)}")
    if not np.isfinite(a).all():
        raise ValueError("points must be finite (no NaN/Inf)")
    return v


class GalleryEntry(BaseModel):
    label: str = Field(..., max_length=200)
    points: list[list[float]] = Field(..., description="(M, 2) landmark constellation")

    @field_validator("points")
    @classmethod
    def _v(cls, v):
        return _finite_points(v)


class RadiographQuery(BaseModel):
    query: list[list[float]]
    gallery: list[GalleryEntry] = Field(..., max_length=MAX_GALLERY, min_length=1)

    @field_validator("query")
    @classmethod
    def _v(cls, v):
        return _finite_points(v)


class ChangeRequest(BaseModel):
    measured_px: float = Field(..., ge=-1e4, le=1e4)
    q_lo: float = Field(..., ge=0, le=1e4)
    q_hi: float = Field(..., ge=0, le=1e4)
    tau: float = Field(6.0, ge=0, le=1e4, description="Clinically meaningful change threshold (px)")
    alpha: float = Field(0.1, gt=0, lt=1)

    @model_validator(mode="after")
    def _order(self):
        if self.q_lo > self.q_hi:
            raise ValueError("q_lo must be <= q_hi")
        return self


class SurfaceRequest(BaseModel):
    measured_mm: float = Field(..., ge=-1e4, le=1e4)
    q_lo: float = Field(..., ge=0, le=1e4)
    q_hi: float = Field(..., ge=0, le=1e4)
    stable_threshold_mm: float = Field(0.35, ge=0, le=1e4)
    change_threshold_mm: float = Field(0.75, ge=0, le=1e4)
    alpha: float = Field(0.1, gt=0, lt=1)

    @model_validator(mode="after")
    def _order(self):
        if self.q_lo > self.q_hi:
            raise ValueError("q_lo must be <= q_hi")
        return self


# --- endpoints -------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/formats")
def formats() -> dict:
    from toothprint.io import SUPPORTED
    return {"supported": SUPPORTED}


@app.post("/api/identify/radiograph")
def identify_radiograph(req: RadiographQuery) -> dict:
    """Match a query landmark constellation against the gallery (smallest residual)."""
    q = np.asarray(req.query, dtype=float)
    residuals = [icp_residual(q, np.asarray(e.points, dtype=float)) for e in req.gallery]
    best = rank1_match(residuals)
    ranking = sorted(
        ({"label": e.label, "residual_px": round(float(r), 4)}
         for e, r in zip(req.gallery, residuals)),
        key=lambda d: d["residual_px"])
    return {"match": req.gallery[best].label,
            "match_residual_px": round(float(residuals[best]), 4),
            "ranking": ranking}


@app.post("/api/certify/change")
def certify_change_endpoint(req: ChangeRequest) -> dict:
    """Certify a measured radiograph change: changed / stable / uncertain."""
    cert = ConformalCertifier(q_lo=req.q_lo, q_hi=req.q_hi, alpha=req.alpha)
    lo, hi = cert.interval(req.measured_px)
    return {"label": cert.classify(req.measured_px, req.tau),
            "measured_px": req.measured_px, "interval_px": [lo, hi], "tau_px": req.tau}


@app.post("/api/certify/surface")
def certify_surface_endpoint(req: SurfaceRequest) -> dict:
    """Certify a measured 3D surface change: changed / stable / uncertain."""
    cert = ConformalCertifier(q_lo=req.q_lo, q_hi=req.q_hi, alpha=req.alpha)
    out = certify_surface_change(
        req.measured_mm, cert,
        stable_threshold_mm=req.stable_threshold_mm,
        change_threshold_mm=req.change_threshold_mm)
    return {"label": out.label, "measured_mm": out.measured_mm,
            "interval_mm": list(out.interval_mm)}


def _summary(obj, filename: str) -> dict:
    from toothprint.io import Radiograph, Scan, Volume
    base = {"filename": filename, "source_format": obj.source_format}
    if isinstance(obj, Radiograph):
        base.update(kind="radiograph", shape=list(obj.shape),
                    pixel_spacing_mm=obj.pixel_spacing_mm, modality=obj.modality,
                    photometric=obj.photometric, bit_depth=obj.bit_depth)
    elif isinstance(obj, Scan):
        v = obj.vertices
        base.update(kind="scan", n_vertices=obj.n_vertices, n_faces=obj.n_faces,
                    bbox_mm=[float(v.min(0).min()), float(v.max(0).max())],
                    extent_mm=[round(float(x), 2) for x in (v.max(0) - v.min(0))])
    else:  # Volume
        base.update(kind="volume", shape=list(obj.shape), spacing_mm=list(obj.spacing_mm))
    return base


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)) -> dict:
    """Safely parse any uploaded medical file (DICOM/STL/PLY/OBJ/3MF/NIfTI/PNG/...)
    and return a normalized summary. Streams to disk under a hard cap; the guarded
    loaders reject anything oversize, corrupt, or hostile (422)."""
    suffix = "".join(Path(file.filename or "upload").suffixes[-2:])   # keep .nii.gz, not just .gz
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="upload too large")
                out.write(chunk)
        from toothprint import io as tio
        try:
            obj = tio.load(tmp)
        except tio.IOError_ as e:
            raise HTTPException(status_code=422, detail=f"cannot load file: {e}")
        return _summary(obj, file.filename or "upload")
    finally:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: no cover - cleanup best-effort
            pass


# --- web app ---------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
