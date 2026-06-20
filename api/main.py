"""ToothPrint API — identification and certificate endpoints + the web app.

Run: ``uvicorn api.main:app --reload`` then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from toothprint import __version__
from toothprint.change.conformal import ConformalCertifier
from toothprint.identity.constellation import icp_residual
from toothprint.identity.metrics import rank1_match
from toothprint.surface.certificate import certify_surface_change

WEB = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="ToothPrint API", version=__version__,
              description="Certified dental identity, change, and surface.")


# --- schemas ---------------------------------------------------------------

class GalleryEntry(BaseModel):
    label: str
    points: list[list[float]] = Field(..., description="(M, 2) landmark constellation")


class RadiographQuery(BaseModel):
    query: list[list[float]]
    gallery: list[GalleryEntry]


class ChangeRequest(BaseModel):
    measured_px: float
    q_lo: float = Field(..., ge=0)
    q_hi: float = Field(..., ge=0)
    tau: float = Field(6.0, description="Clinically meaningful change threshold (px)")
    alpha: float = 0.1


class SurfaceRequest(BaseModel):
    measured_mm: float
    q_lo: float = Field(..., ge=0)
    q_hi: float = Field(..., ge=0)
    stable_threshold_mm: float = 0.35
    change_threshold_mm: float = 0.75
    alpha: float = 0.1


# --- endpoints -------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


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


# --- web app ---------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
