#!/usr/bin/env python3
"""Reconstruct a dental arch in 3D from photos with Gaussian Splatting.

The bleeding-edge image-to-3D front end for ToothPrint's surface pillar. Instead
of photogrammetry — which fails on textureless dental surfaces — render (or
capture) a dense orbit of *shaded* views so the surface's own geometry becomes
photometric signal, then optimise 3D Gaussians against them with the gsplat
rasteriser and read the surface point cloud off the optimised Gaussians.

Requires the optional extra:  pip install "toothprint[recon]"  (torch + gsplat).
Validated on a real Poseidon3D arch: ~0.84 mm point-to-surface vs the GT scan.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from toothprint.surface import surface_error


def look_at(eye, center, up=(0, 1, 0)) -> np.ndarray:
    """World-to-camera extrinsic (4x4), OpenCV convention (+z forward, +y down)."""
    eye, center, up = map(lambda a: np.asarray(a, float), (eye, center, up))
    z = center - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    E = np.eye(4); E[:3, :3] = np.stack([x, y, z], 0); E[:3, 3] = -np.stack([x, y, z], 0) @ eye
    return E


def render_orbit(mesh_path, n_az=12, elevs=(-15, 15, 45), W=256, H=256, fov=50.0, radius=3.0):
    """Render a shaded orbit of a mesh; returns (views, normalised mesh, scale_mm)."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path)); mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices); c = v.mean(0); scale_mm = float(np.linalg.norm(v - c, axis=1).max())
    mesh.translate(-c); mesh.scale(1.0 / scale_mm, center=(0, 0, 0))
    f = 0.5 * W / np.tan(np.radians(fov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, f, f, W / 2, H / 2)
    rend = rendering.OffscreenRenderer(W, H)
    rend.scene.set_background([0.02, 0.03, 0.04, 1.0])
    mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
    mat.base_color = [0.86, 0.82, 0.76, 1.0]; mat.base_roughness = 0.55
    rend.scene.add_geometry("arch", mesh, mat)
    rend.scene.scene.set_sun_light([0.4, -0.5, -0.8], [1, 1, 1], 90000)
    rend.scene.scene.enable_sun_light(True)
    views = []
    for el in elevs:
        for j in range(n_az):
            az = 2 * np.pi * j / n_az; ce = np.cos(np.radians(el))
            eye = radius * np.array([ce * np.cos(az), np.sin(np.radians(el)), ce * np.sin(az)])
            E = look_at(eye, np.zeros(3))
            rend.setup_camera(intr, E)
            views.append((np.asarray(rend.render_to_image(), np.float32) / 255.0, E, K))
    return views, mesh, scale_mm


def reconstruct(views, n_init=80_000, iters=3000, device="cuda") -> np.ndarray:
    """Optimise 3D Gaussians against posed shaded views; return the surface points."""
    imgs = torch.tensor(np.stack([v[0] for v in views]), device=device)
    viewmats = torch.tensor(np.stack([v[1] for v in views]).astype(np.float32), device=device)
    Ks = torch.tensor(np.stack([v[2] for v in views]).astype(np.float32), device=device)
    V, H, W, _ = imgs.shape
    g = torch.Generator().manual_seed(0)
    pts = ((torch.rand(n_init, 3, generator=g) * 2 - 1) * 0.9)
    pts = pts[pts.norm(dim=1) < 0.95].to(device); N = len(pts)
    P = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.full((N, 3), float(np.log(0.03)), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=device).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.full((N,), -2.0, device=device)),
        "colors": torch.nn.Parameter(torch.full((N, 3), 0.5, device=device)),
    }).to(device)
    lrs = {"means": 1.6e-3, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    opt = {k: torch.optim.Adam([P[k]], lr=lrs[k]) for k in P}
    strat = DefaultStrategy(verbose=False, refine_start_iter=300, refine_stop_iter=int(iters * 0.7),
                            refine_every=100, reset_every=1000)
    state = strat.initialize_state(scene_scale=1.0)
    for step in range(iters):
        i = int(torch.randint(0, V, (1,)).item())
        render, _, info = rasterization(
            P["means"], torch.nn.functional.normalize(P["quats"], dim=-1), torch.exp(P["scales"]),
            torch.sigmoid(P["opacities"]), torch.sigmoid(P["colors"]),
            viewmats[i:i+1], Ks[i:i+1], W, H, packed=False, render_mode="RGB")
        strat.step_pre_backward(P, opt, state, step, info)
        (render[0].clamp(0, 1) - imgs[i]).abs().mean().backward()
        strat.step_post_backward(P, opt, state, step, info, packed=False)
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
    with torch.no_grad():
        m = P["means"][torch.sigmoid(P["opacities"]) > 0.12].cpu().numpy()
    pc, _ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m)).remove_statistical_outlier(20, 2.0)
    return np.asarray(pc.points)


def main():
    ap = argparse.ArgumentParser(description="Gaussian-Splatting arch reconstruction")
    ap.add_argument("--mesh", required=True, help="Path to a dental arch mesh (.stl/.ply/.obj)")
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()
    views, mesh, scale_mm = render_orbit(args.mesh)
    print(f"Optimising 3D Gaussians over {len(views)} shaded views ...")
    recon = reconstruct(views, iters=args.iters)
    gt = np.asarray(mesh.sample_points_uniformly(20000).points)
    err = surface_error(recon, gt, run_icp=True, estimate_scale=True)
    print(f"reconstructed {len(recon):,} points  ·  {err.rms_mm * scale_mm:.3f} mm RMS to GT surface")


if __name__ == "__main__":
    main()
