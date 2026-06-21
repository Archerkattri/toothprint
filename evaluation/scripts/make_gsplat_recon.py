#!/usr/bin/env python3
"""3D Gaussian Splatting reconstruction of a real dental arch from shaded photos.

Bleeding-edge image-to-3D: instead of photogrammetry, render a dense orbit of
*shaded* views (so the surface's own geometry becomes photometric signal — the
textureless-surface problem that sinks DUSt3R), then optimise 3D Gaussians
against them with the gsplat rasteriser and extract the surface point cloud.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy


def look_at(eye, center, up=(0, 1, 0)):
    """World-to-camera extrinsic (4x4) in OpenCV convention (+z forward, +y down)."""
    eye = np.asarray(eye, float); center = np.asarray(center, float); up = np.asarray(up, float)
    z = center - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)          # world-to-camera rotation
    t = -R @ eye
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
    return E


def render_orbit(mesh_path, n_az=12, elevs=(-15, 15, 45), W=256, H=256, fov=50.0, radius=3.0):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path)); mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices); c = v.mean(0); scale_mm = np.linalg.norm(v - c, axis=1).max()
    mesh.translate(-c); mesh.scale(1.0 / scale_mm, center=(0, 0, 0))   # normalise to unit sphere

    f = 0.5 * W / np.tan(np.radians(fov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, f, f, W / 2, H / 2)

    rend = rendering.OffscreenRenderer(W, H)
    rend.scene.set_background([0.02, 0.03, 0.04, 1.0])
    mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
    mat.base_color = [0.86, 0.82, 0.76, 1.0]; mat.base_roughness = 0.55; mat.base_metallic = 0.0
    rend.scene.add_geometry("arch", mesh, mat)
    rend.scene.scene.set_sun_light([0.4, -0.5, -0.8], [1, 1, 1], 90000)
    rend.scene.scene.enable_sun_light(True)

    views = []
    for el in elevs:
        for j in range(n_az):
            az = 2 * np.pi * j / n_az
            ce = np.cos(np.radians(el))
            eye = radius * np.array([ce * np.cos(az), np.sin(np.radians(el)), ce * np.sin(az)])
            E = look_at(eye, np.zeros(3))
            rend.setup_camera(intr, E)
            img = np.asarray(rend.render_to_image(), dtype=np.float32) / 255.0
            views.append((img, E.astype(np.float32), K.astype(np.float32)))
    return views, mesh, float(scale_mm)


def train(views, n_init=80_000, iters=3000, device="cuda"):
    torch.manual_seed(0)
    imgs = torch.tensor(np.stack([v[0] for v in views]), device=device)            # (V,H,W,3)
    viewmats = torch.tensor(np.stack([v[1] for v in views]), device=device)        # (V,4,4)
    Ks = torch.tensor(np.stack([v[2] for v in views]), device=device)              # (V,3,3)
    V, H, W, _ = imgs.shape

    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(n_init, 3, generator=g) * 2 - 1) * 0.9
    keep = pts.norm(dim=1) < 0.95
    pts = pts[keep].to(device)
    N = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.full((N, 3), np.log(0.03), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=device).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.full((N,), -2.0, device=device)),     # sigmoid~0.12
        "colors": torch.nn.Parameter(torch.full((N, 3), 0.5, device=device)),
    }).to(device)
    lrs = {"means": 1.6e-3, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    opt = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}

    strat = DefaultStrategy(verbose=False, refine_start_iter=300, refine_stop_iter=2200,
                            refine_every=100, reset_every=1000)
    state = strat.initialize_state(scene_scale=1.0)

    for step in range(iters):
        i = torch.randint(0, V, (1,)).item()
        render, alpha, info = rasterization(
            params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"]), viewmats[i:i+1], Ks[i:i+1], W, H,
            packed=False, render_mode="RGB")
        strat.step_pre_backward(params, opt, state, step, info)
        pred = render[0].clamp(0, 1)
        loss = (pred - imgs[i]).abs().mean()
        loss.backward()
        strat.step_post_backward(params, opt, state, step, info, packed=False)
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  gaussians {params['means'].shape[0]:,}")

    with torch.no_grad():
        op = torch.sigmoid(params["opacities"])
        m = params["means"][op > 0.12].cpu().numpy()
    # drop floaters
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pc.points), float(loss.item())


def occlusal(pts):
    c = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:2].T, c @ Vt[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/poseidon3d/extracted/data")
    p.add_argument("--out", default="docs/dmc_gsplat_recon.png")
    p.add_argument("--iters", type=int, default=3000)
    args = p.parse_args()

    mesh_path = sorted(Path(args.data).glob("*/*.stl"))[2]
    print(f"Rendering shaded orbit of {mesh_path.parent.name} ...")
    views, mesh, scale_mm = render_orbit(mesh_path)
    print(f"Optimising 3D Gaussians over {len(views)} views ...")
    recon, loss = train(views, iters=args.iters)
    gt = np.asarray(mesh.sample_points_uniformly(20000).points)
    # accuracy vs the GT scan (scale-aware align; the unit-normalised mesh -> *scale_mm)
    from dentalmapcert.surface_error import surface_error_mm
    err = surface_error_mm(recon, gt, run_icp=True, estimate_scale=True)
    acc = err.point_to_surface_mm * scale_mm
    print(f"reconstructed {len(recon):,} surface gaussians  (final L1 {loss:.4f}, "
          f"point-to-surface {acc:.3f} mm)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(views[len(views) // 2][0]); axes[0].axis("off")
    axes[0].set_title("one of 36 shaded input views", fontsize=11)
    for ax, (pts, title, cmap) in zip(axes[1:], [
            (gt, "ground-truth scan", "bone"),
            (recon, f"3D Gaussian Splatting  ·  {len(recon):,} pts  ·  {acc:.2f} mm to surface", "copper")]):
        xy, d = occlusal(pts)
        ax.scatter(xy[:, 0], xy[:, 1], c=d, cmap=cmap, s=2, alpha=0.75)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)
    fig.suptitle("Gaussian-Splatting reconstruction of a real dental arch from shaded photos "
                 "(no scanner, no photogrammetry)", fontsize=13, y=0.99)
    fig.patch.set_facecolor("white")
    out = Path(args.out); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
