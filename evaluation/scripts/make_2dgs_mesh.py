#!/usr/bin/env python3
"""2D Gaussian Splatting reconstruction — flat oriented surfels that hug the surface,
with normal-consistency + distortion regularisation, TSDF-fused into a watertight mesh.

2DGS (Huang et al. 2024) replaces 3D ellipsoids with 2D disks aligned to the surface,
so the rendered depth and normals are far more surface-accurate than 3DGS — better
geometry for meshing. Compares median/chamfer vs the 3DGS+TSDF baseline on the same arch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_gsplat_recon import render_orbit


def train_2dgs(views, n_init=100_000, iters=5000, device="cuda"):
    torch.manual_seed(0)
    imgs = torch.tensor(np.stack([v[0] for v in views]), device=device)
    viewmats = torch.tensor(np.stack([v[1] for v in views]), device=device)
    Ks = torch.tensor(np.stack([v[2] for v in views]), device=device)
    V, H, W, _ = imgs.shape

    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(n_init, 3, generator=g) * 2 - 1) * 0.9
    pts = pts[pts.norm(dim=1) < 0.95].to(device)
    N = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.full((N, 3), np.log(0.02), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=device).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.full((N,), -2.0, device=device)),
        "colors": torch.nn.Parameter(torch.full((N, 3), 0.5, device=device)),
    }).to(device)
    lrs = {"means": 1.6e-3, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    opt = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}
    strat = DefaultStrategy(verbose=False, refine_start_iter=500, refine_stop_iter=int(iters * 0.8),
                            refine_every=100, reset_every=1000)
    state = strat.initialize_state(scene_scale=1.0)

    for step in range(iters):
        i = torch.randint(0, V, (1,)).item()
        out = rasterization_2dgs(
            params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"])[None], viewmats[i:i + 1], Ks[i:i + 1], W, H,
            render_mode="RGB+ED", distloss=True, depth_mode="median", packed=False)
        render, _, normals, surf_normals, distort, _, info = out
        strat.step_pre_backward(params, opt, state, step, info)
        rgb = render[0, ..., :3].clamp(0, 1)
        l_rgb = (rgb - imgs[i]).abs().mean()
        l_normal = (1 - (normals[0] * surf_normals).sum(-1)).mean()       # surfels lie on the surface
        lam_d = 0.1 if step > iters // 2 else 0.0                          # distortion, ramped in
        loss = l_rgb + 0.05 * l_normal + lam_d * distort.mean()
        loss.backward()
        strat.step_post_backward(params, opt, state, step, info, packed=False)
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            print(f"  step {step:4d}  rgb {l_rgb.item():.4f}  normal {l_normal.item():.3f}  "
                  f"gaussians {params['means'].shape[0]:,}", flush=True)
    return params, viewmats, Ks, int(H), int(W)


def extract_mesh_2dgs(params, viewmats, Ks, H, W, *, voxel=0.004, trunc=0.02, alpha_thr=0.5):
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    means = params["means"].detach()
    quats = torch.nn.functional.normalize(params["quats"].detach(), dim=-1)
    scales = torch.exp(params["scales"].detach())
    opac = torch.sigmoid(params["opacities"].detach())
    cols = torch.sigmoid(params["colors"].detach())
    with torch.no_grad():
        for i in range(viewmats.shape[0]):
            out = rasterization_2dgs(means, quats, scales, opac, cols[None], viewmats[i:i + 1],
                                     Ks[i:i + 1], W, H, render_mode="RGB+ED", depth_mode="median", packed=False)
            render, alpha, median = out[0], out[1], out[5]
            rgb = render[0, ..., :3].clamp(0, 1).cpu().numpy()
            depth = median[0, ..., 0].cpu().numpy().astype(np.float32)   # median = first-surface depth (surfels)
            a = alpha[0, ..., 0].cpu().numpy()
            depth[a < alpha_thr] = 0.0
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray((rgb * 255).astype(np.uint8))),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False)
            K = Ks[i].cpu().numpy()
            intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            vol.integrate(rgbd, intr, viewmats[i].cpu().numpy())
    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    idx, counts, _ = mesh.cluster_connected_triangles()
    idx = np.asarray(idx); counts = np.asarray(counts)
    if len(counts):
        mesh.remove_triangles_by_mask(idx != int(counts.argmax())); mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=8); mesh.compute_vertex_normals()
    return mesh


def accuracy(mesh, gt_mesh, scale_mm):
    rec = mesh.sample_points_uniformly(60000); gt = gt_mesh.sample_points_uniformly(60000)
    reg = o3d.pipelines.registration.registration_icp(
        rec, gt, 0.05, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
    mesh.transform(reg.transformation); rec.transform(reg.transformation)
    d_rec = np.asarray(rec.compute_point_cloud_distance(gt)) * scale_mm
    d_gt = np.asarray(gt.compute_point_cloud_distance(rec)) * scale_mm
    return float(d_rec.mean()), float(np.median(d_rec)), float((d_rec.mean() + d_gt.mean()) / 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/poseidon3d/extracted/data")
    p.add_argument("--idx", type=int, default=2)
    p.add_argument("--iters", type=int, default=5000)
    args = p.parse_args()
    mesh_path = sorted(Path(args.data).glob("*/*.stl"))[args.idx]
    print(f"2DGS reconstruction of {mesh_path.parent.name} ...", flush=True)
    views, gt_mesh, scale_mm = render_orbit(mesh_path, n_az=22, elevs=(-25, -5, 20, 45), W=320, H=320)
    params, viewmats, Ks, H, W = train_2dgs(views, iters=args.iters)
    mesh = extract_mesh_2dgs(params, viewmats, Ks, H, W)
    mean, med, chamfer = accuracy(mesh, gt_mesh, scale_mm)
    print(f"\n2DGS MESH: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris", flush=True)
    print(f"  accuracy: mean {mean:.3f} mm | median {med:.3f} mm | chamfer {chamfer:.3f} mm "
          f"(3DGS baseline: median 0.42 / chamfer 0.54)", flush=True)
    # `mesh` is now ICP-aligned to GT (accuracy() transforms it); save the turntable inputs
    # in the make_recon_gif format, in real mm, overwriting the 3DGS outputs.
    base = Path("outputs/gsplat_mesh"); base.mkdir(parents=True, exist_ok=True)
    gt_pcd = gt_mesh.sample_points_uniformly(60000)
    vd = np.asarray(o3d.geometry.PointCloud(mesh.vertices).compute_point_cloud_distance(gt_pcd)) * scale_mm
    m_mm = o3d.geometry.TriangleMesh(mesh); m_mm.scale(scale_mm, center=(0, 0, 0)); m_mm.compute_vertex_normals()
    gt_mm = o3d.geometry.TriangleMesh(gt_mesh); gt_mm.scale(scale_mm, center=(0, 0, 0)); gt_mm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(base / "arch_mesh.ply"), m_mm)
    o3d.io.write_triangle_mesh(str(base / "gt_mesh.ply"), gt_mm)
    np.save(base / "vertex_error_mm.npy", vd)
    (base / "accuracy.txt").write_text(
        f"verts={len(mesh.vertices)} tris={len(mesh.triangles)} mean={mean:.3f} median={med:.3f} "
        f"chamfer={chamfer:.3f} scale_mm={scale_mm:.2f} method=2dgs\n")
    print(f"  saved {base}/arch_mesh.ply + gt_mesh.ply + vertex_error_mm.npy (real mm)", flush=True)


if __name__ == "__main__":
    main()
