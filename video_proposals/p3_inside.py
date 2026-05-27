"""Proposal 3: Inside the Panorama — first-person immersive.
We stand inside the sphere, look around, depth materialises, then we pull back to the point cloud."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     write_video, fresh_frames_dir, save_frame,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT)

W, H, FPS = 1280, 720, 24
HERO = "eth_campus_plaza"


def erp_perspective_view(erp: np.ndarray, depth: np.ndarray | None,
                          yaw: float, pitch: float, fov_deg: float = 78.0,
                          out_w: int = W, out_h: int = H) -> tuple[np.ndarray, np.ndarray | None]:
    """Render a perspective view from the centre of the panorama looking at (yaw, pitch).
    Optionally return the per-pixel depth (sampled the same way)."""
    H_e, W_e = erp.shape[:2]
    f = out_h * 0.5 / np.tan(np.deg2rad(fov_deg) * 0.5)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    x = (xx - out_w * 0.5) / f
    y = -(yy - out_h * 0.5) / f
    z = np.ones_like(x)
    # rotate by yaw (Y) then pitch (X)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp_, sp_ = np.cos(pitch), np.sin(pitch)
    # yaw: x' = c*x + s*z, z' = -s*x + c*z
    x1 = cy_ * x + sy_ * z
    y1 = y
    z1 = -sy_ * x + cy_ * z
    # pitch: y' = cp*y - sp*z, z' = sp*y + cp*z
    x2 = x1
    y2 = cp_ * y1 - sp_ * z1
    z2 = sp_ * y1 + cp_ * z1
    norm = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    dx, dy, dz = x2 / norm, y2 / norm, z2 / norm
    lon = np.arctan2(dx, dz)         # [-pi, pi]
    lat = np.arcsin(dy.clip(-1, 1))  # [-pi/2, pi/2]
    u = ((lon + np.pi) / (2 * np.pi) * (W_e - 1)).astype(np.int32) % W_e
    v = ((0.5 - lat / np.pi) * (H_e - 1)).clip(0, H_e - 1).astype(np.int32)
    img = erp[v, u]
    if depth is not None:
        d = depth[v, u]
        return img, d
    return img, None


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p3_inside")

    # ── scene 1: hands-on opener — flat ERP scrolls (4s)
    intro_frames = 48
    erp_small = resize(s.rgb, w=int(W * 0.85))
    for i in range(intro_frames):
        t = i / max(intro_frames - 1, 1)
        f = blank(W, H)
        shift = int(-erp_small.shape[1] * 0.20 * t)
        erp_h = erp_small.shape[0]
        f = paste(f, np.roll(erp_small, shift, axis=1),
                  (W - erp_small.shape[1]) // 2, (H - erp_h) // 2,
                  alpha=min(1.0, t * 3))
        a = min(1.0, t * 3)
        f = draw_text(f, "input · 360° panorama", (W // 2, H - 50),
                      size=22, color=TEXT_DIM, anchor="mm", alpha=a)
        save_frame(f, frames_dir, i)
    cursor = intro_frames

    # ── scene 2: step into the sphere — 360° look-around (5s)
    look_frames = 120
    rgb = s.rgb
    for i in range(look_frames):
        t = i / max(look_frames - 1, 1)
        yaw = -np.pi * 0.8 + t * 2 * np.pi * 1.0   # ~360° sweep
        pitch = 0.05 * np.sin(t * np.pi * 1.5)
        view, _ = erp_perspective_view(rgb, None, yaw=yaw, pitch=pitch,
                                         fov_deg=85, out_w=W, out_h=H)
        # entrance: zoom-in feel for first 8 frames
        z = max(0.0, 1.0 - i / 8) if i < 8 else 0.0
        if z > 0:
            scale = 1.0 + z * 0.35
            cw = int(W * scale); ch = int(H * scale)
            big = cv2.resize(view, (cw, ch))
            view = big[(ch - H) // 2:(ch - H) // 2 + H, (cw - W) // 2:(cw - W) // 2 + W]
        view = draw_text(view, "step inside the panorama", (W // 2, H - 50),
                          size=22, color=TEXT_LIGHT, anchor="mm", alpha=0.85)
        save_frame(view, frames_dir, cursor + i)
    cursor += look_frames

    # ── scene 3: depth fog materialises (4s)
    depth_norm = (s.depth_raw - s.depth_raw.min()) / max(1e-3, np.percentile(s.depth_raw, 99) - s.depth_raw.min())
    depth_norm = depth_norm.clip(0, 1)
    # apply a Spectral colormap manually via depth_viz already
    fog_frames = 96
    yaw_start = -np.pi * 0.8 + 2 * np.pi  # continue from where we ended
    for i in range(fog_frames):
        t = i / max(fog_frames - 1, 1); e = ease_in_out(t)
        yaw = yaw_start + 0.15 * e
        pitch = 0.06 * np.sin(t * np.pi * 0.8)
        view_rgb, _ = erp_perspective_view(rgb, None, yaw=yaw, pitch=pitch,
                                             fov_deg=78, out_w=W, out_h=H)
        view_depth, _ = erp_perspective_view(s.depth_viz, None, yaw=yaw, pitch=pitch,
                                                fov_deg=78, out_w=W, out_h=H)
        out = blend(view_rgb, view_depth, e * 0.85)
        out = draw_text(out, "depth materialises", (W // 2, H - 50),
                          size=22, color=TEXT_LIGHT, anchor="mm", alpha=0.85)
        out = draw_text(out, "scale-invariant z-depth", (W // 2, H - 22),
                          size=15, color=TEXT_DIM, anchor="mm", alpha=0.7)
        save_frame(out, frames_dir, cursor + i)
    cursor += fog_frames

    # ── scene 4: normals reveal (3s)
    norm_frames = 72
    yaw0 = yaw_start + 0.15
    for i in range(norm_frames):
        t = i / max(norm_frames - 1, 1); e = ease_in_out(t)
        yaw = yaw0 + 0.10 * e
        view_depth, _ = erp_perspective_view(s.depth_viz, None, yaw=yaw, pitch=0,
                                                fov_deg=78, out_w=W, out_h=H)
        view_norm, _ = erp_perspective_view(s.normals_viz, None, yaw=yaw, pitch=0,
                                              fov_deg=78, out_w=W, out_h=H)
        out = blend(view_depth, view_norm, e)
        out = draw_text(out, "normals shade the surfaces", (W // 2, H - 50),
                          size=22, color=TEXT_LIGHT, anchor="mm", alpha=0.85)
        save_frame(out, frames_dir, cursor + i)
    cursor += norm_frames

    # ── scene 5: sky dissolves (2s)
    sky_frames = 48
    yaw0 = yaw_start + 0.25
    sky_mask = s.sky_mask.astype(np.float32) / 255.0
    for i in range(sky_frames):
        t = i / max(sky_frames - 1, 1); e = ease_out(t)
        yaw = yaw0 + 0.05 * e
        rgb_view, _ = erp_perspective_view(s.rgb, None, yaw=yaw, pitch=0,
                                             fov_deg=78, out_w=W, out_h=H)
        norm_view, _ = erp_perspective_view(s.normals_viz, None, yaw=yaw, pitch=0,
                                              fov_deg=78, out_w=W, out_h=H)
        sky_view, _ = erp_perspective_view(
            (sky_mask[:, :, None] * np.ones((1, 1, 3), dtype=np.float32) * 255).astype(np.uint8),
            None, yaw=yaw, pitch=0, fov_deg=78, out_w=W, out_h=H)
        a = (sky_view.astype(np.float32) / 255.0).mean(axis=2, keepdims=True)
        # erase sky to dark
        out = norm_view.astype(np.float32) * (1 - a * e) + np.array(BG_DARK, np.float32) * (a * e)
        out = out.clip(0, 255).astype(np.uint8)
        out = draw_text(out, "sky head masks unbounded depth", (W // 2, H - 50),
                          size=22, color=TEXT_LIGHT, anchor="mm", alpha=0.85)
        save_frame(out, frames_dir, cursor + i)
    cursor += sky_frames

    # ── scene 6: pull back to point cloud (5s)
    pull_frames = 120
    P = center_points(s.points_xyz)
    d_far = autoframe_distance(P, fov_deg=55) * 1.05
    for i in range(pull_frames):
        t = i / max(pull_frames - 1, 1); e = ease_in_out(t)
        # camera distance: starts ~1 (inside) and pulls back to d_far
        dist = 0.8 + e * (d_far - 0.8)
        yaw = 0.0 + e * 1.8
        pitch = 0.10 + 0.10 * np.sin(t * np.pi)
        img = project_pointcloud(P, s.points_rgb, canvas_w=W, canvas_h=H,
                                  yaw=yaw, pitch=pitch, distance=dist, splat=2)
        out = img
        if t > 0.55:
            a = min(1.0, (t - 0.55) / 0.35)
            out = draw_text(out, "PaGeR", (W // 2, H // 2 - 20), size=70,
                              color=TEXT_LIGHT, bold=True, anchor="mm", alpha=a)
            out = draw_text(out, "from a single panorama, the world reconstructs",
                              (W // 2, H // 2 + 32), size=20,
                              color=TEXT_DIM, anchor="mm", alpha=a)
        save_frame(out, frames_dir, cursor + i)
    cursor += pull_frames

    write_video(frames_dir, OUT / "p3_inside.mp4", fps=FPS)
    print(f"✓ wrote {OUT / 'p3_inside.mp4'} ({cursor} frames)")


if __name__ == "__main__":
    main()
