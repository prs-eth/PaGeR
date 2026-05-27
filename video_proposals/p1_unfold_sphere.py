"""Proposal 1: Unfold the Sphere — cinematic, geometry-forward.
Flow: ERP drifts in → wraps to sphere → splits into cubemap → backbone flow → 4 head ribbons → point cloud."""
from __future__ import annotations
import sys; sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import numpy as np, cv2
from pathlib import Path
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     erp_to_sphere, write_video, fresh_frames_dir, save_frame,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT)

W, H, FPS = 1280, 720, 24
HERO = "eth_campus_plaza"


def scene_intro(s, frames_dir, start, dur):
    """ERP flat drifts in over the void."""
    erp = resize(s.rgb, w=int(W * 0.78))
    erp_h, erp_w = erp.shape[:2]
    for i in range(dur):
        t = i / max(dur - 1, 1)
        e = ease_out(t)
        f = blank(W, H)
        y = int(H * 0.42 - erp_h / 2)
        x = int(W / 2 - erp_w / 2)
        # subtle "scroll" feel: shift the ERP a bit horizontally
        shift = int(-erp_w * 0.15 * (1 - e))
        f = paste(f, np.roll(erp, shift, axis=1), x, y, alpha=e)
        f = draw_text(f, "PaGeR", (W // 2, H - 110), size=58, color=TEXT_LIGHT,
                      bold=True, anchor="mm", alpha=e)
        f = draw_text(f, "Panoramic Geometry Reconstruction", (W // 2, H - 60),
                      size=22, color=TEXT_DIM, anchor="mm", alpha=e * 0.85)
        save_frame(f, frames_dir, start + i)


def scene_wrap(s, frames_dir, start, dur):
    """ERP morphs into a sphere as the camera orbits."""
    erp = resize(s.rgb, w=int(W * 0.78))
    erp_h, erp_w = erp.shape[:2]
    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        # cross-fade flat→sphere by overlaying the sphere render with rising alpha
        if e < 0.99:
            x = int(W / 2 - erp_w / 2); y = int(H * 0.42 - erp_h / 2)
            f = paste(f, erp, x, y, alpha=1 - e)
        if e > 0.01:
            yaw = -0.25 + e * 0.4
            sph = erp_to_sphere(s.rgb, canvas_w=W, canvas_h=H, yaw=yaw, radius_frac=0.30 + 0.04 * e)
            f = blend(f, sph, 1.0) if e > 0.99 else blend(f, sph, e)
        f = draw_text(f, "1.  ERP panorama → sphere", (60, H - 50), size=22, color=TEXT_DIM)
        save_frame(f, frames_dir, start + i)


def scene_cubemap(s, frames_dir, start, dur):
    """Sphere bursts into 6 cubemap faces, splayed out."""
    faces = s.cubemap  # (6, 3, h, w) -> [F, R, B, L, U, D]
    face_imgs = [(np.transpose(f, (1, 2, 0)) * 255).astype(np.uint8) for f in faces]
    face_imgs = [resize(im, w=180) for im in face_imgs]
    fh, fw = face_imgs[0].shape[:2]
    # cross layout: F center, R right, L left, U above, D below, B far-right
    layout = [
        (W // 2 - fw // 2,                H // 2 - fh // 2),       # F
        (W // 2 - fw // 2 + int(fw * 1.05), H // 2 - fh // 2),     # R
        (W // 2 - fw // 2 + int(fw * 2.10), H // 2 - fh // 2),     # B
        (W // 2 - fw // 2 - int(fw * 1.05), H // 2 - fh // 2),     # L
        (W // 2 - fw // 2,                H // 2 - fh // 2 - int(fh * 1.05)),  # U
        (W // 2 - fw // 2,                H // 2 - fh // 2 + int(fh * 1.05)),  # D
    ]
    labels = ["F", "R", "B", "L", "U", "D"]
    sph_yaw_end = 0.15
    sph = erp_to_sphere(s.rgb, canvas_w=W, canvas_h=H, yaw=sph_yaw_end, radius_frac=0.34)
    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        # sphere fades out as faces fly out
        f = blend(f, sph, max(0.0, 1.0 - e * 1.4))
        # faces fly from center
        for (fx, fy), face, lab in zip(layout, face_imgs, labels):
            cx, cy = W // 2 - fw // 2, H // 2 - fh // 2
            x = int(cx + (fx - cx) * e); y = int(cy + (fy - cy) * e)
            f = paste(f, face, x, y, alpha=min(1.0, e * 1.5))
            if e > 0.5:
                f = draw_text(f, lab, (x + 8, y + 6), size=18, color=ACCENT,
                              bold=True, alpha=min(1.0, (e - 0.5) * 2))
        f = draw_text(f, "2.  6-face cubemap (504×504, FoV 90°)",
                      (60, H - 50), size=22, color=TEXT_DIM)
        save_frame(f, frames_dir, start + i)


def scene_backbone(s, frames_dir, start, dur):
    """Cubemap faces stream through a stylised ViT 'neural manifold'."""
    faces = s.cubemap
    face_imgs = [(np.transpose(f, (1, 2, 0)) * 255).astype(np.uint8) for f in faces]
    face_imgs = [resize(im, w=110) for im in face_imgs]
    # token grid background
    rng = np.random.default_rng(42)
    tokens = rng.uniform(0, 1, size=(220, 2))
    tokens[:, 0] = tokens[:, 0] * W
    tokens[:, 1] = 180 + tokens[:, 1] * (H - 360)
    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)
        # tokens shimmer
        shimmer = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + tokens[:, 0] / W * 4)))
        for (tx, ty), sh in zip(tokens, shimmer):
            r = 1 + int(sh * 1.5)
            c = int(180 * sh)
            cv2.circle(f, (int(tx), int(ty)), r, (c, c, c + 20), -1)
        # input column (6 faces) left, swept right across the manifold over time
        in_x = int(80 + t * (W * 0.35))
        for k, face in enumerate(face_imgs):
            fy = 90 + k * 90
            f = paste(f, face, in_x, fy, alpha=0.85)
        # output stub on the right
        out_x = int(W - 80 - face_imgs[0].shape[1])
        if t > 0.4:
            a = min(1.0, (t - 0.4) / 0.4)
            for k, face in enumerate(face_imgs):
                fy = 90 + k * 90
                f = paste(f, face, out_x, fy, alpha=a * 0.95)
        # central label
        f = draw_text(f, "Multi-view Backbone", (W // 2, 60), size=28,
                      color=TEXT_LIGHT, bold=True, anchor="mm")
        f = draw_text(f, "ViT-Giant · cross-face attention", (W // 2, 96),
                      size=18, color=TEXT_DIM, anchor="mm")
        f = draw_text(f, "3.  Cubemap tokens → multi-view ViT",
                      (60, H - 50), size=22, color=TEXT_DIM)
        save_frame(f, frames_dir, start + i)


def scene_heads(s, frames_dir, start, dur):
    """Four ribbons (Depth, Normals, Sky, Scale) emerge in parallel."""
    rgb = resize(s.rgb, w=int(W * 0.42))
    depth = resize(s.depth_viz, w=int(W * 0.42))
    norms = resize(s.normals_viz, w=int(W * 0.42))
    sky = cv2.applyColorMap(s.sky_mask, cv2.COLORMAP_BONE)
    sky = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB)
    sky = resize(sky, w=int(W * 0.42))
    rh, rw = rgb.shape[:2]
    tile = (rw, rh)
    cells = [
        ("Depth",   depth, ACCENT),
        ("Normals", norms, (140, 200, 255)),
        ("Sky",     sky,   (200, 230, 255)),
        ("Scale",   None,  (255, 180, 200)),
    ]
    pos = [(60, 80), (W - rw - 60, 80), (60, H - rh - 80), (W - rw - 60, H - rh - 80)]
    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        # input rgb centered, shrinking
        scale = 1.0 - 0.45 * e
        cx, cy = W // 2, H // 2
        cw = int(rw * scale); ch = int(rh * scale)
        cur = resize(rgb, w=cw)
        f = paste(f, cur, cx - cw // 2, cy - ch // 2, alpha=max(0.25, 1 - e * 0.7))
        for (label, img, col), (x, y) in zip(cells, pos):
            a = max(0.0, min(1.0, (e - 0.2) * 2.0))
            if a <= 0: continue
            if img is not None:
                f = paste(f, img, x, y, alpha=a)
            else:
                # scale head: a stylised gauge swinging between indoor/outdoor
                cv2.rectangle(f, (x, y), (x + rw, y + rh), (40, 42, 52), -1)
                cx2 = x + rw // 2; cy2 = y + rh - 30
                radius = int(rh * 0.55)
                # arc baseline
                cv2.ellipse(f, (cx2, cy2), (radius, radius), 0, 200, 340, (90, 95, 110), 2)
                # needle
                ang_deg = 200 + 140 * (0.85 if s.scene == "Outdoor" else 0.18)
                ang = np.deg2rad(ang_deg)
                ex = int(cx2 + radius * np.cos(ang)); ey = int(cy2 + radius * np.sin(ang))
                cv2.line(f, (cx2, cy2), (ex, ey), col, 3)
                f = draw_text(f, "Indoor",  (x + 24,        y + rh - 8), size=14, color=TEXT_DIM)
                f = draw_text(f, "Outdoor", (x + rw - 70,   y + rh - 8), size=14, color=TEXT_DIM)
                f = draw_text(f, s.scene.upper(), (cx2, y + 30), size=16,
                              color=ACCENT, bold=True, anchor="mm")
            # label bar
            cv2.rectangle(f, (x, y - 22), (x + rw, y), (20, 22, 28), -1)
            cv2.rectangle(f, (x, y - 22), (x + 6, y), col, -1)
            f = draw_text(f, label, (x + 14, y - 18), size=16,
                          color=TEXT_LIGHT, bold=True, alpha=a)
        f = draw_text(f, "4.  Four heads, one forward pass",
                      (60, H - 50), size=22, color=TEXT_DIM)
        save_frame(f, frames_dir, start + i)


def scene_pointcloud(s, frames_dir, start, dur):
    """Final reveal — orbit the colored point cloud."""
    P = center_points(s.points_xyz)
    d = autoframe_distance(P, fov_deg=55)
    for i in range(dur):
        t = i / max(dur - 1, 1)
        yaw = -0.4 + ease_out(t) * 1.6
        pitch = 0.05 + 0.06 * np.sin(t * np.pi)
        img = project_pointcloud(P, s.points_rgb, canvas_w=W, canvas_h=H,
                                  yaw=yaw, pitch=pitch, distance=d, splat=2)
        # fade-in + title at the end
        fade = min(1.0, t * 3.0)
        img = (img.astype(np.float32) * fade).clip(0, 255).astype(np.uint8)
        if t > 0.5:
            a = min(1.0, (t - 0.5) * 2.5)
            img = draw_text(img, "PaGeR", (W // 2, H // 2 - 20), size=72,
                            color=TEXT_LIGHT, bold=True, anchor="mm", alpha=a)
            img = draw_text(img, "single forward pass · panorama → metric 3D",
                            (W // 2, H // 2 + 36), size=22, color=TEXT_DIM,
                            anchor="mm", alpha=a)
        save_frame(img, frames_dir, start + i)


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1_unfold")
    # timings (frames at 24fps)
    timings = [
        (scene_intro,      36),
        (scene_wrap,       54),
        (scene_cubemap,    60),
        (scene_backbone,   72),
        (scene_heads,      72),
        (scene_pointcloud, 96),
    ]
    cursor = 0
    for fn, n in timings:
        fn(s, frames_dir, cursor, n); cursor += n
        print(f"  scene done: {fn.__name__} ({n} frames, total {cursor})")
    write_video(frames_dir, OUT / "p1_unfold_sphere.mp4", fps=FPS)
    print(f"✓ wrote {OUT / 'p1_unfold_sphere.mp4'}")


if __name__ == "__main__":
    main()
