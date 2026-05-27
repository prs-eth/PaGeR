"""Proposal 2: The Atlas — editorial dashboard / paper-figure aesthetic.
White background, 3×3 grid populates tile by tile."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     write_video, fresh_frames_dir, save_frame,
                     BG_WHITE, TEXT_DARK, TEXT_DIM, ACCENT, OUT)

W, H, FPS = 1280, 720, 24
HERO = "livingroom_synth"

# 3×3 grid layout, anchored at the top-left of the canvas
GRID_PAD = 36
GAP = 16
G_W = (W - 2 * GRID_PAD - 2 * GAP) // 3
G_H = (H - 2 * GRID_PAD - 2 * GAP) // 3


def grid_cell(col: int, row: int) -> tuple[int, int, int, int]:
    x = GRID_PAD + col * (G_W + GAP)
    y = GRID_PAD + row * (G_H + GAP)
    return x, y, G_W, G_H


def draw_tile(canvas: np.ndarray, col: int, row: int,
              content: np.ndarray | None, title: str, sub: str = "",
              accent=(40, 40, 50), alpha: float = 1.0) -> np.ndarray:
    x, y, w, h = grid_cell(col, row)
    # tile background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (252, 252, 250), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), accent, 1)
    canvas = blend(canvas, overlay, alpha)
    # title bar
    bar_h = 30
    cv2.rectangle(canvas, (x, y), (x + w, y + bar_h), (245, 244, 240), -1)
    cv2.rectangle(canvas, (x, y), (x + 4, y + bar_h), accent, -1)
    canvas = draw_text(canvas, title, (x + 12, y + 6), size=16,
                       color=TEXT_DARK, bold=True, alpha=alpha)
    if sub:
        canvas = draw_text(canvas, sub, (x + w - 12, y + 9), size=12,
                           color=TEXT_DIM, anchor="rt", alpha=alpha * 0.9)
    # content
    if content is not None:
        cw = w - 12; ch = h - bar_h - 12
        cont = resize(content, w=cw, h=None)
        if cont.shape[0] > ch:
            cont = resize(content, w=None, h=ch)
        cx = x + (w - cont.shape[1]) // 2
        cy = y + bar_h + 6 + (ch - cont.shape[0]) // 2
        canvas = paste(canvas, cont, cx, cy, alpha=alpha)
    return canvas


def make_cubemap_strip(faces, target_w):
    """6 cubemap thumbnails in a row."""
    imgs = [(np.transpose(f, (1, 2, 0)) * 255).astype(np.uint8) for f in faces]
    each_w = target_w // 6 - 4
    thumbs = [resize(im, w=each_w) for im in imgs]
    th = thumbs[0].shape[0]
    strip = np.full((th, target_w, 3), 252, dtype=np.uint8)
    x = 0
    for t in thumbs:
        strip[:, x:x + t.shape[1]] = t; x += t.shape[1] + 4
    return strip


def make_backbone_diagram(w, h):
    """A small paper-style schematic: tokens → ViT → heads."""
    img = np.full((h, w, 3), 252, dtype=np.uint8)
    # ViT block (centre)
    cv2.rectangle(img, (w // 2 - 50, 30), (w // 2 + 50, h - 30), (40, 40, 50), 1)
    img = draw_text(img, "ViT-Giant", (w // 2, h // 2), size=14, color=TEXT_DARK, anchor="mm")
    # input arrow (left)
    cv2.arrowedLine(img, (20, h // 2), (w // 2 - 52, h // 2), (60, 60, 75), 1, tipLength=0.2)
    img = draw_text(img, "cubemap", (20, h // 2 - 22), size=12, color=TEXT_DIM)
    # output arrows (right) -> 4 heads
    for i, lab in enumerate(["depth", "normals", "sky", "scale"]):
        ty = 20 + i * (h - 40) // 3
        cv2.arrowedLine(img, (w // 2 + 52, h // 2), (w - 12, ty + 10), (60, 60, 75), 1, tipLength=0.18)
        img = draw_text(img, lab, (w - 12, ty - 2), size=11, color=TEXT_DARK, anchor="rt")
    return img


def make_metric_panel(w, h, t):
    """Animated metrics card."""
    img = np.full((h, w, 3), 252, dtype=np.uint8)
    # bars
    metrics = [("AbsRel ↓", 0.073, 0.10), ("δ₁ ↑", 0.962, 1.0), ("RMSE ↓", 0.45, 0.8)]
    for i, (lab, v, vmax) in enumerate(metrics):
        cy = 18 + i * 28
        img = draw_text(img, lab, (12, cy - 2), size=12, color=TEXT_DARK, bold=True)
        bw = w - 24
        cv2.rectangle(img, (12, cy + 14), (12 + bw, cy + 22), (228, 228, 220), -1)
        cur = t * v / vmax
        cv2.rectangle(img, (12, cy + 14), (12 + int(bw * cur), cy + 22), (90, 130, 200), -1)
        val_str = f"{v * t:.3f}"
        img = draw_text(img, val_str, (12 + bw, cy + 2), size=11,
                        color=TEXT_DIM, anchor="rt")
    return img


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p2_atlas")
    # pre-compute artifacts
    cube_strip = make_cubemap_strip(s.cubemap, target_w=G_W - 16)
    bone = make_backbone_diagram(G_W - 16, G_H - 50)
    sky_rgb = cv2.cvtColor(cv2.applyColorMap(s.sky_mask, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)

    # ordered reveal of 9 tiles over 9 * step + final hold
    # tile schedule: (col, row, title, sub, content, accent)
    reveal_specs = [
        (0, 0, "Input ERP", "2048×1024", s.rgb, (40, 40, 50)),
        (1, 0, "Cubemap", "6 × 504²", cube_strip, (40, 40, 50)),
        (2, 0, "Backbone", "ViT-Giant", bone, (40, 40, 50)),
        (0, 1, "Depth",     "metric", s.depth_viz, (200, 80, 80)),
        (1, 1, "Normals",   "world frame", s.normals_viz, (80, 130, 200)),
        (2, 1, "Sky",       "mask", sky_rgb, (90, 130, 110)),
        (0, 2, "Scale",     s.scene.lower(), None, (200, 140, 60)),
        (1, 2, "Point cloud", "RGB", None, (60, 60, 70)),
        (2, 2, "Metrics",   "on Stanford2D3D", None, (130, 100, 180)),
    ]
    step = 8  # frames per tile reveal
    intro = 24
    pc_window_frames = 110   # while metrics tick
    total = intro + step * len(reveal_specs) + pc_window_frames
    # final hold
    hold = 30
    total += hold

    # cached point cloud orbit frames (compute on demand)
    P = center_points(s.points_xyz)
    d = autoframe_distance(P, fov_deg=50)

    for i in range(total):
        f = blank(W, H, BG_WHITE)
        # title strip
        f = draw_text(f, "PaGeR", (GRID_PAD, 14), size=22, color=TEXT_DARK, bold=True)
        f = draw_text(f, "Panoramic Geometry Reconstruction — one forward pass.",
                      (GRID_PAD + 90, 16), size=16, color=TEXT_DIM)

        # tile reveal alpha computation
        for k, (c, r, ttl, sub, ct, acc) in enumerate(reveal_specs):
            start = intro + k * step
            t = (i - start) / max(step, 1)
            alpha = max(0.0, min(1.0, t))
            if alpha <= 0: continue
            # the point cloud tile renders live once unlocked
            if (c, r) == (1, 2):
                pt_t = (i - (intro + 7 * step)) / max(1, pc_window_frames + hold)
                yaw = -0.4 + ease_in_out(min(1.0, pt_t)) * 1.5
                pitch = 0.1
                pc_img = project_pointcloud(P, s.points_rgb,
                                              canvas_w=G_W - 16, canvas_h=G_H - 50,
                                              yaw=yaw, pitch=pitch, distance=d,
                                              splat=2, bg=(252, 252, 250))
                f = draw_tile(f, c, r, pc_img, ttl, sub, accent=acc, alpha=alpha)
            elif (c, r) == (0, 2):
                # scale gauge
                gw = G_W - 16; gh = G_H - 50
                gauge = np.full((gh, gw, 3), 252, dtype=np.uint8)
                cx2 = gw // 2; cy2 = gh - 24; rad = int(gh * 0.6)
                cv2.ellipse(gauge, (cx2, cy2), (rad, rad), 0, 200, 340, (200, 200, 195), 2)
                ang_deg = 200 + 140 * (0.85 if s.scene == "Outdoor" else 0.18)
                ang = np.deg2rad(ang_deg)
                ex = int(cx2 + rad * np.cos(ang)); ey = int(cy2 + rad * np.sin(ang))
                cv2.line(gauge, (cx2, cy2), (ex, ey), acc, 3)
                gauge = draw_text(gauge, "indoor", (16, gh - 14), size=11, color=TEXT_DIM)
                gauge = draw_text(gauge, "outdoor", (gw - 50, gh - 14), size=11, color=TEXT_DIM)
                gauge = draw_text(gauge, "CLIP router", (gw // 2, 8), size=11,
                                   color=TEXT_DIM, anchor="mt")
                gauge = draw_text(gauge, s.scene.upper(), (gw // 2, gh // 2 - 4),
                                   size=18, color=TEXT_DARK, bold=True, anchor="mm")
                f = draw_tile(f, c, r, gauge, ttl, sub, accent=acc, alpha=alpha)
            elif (c, r) == (2, 2):
                # metrics ticking
                mt = max(0.0, min(1.0, (i - (intro + 8 * step)) / max(1, pc_window_frames * 0.8)))
                mp = make_metric_panel(G_W - 16, G_H - 50, mt)
                f = draw_tile(f, c, r, mp, ttl, sub, accent=acc, alpha=alpha)
            else:
                f = draw_tile(f, c, r, ct, ttl, sub, accent=acc, alpha=alpha)

        save_frame(f, frames_dir, i)

    write_video(frames_dir, OUT / "p2_atlas.mp4", fps=FPS)
    print(f"✓ wrote {OUT / 'p2_atlas.mp4'} ({total} frames)")


if __name__ == "__main__":
    main()
