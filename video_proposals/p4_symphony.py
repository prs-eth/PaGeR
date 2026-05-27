"""Proposal 4: Multi-Head Symphony — architecture-forward, parallel head outputs.
Persistent layout: ERP top, backbone middle, 4 head quadrants, CLIP router bar."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     write_video, fresh_frames_dir, save_frame,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT)

W, H, FPS = 1280, 720, 24

INDOOR  = "livingroom_synth"
OUTDOOR = "eth_campus_plaza"


def panel_frame(canvas, x, y, w, h, label, accent, alpha=1.0):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 64, 78), 1)
    cv2.rectangle(canvas, (x, y), (x + 5, y + h), accent, -1)
    canvas = draw_text(canvas, label, (x + 12, y + 6), size=14,
                       color=TEXT_LIGHT, bold=True, alpha=alpha)
    return canvas


def render_layout(s, *, t_heads: float, router_pos: float, router_target: float,
                  show_pc: bool = False, pc_yaw: float = 0.0, pc_dist: float = 5.0,
                  P: np.ndarray | None = None) -> np.ndarray:
    """Render one frame of the persistent layout.
    t_heads ∈ [0,1] controls fade-in of head outputs.
    router_pos ∈ [0,1] is the live needle position (0=indoor, 1=outdoor).
    """
    f = blank(W, H)
    # title
    f = draw_text(f, "PaGeR", (24, 14), size=22, color=TEXT_LIGHT, bold=True)
    f = draw_text(f, "single backbone · four heads · one forward pass",
                  (24 + 88, 18), size=14, color=TEXT_DIM)

    # input strip (top, scrolls) — stretched to full width
    erp_h_target = 90
    ex = 24; ey = 48; ew = W - 48
    erp = cv2.resize(s.rgb, (ew, erp_h_target), interpolation=cv2.INTER_AREA)
    shift = int((np.sin(router_pos * np.pi) * 0.05 + 0.0) * ew)
    erp_roll = np.roll(erp, shift, axis=1)
    f = paste(f, erp_roll, ex, ey)
    f = draw_text(f, "INPUT · ERP panorama (1024×2048)", (ex + 8, ey + erp_h_target - 22),
                  size=12, color=(220, 220, 220))

    # backbone band (middle-narrow)
    by = ey + erp_h_target + 14
    bh = 70
    cv2.rectangle(f, (ex, by), (ex + ew, by + bh), (16, 18, 24), -1)
    cv2.rectangle(f, (ex, by), (ex + ew, by + bh), (60, 64, 78), 1)
    # animated token stream
    n = 60
    for k in range(n):
        phase = (router_pos * 2 + k / n) % 1.0
        xk = ex + int(phase * ew)
        yk = by + int(bh * (0.3 + 0.4 * np.sin(k * 0.3 + router_pos * 6)))
        cv2.circle(f, (xk, yk), 1, (180, 200, 220), -1)
    f = draw_text(f, "ViT-Giant  ·  multi-view cross-attention",
                  (ex + ew // 2, by + bh // 2), size=18,
                  color=TEXT_LIGHT, bold=True, anchor="mm")

    # four quadrants below
    qy = by + bh + 14
    qh = (H - qy - 80) // 2 - 6   # below leave 80px for the router
    qw = (ew - 12) // 2
    panels = [
        (ex,                qy,                  "DEPTH",   ACCENT,            s.depth_viz),
        (ex + qw + 12,      qy,                  "NORMALS", (140, 200, 255),    s.normals_viz),
        (ex,                qy + qh + 12,        "SKY",     (200, 230, 255),    None),
        (ex + qw + 12,      qy + qh + 12,        "SCALE",   (255, 180, 200),    None),
    ]
    for px, py, lab, col, img in panels:
        f = panel_frame(f, px, py, qw, qh, lab, col, alpha=1.0)
        a = t_heads
        if img is not None:
            cont = resize(img, w=qw - 14, h=qh - 32)
            f = paste(f, cont, px + 7, py + 26, alpha=a)
        elif lab == "SKY":
            # depth-derived sky mask, prettified
            m = (s.sky_mask.astype(np.float32) / 255.0)
            # render as cyan-tinted mask
            sky_rgb = np.zeros((*m.shape, 3), dtype=np.float32)
            sky_rgb[..., 0] = m * 120
            sky_rgb[..., 1] = m * 200
            sky_rgb[..., 2] = m * 255
            cont = resize(sky_rgb.astype(np.uint8), w=qw - 14, h=qh - 32)
            f = paste(f, cont, px + 7, py + 26, alpha=a)
        else:  # SCALE
            cw = qw - 14; ch = qh - 32
            gauge = np.full((ch, cw, 3), 22, dtype=np.uint8)
            cx2 = cw // 2; cy2 = ch - 28; rad = int(ch * 0.55)
            cv2.ellipse(gauge, (cx2, cy2), (rad, rad), 0, 200, 340, (70, 75, 90), 2)
            ang_deg = 200 + 140 * router_pos
            ang = np.deg2rad(ang_deg)
            ex2 = int(cx2 + rad * np.cos(ang)); ey2 = int(cy2 + rad * np.sin(ang))
            cv2.line(gauge, (cx2, cy2), (ex2, ey2), col, 3)
            gauge = draw_text(gauge, "indoor",  (12, ch - 14), size=11, color=TEXT_DIM)
            gauge = draw_text(gauge, "outdoor", (cw - 50, ch - 14), size=11, color=TEXT_DIM)
            tag = "OUTDOOR" if router_pos > 0.5 else "INDOOR"
            gauge = draw_text(gauge, tag, (cx2, ch // 2 - 4), size=18,
                                color=TEXT_LIGHT, bold=True, anchor="mm")
            f = paste(f, gauge, px + 7, py + 26, alpha=a)

    # CLIP router strip (bottom)
    ry = H - 56
    rh = 36
    cv2.rectangle(f, (ex, ry), (ex + ew, ry + rh), (16, 18, 24), -1)
    cv2.rectangle(f, (ex, ry), (ex + ew, ry + rh), (60, 64, 78), 1)
    f = draw_text(f, "CLIP router", (ex + 12, ry + 8), size=13,
                  color=TEXT_DIM, bold=True)
    f = draw_text(f, "INDOOR", (ex + 110, ry + 8), size=13, color=TEXT_DIM)
    f = draw_text(f, "OUTDOOR", (ex + ew - 12, ry + 8), size=13,
                  color=TEXT_DIM, anchor="rt")
    bar_x0 = ex + 175; bar_x1 = ex + ew - 100
    bar_y = ry + rh - 14
    cv2.line(f, (bar_x0, bar_y), (bar_x1, bar_y), (60, 64, 78), 2)
    needle_x = bar_x0 + int((bar_x1 - bar_x0) * router_pos)
    cv2.circle(f, (needle_x, bar_y), 6, ACCENT, -1)

    # if we're in the PC reveal, overlay an enlarging point cloud
    if show_pc and P is not None:
        pc = project_pointcloud(P, s.points_rgb,
                                canvas_w=ew, canvas_h=H - qy - 80,
                                yaw=pc_yaw, pitch=0.1, distance=pc_dist, splat=2)
        # show full-width over the quadrants
        f = paste(f, pc, ex, qy, alpha=1.0)
        # final label
        f = draw_text(f, "→ point cloud", (W // 2, qy + 28), size=18,
                       color=TEXT_LIGHT, bold=True, anchor="mm")

    return f


def main():
    s_in = load_sample(INDOOR)
    s_out = load_sample(OUTDOOR)
    frames_dir = fresh_frames_dir("p4_symphony")

    cursor = 0
    # ── Movement 1: indoor (5s)
    n = 120
    for i in range(n):
        t = i / max(n - 1, 1)
        # heads fade in over the first 1.5s
        t_heads = min(1.0, t * 4)
        router = 0.18  # indoor
        # tiny noise
        router = router + 0.01 * np.sin(i * 0.3)
        f = render_layout(s_in, t_heads=t_heads, router_pos=router, router_target=0.18)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    # ── Transition (1.5s): router needle swings indoor → outdoor while ERP cross-fades
    n = 36
    for i in range(n):
        t = i / max(n - 1, 1); e = ease_in_out(t)
        router = 0.18 + (0.85 - 0.18) * e
        # blend two layouts
        f_in = render_layout(s_in, t_heads=1.0, router_pos=router, router_target=0.85)
        f_out = render_layout(s_out, t_heads=1.0, router_pos=router, router_target=0.85)
        f = blend(f_in, f_out, e)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    # ── Movement 2: outdoor (5s)
    n = 120
    for i in range(n):
        t = i / max(n - 1, 1)
        router = 0.85 + 0.01 * np.sin(i * 0.3)
        f = render_layout(s_out, t_heads=1.0, router_pos=router, router_target=0.85)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    # ── Movement 3: collapse to point cloud (5s)
    n = 120
    P = center_points(s_out.points_xyz)
    d = autoframe_distance(P, fov_deg=55) * 1.05
    for i in range(n):
        t = i / max(n - 1, 1); e = ease_in_out(t)
        router = 0.85
        # ramp pc overlay
        if e > 0.15:
            a = (e - 0.15) / 0.85
            pc_yaw = -0.3 + a * 1.4
            f = render_layout(s_out, t_heads=1.0 - a * 0.85, router_pos=router,
                                router_target=0.85, show_pc=True,
                                pc_yaw=pc_yaw, pc_dist=d, P=P)
        else:
            f = render_layout(s_out, t_heads=1.0, router_pos=router, router_target=0.85)
        if e > 0.6:
            a = min(1.0, (e - 0.6) / 0.35)
            f = draw_text(f, "PaGeR", (W // 2, H - 76), size=42,
                           color=TEXT_LIGHT, bold=True, anchor="mm", alpha=a)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    write_video(frames_dir, OUT / "p4_symphony.mp4", fps=FPS)
    print(f"✓ wrote {OUT / 'p4_symphony.mp4'} ({cursor} frames)")


if __name__ == "__main__":
    main()
