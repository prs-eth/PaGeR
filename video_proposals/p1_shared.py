"""Shared prologue (intro → wrap → splay → stack) and epilogue (PC orbit) for the p1 variants."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     erp_to_sphere, render_sphere_morph, save_frame,
                     make_coarse_metric_depth, make_sky_viz, make_cubemap_stack,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT)

W, H, FPS = 1280, 720, 24

# Per-head accent colors
COLOR_DEPTH        = ACCENT                  # warm gold (Spectral-ish)
COLOR_METRIC       = (244, 124,  95)         # inferno-orange
COLOR_NORMALS      = (140, 200, 255)         # cool blue
COLOR_SKY          = (200, 230, 255)         # pale cyan

HEAD_LABELS = ["DEPTH", "METRIC DEPTH", "NORMALS", "SKY"]
HEAD_SUBS   = ["scale-invariant", "coarse · metric", "world frame", "mask"]
HEAD_COLORS = [COLOR_DEPTH, COLOR_METRIC, COLOR_NORMALS, COLOR_SKY]


def make_head_panels(s) -> list[np.ndarray]:
    """Return the four head-output panels at full ERP resolution: depth, coarse_metric, normals, sky."""
    coarse = make_coarse_metric_depth(s.depth_raw, s.sky_mask, downsample=10, blur_sigma=4.0)
    sky = make_sky_viz(s.sky_mask)
    return [s.depth_viz, coarse, s.normals_viz, sky]


# ────────────────────────────────────────────────────────────────────
# Prologue scenes (shared with original p1, slightly tweaked)
# ────────────────────────────────────────────────────────────────────
def scene_intro(s, frames_dir, start, dur):
    """Longer ERP scroll-in. PaGeR title softly fades in, holds, and fades out
    before the wrap scene takes over — no hard cuts on the title."""
    erp = resize(s.rgb, w=int(W * 0.82))
    erp_h, erp_w = erp.shape[:2]
    y = int(H * 0.42 - erp_h / 2); x = int(W / 2 - erp_w / 2)
    fade_in_dur = max(8, int(dur * 0.20))
    # title timing (fractions of dur): very slow fade-in (~50% of intro), brief hold, soft fade-out
    t_in_a = int(dur * 0.30); t_in_b = int(dur * 0.82)
    t_out_a = int(dur * 0.90); t_out_b = dur - 1
    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)
        a = min(1.0, i / max(1, fade_in_dur))
        shift = int(-erp_w * 1.0 * t * 0.5)
        f = paste(f, np.roll(erp, shift, axis=1), x, y, alpha=a)
        # PaGeR title — soft fade in, hold, soft fade out
        if i < t_in_a:
            ta = 0.0
        elif i < t_in_b:
            ta = ease_in_out((i - t_in_a) / max(1, t_in_b - t_in_a))
        elif i < t_out_a:
            ta = 1.0
        else:
            ta = 1.0 - ease_in_out((i - t_out_a) / max(1, t_out_b - t_out_a))
        if ta > 0.001:
            f = draw_text(f, "PaGeR", (W // 2, H - 110), size=58,
                            color=TEXT_LIGHT, bold=True, anchor="mm", alpha=ta)
            f = draw_text(f, "Panoramic Geometry Reconstruction",
                            (W // 2, H - 60), size=22, color=TEXT_DIM,
                            anchor="mm", alpha=ta * 0.85)
        save_frame(f, frames_dir, start + i)


def scene_wrap(s, frames_dir, start, dur):
    """Geometric morph: a dense grid of ERP samples bends in 3D from a flat rectangle
    to a sphere. The first frame matches the intro's final ERP (size, pos, 50% roll);
    in the last ~15% we crossfade into the analytic erp_to_sphere render so the polar
    aliasing inherent to the forward warp disappears before we hand off to splay."""
    clean_sphere = erp_to_sphere(s.rgb, canvas_w=W, canvas_h=H,
                                  yaw=0.0, radius_frac=0.34)
    for i in range(dur):
        t = i / max(dur - 1, 1); a = ease_in_out(t)
        f = render_sphere_morph(s.rgb, alpha=a, canvas_w=W, canvas_h=H,
                                  flat_aspect=1.5, R=1.0,
                                  scale_flat=350.0, scale_sphere=245.0,
                                  cy_flat=int(H * 0.42), cy_sphere=H // 2,
                                  texture_yaw_flat=np.pi, splat=2,
                                  Nu=700, Nv=350, bg=BG_DARK)
        if a > 0.82:
            t_clean = ease_in_out(min(1.0, (a - 0.82) / 0.18))
            f = blend(f, clean_sphere, t_clean)
        save_frame(f, frames_dir, start + i)


def scene_splay(s, frames_dir, start, dur):
    """Sphere bursts into 6 cubemap faces splayed out as a cross."""
    faces = s.cubemap
    face_imgs = [(np.transpose(f, (1, 2, 0)) * 255).astype(np.uint8) for f in faces]
    face_imgs = [resize(im, w=160) for im in face_imgs]
    fh, fw = face_imgs[0].shape[:2]
    layout = [
        (W // 2 - fw // 2,                H // 2 - fh // 2),
        (W // 2 - fw // 2 + int(fw * 1.05), H // 2 - fh // 2),
        (W // 2 - fw // 2 + int(fw * 2.10), H // 2 - fh // 2),
        (W // 2 - fw // 2 - int(fw * 1.05), H // 2 - fh // 2),
        (W // 2 - fw // 2,                H // 2 - fh // 2 - int(fh * 1.05)),
        (W // 2 - fw // 2,                H // 2 - fh // 2 + int(fh * 1.05)),
    ]
    labels = ["F", "R", "B", "L", "U", "D"]
    # splay starts with the same sphere the morph ends on (yaw=0, radius_frac=0.34)
    sph_yaw_end = 0.0
    sph = erp_to_sphere(s.rgb, canvas_w=W, canvas_h=H, yaw=sph_yaw_end, radius_frac=0.34)
    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        f = blend(f, sph, max(0.0, 1.0 - e * 1.4))
        for (fx, fy), face, lab in zip(layout, face_imgs, labels):
            cx, cy = W // 2 - fw // 2, H // 2 - fh // 2
            x = int(cx + (fx - cx) * e); y = int(cy + (fy - cy) * e)
            f = paste(f, face, x, y, alpha=min(1.0, e * 1.5))
            if e > 0.5:
                f = draw_text(f, lab, (x + 8, y + 6), size=18, color=ACCENT,
                              bold=True, alpha=min(1.0, (e - 0.5) * 2))
        save_frame(f, frames_dir, start + i)


def scene_collapse_to_zstack(s, frames_dir, start, dur,
                              cx: int = 640, cy: int = 360, front_w: int = 230,
                              dx_off: int = 11, dy_off: int = -11,
                              scale_decay: float = 0.95):
    """Cross splayed → Z-axis stack of 6 cubemap faces overlapping perpendicular to the screen.
    Cards converge to roughly the same (x,y) with small offsets in the depth direction;
    back cards are darkened to suggest depth."""
    faces = s.cubemap
    face_imgs_big = [(np.transpose(f, (1, 2, 0)) * 255).astype(np.uint8) for f in faces]
    face_imgs_big = [resize(im, w=160) for im in face_imgs_big]
    fh, fw = face_imgs_big[0].shape[:2]
    # cross layout (same as scene_splay's end positions)
    start_pos = [
        (W // 2 - fw // 2,                  H // 2 - fh // 2),                      # F
        (W // 2 - fw // 2 + int(fw * 1.05), H // 2 - fh // 2),                      # R
        (W // 2 - fw // 2 + int(fw * 2.10), H // 2 - fh // 2),                      # B
        (W // 2 - fw // 2 - int(fw * 1.05), H // 2 - fh // 2),                      # L
        (W // 2 - fw // 2,                  H // 2 - fh // 2 - int(fh * 1.05)),     # U
        (W // 2 - fw // 2,                  H // 2 - fh // 2 + int(fh * 1.05)),     # D
    ]
    # Z-stack end positions: layer k retreats further into the screen
    # (top-left = front; back layers peek out down-right.)
    end_info = []
    for k in range(6):
        w = int(front_w * (scale_decay ** k))
        x_top_left = cx + k * dx_off - w // 2
        y_top_left = cy + k * dy_off - w // 2
        end_info.append((x_top_left, y_top_left, w))
    labels = ["F", "R", "B", "L", "U", "D"]

    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        # render back-to-front so the front card occludes the back
        for k in reversed(range(6)):
            sx, sy = start_pos[k]
            ex_, ey_, ew = end_info[k]
            cur_w = int(fw + (ew - fw) * e)
            # interpolate from start corner to end-center-of-card
            # (start_pos is top-left of an fw×fw card; end is top-left of an ew×ew card)
            ax = sx + (ex_ - sx) * e
            ay = sy + (ey_ - sy) * e
            # but during transit, we use cur_w which may differ from both — recenter so the card grows around its center
            sc_x = sx + fw / 2; sc_y = sy + fw / 2
            ec_x = ex_ + ew / 2; ec_y = ey_ + ew / 2
            cur_cx = sc_x + (ec_x - sc_x) * e
            cur_cy = sc_y + (ec_y - sc_y) * e
            x = int(cur_cx - cur_w / 2); y = int(cur_cy - cur_w / 2)
            cur = cv2.resize(face_imgs_big[k], (cur_w, cur_w), interpolation=cv2.INTER_AREA)
            # darken back cards progressively as the stack forms
            shade = 1.0 - e * (k / 5.0) * 0.45
            cur = (cur.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)
            f = paste(f, cur, x, y, alpha=1.0)
            cv2.rectangle(f, (x, y), (x + cur_w, y + cur_w),
                          (90 + int(20 * (1 - k / 5)), 95 + int(20 * (1 - k / 5)), 110 + int(20 * (1 - k / 5))), 1)
        save_frame(f, frames_dir, start + i)


# back-compat alias so existing variants keep importing the same symbol;
# they'll just use the Z-stack now.
scene_collapse_to_stack = scene_collapse_to_zstack


# ────────────────────────────────────────────────────────────────────
# Shared epilogue — long PC orbit (~7s)
# ────────────────────────────────────────────────────────────────────
def scene_pc_orbit(s, frames_dir, start, dur, *,
                    initial_d: float = 0.9, splat: int = 2,
                    title_in_t: float = 0.35, title_full_t: float = 0.55,
                    title_out_t: float = 0.88, title_gone_t: float = 1.00,
                    intro_blackin: int = 0,
                    pitch_start: float = 0.05, pitch_end: float = 0.85):
    """Long pull-back + orbit reveal. Distance grows (zoom out), yaw rotates,
    and pitch ramps upward so the camera lifts diagonally and ends in a
    bird's-eye / sky-box view of the scene. PaGeR title fades in then out."""
    P = center_points(s.points_xyz)
    d_far = autoframe_distance(P, fov_deg=55) * 1.05
    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        dist = initial_d + e * (d_far - initial_d)
        yaw = -0.4 + e * 2.0
        # pitch climbs monotonically from a slight tilt to a clear bird's-eye angle
        pitch = pitch_start + (pitch_end - pitch_start) * e
        img = project_pointcloud(P, s.points_rgb, canvas_w=W, canvas_h=H,
                                  yaw=yaw, pitch=pitch, distance=dist, splat=splat)
        if intro_blackin and i < intro_blackin:
            a = i / max(1, intro_blackin - 1)
            img = (img.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
        # PaGeR title — fade in, hold, fade out (all smooth)
        if t < title_in_t:
            ta = 0.0
        elif t < title_full_t:
            ta = ease_in_out((t - title_in_t) / max(1e-3, title_full_t - title_in_t))
        elif t < title_out_t:
            ta = 1.0
        else:
            ta = 1.0 - ease_in_out(min(1.0, (t - title_out_t) /
                                              max(1e-3, title_gone_t - title_out_t)))
        if ta > 0.001:
            img = draw_text(img, "PaGeR", (W // 2, H // 2 - 24), size=78,
                              color=TEXT_LIGHT, bold=True, anchor="mm", alpha=ta)
            img = draw_text(img, "single forward pass · panorama → metric 3D",
                              (W // 2, H // 2 + 36), size=22, color=TEXT_DIM,
                              anchor="mm", alpha=ta)
        save_frame(img, frames_dir, start + i)


def run_prologue(s, frames_dir) -> int:
    """Render the shared opening (longer intro → wrap → splay → Z-stack). Return cursor."""
    cursor = 0
    for fn, n in [(scene_intro, 120), (scene_wrap, 60),
                   (scene_splay, 60), (scene_collapse_to_zstack, 66)]:
        fn(s, frames_dir, cursor, n); cursor += n
    return cursor
