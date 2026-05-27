"""New post-prologue choreography for the iterated p1.

Sequence after the centred Z-stack lands:
    1.  scene_stack_spin               — the 6 cards orbit the centre and re-converge.
    2.  scene_pop_to_modalities        — 4 mini cubemap-stacks emerge at top/right/bottom/left,
                                          each one a Z-stack of cubemap faces in the modality's colour.
    3.  scene_unfold_to_erps           — each mini stack splays + morphs into a horizontal ERP strip.
    4.  scene_blend_to_pc              — the 4 ERPs converge to the centre, blend, and the
                                          point cloud blooms outward.
    5.  scene_pc_orbit (from shared)   — long orbit reveal with PaGeR fading in/out.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from pathlib import Path
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     save_frame, write_video, fresh_frames_dir,
                     make_coarse_metric_depth, make_sky_viz,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT, ROOT)
from p1_shared import (W, H, FPS,
                        COLOR_DEPTH, COLOR_METRIC, COLOR_NORMALS, COLOR_SKY,
                        HEAD_LABELS)

# ────────────────────────────────────────────────────────────────────
# Modality-cubemap precomputation (uses the project's own erp_to_cubemap)
# ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from src.utils.geometry_utils import erp_to_cubemap


def erp_uint8_to_cube_faces(erp: np.ndarray, face_w: int = 504, fov: float = 90.0) -> np.ndarray:
    """Convert an (H, W, 3) uint8 ERP to 6 cubemap faces, (6, face_w, face_w, 3) uint8."""
    if erp.ndim == 2:
        erp = np.stack([erp, erp, erp], axis=-1)
    t = torch.from_numpy(erp).permute(2, 0, 1).to(torch.float32) / 255.0 * 2 - 1
    cube = erp_to_cubemap(t, face_w=face_w, fov=fov)              # (6, 3, fw, fw)
    faces = cube.permute(0, 2, 3, 1).numpy()
    faces = ((faces + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return faces


def build_modality_cubemaps(s, face_w: int = 256) -> dict[str, np.ndarray]:
    """Return per-modality cubemap thumbnails: {name: (6, fw, fw, 3) uint8}."""
    coarse_metric = make_coarse_metric_depth(s.depth_raw, s.sky_mask,
                                              downsample=4, blur_sigma=1.0)
    sky_rgb = make_sky_viz(s.sky_mask)
    return {
        "depth":   erp_uint8_to_cube_faces(s.depth_viz,   face_w=face_w),
        "metric":  erp_uint8_to_cube_faces(coarse_metric, face_w=face_w),
        "normals": erp_uint8_to_cube_faces(s.normals_viz, face_w=face_w),
        "sky":     erp_uint8_to_cube_faces(sky_rgb,       face_w=face_w),
    }


def build_modality_erps(s) -> dict[str, np.ndarray]:
    """ERP-form output for each head."""
    return {
        "depth":   s.depth_viz,
        "metric":  make_coarse_metric_depth(s.depth_raw, s.sky_mask,
                                             downsample=10, blur_sigma=4.0),
        "normals": s.normals_viz,
        "sky":     make_sky_viz(s.sky_mask),
    }


MODALITY_ORDER = ["depth", "metric", "normals", "sky"]
MODALITY_COLORS = {
    "depth":   COLOR_DEPTH,
    "metric":  COLOR_METRIC,
    "normals": COLOR_NORMALS,
    "sky":     COLOR_SKY,
}
MODALITY_DISPLAY = {
    "depth":   "DEPTH",
    "metric":  "METRIC DEPTH",
    "normals": "NORMALS",
    "sky":     "SKY",
}

# Corner anchor positions (centres) for the 4 modality stacks.
EDGE_POSITIONS = {
    "depth":   (210,        130),    # top-left
    "metric":  (W - 210,    130),    # top-right
    "normals": (210,        H - 130),# bottom-left
    "sky":     (W - 210,    H - 130),# bottom-right
}


# ────────────────────────────────────────────────────────────────────
# Helpers — render a single Z-stack at (cx, cy) given a faces array
# ────────────────────────────────────────────────────────────────────
def draw_zstack(canvas: np.ndarray, faces: np.ndarray, cx: int, cy: int,
                 front_w: int = 230, dx_off: int = 11, dy_off: int = -11,
                 scale_decay: float = 0.95, alpha: float = 1.0,
                 darken_back: float = 0.45, border=(110, 115, 130),
                 angle: float = 0.0) -> np.ndarray:
    """Draw 6 cubemap faces as a Z-stack centred at (cx, cy).
    ``angle`` rotates each card's offset around (cx, cy) in image-plane radians,
    so the stack can be 'spun' without rotating the card textures themselves."""
    H_, W_ = canvas.shape[:2]
    ca, sa = np.cos(angle), np.sin(angle)
    # render back-to-front
    for k in reversed(range(6)):
        w = max(8, int(round(front_w * (scale_decay ** k))))
        # apply 2D rotation to the offsets around the centre
        ox = k * dx_off; oy = k * dy_off
        rx = ox * ca - oy * sa
        ry = ox * sa + oy * ca
        x = int(cx + rx - w / 2); y = int(cy + ry - w / 2)
        # face_src must be (fw, fw, 3) uint8 for cv2.resize
        face_k = faces[k]
        if face_k.ndim == 3 and face_k.shape[0] == 3 and face_k.shape[-1] != 3:
            # channels-first float (3, fw, fw) — convert
            face_src = (np.transpose(face_k, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        else:
            face_src = face_k.astype(np.uint8) if face_k.dtype != np.uint8 else face_k
        thumb = cv2.resize(face_src, (w, w), interpolation=cv2.INTER_AREA)
        shade = 1.0 - (k / 5.0) * darken_back
        thumb = (thumb.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)
        canvas = paste(canvas, thumb, x, y, alpha=alpha)
        cv2.rectangle(canvas, (x, y), (x + w, y + w),
                      tuple(int(c * shade) for c in border), 1)
    return canvas


# ────────────────────────────────────────────────────────────────────
# Scene 1: stack spins around the centre
# ────────────────────────────────────────────────────────────────────
def scene_stack_processing(s, frames_dir, start, dur, *,
                            cx: int = 640, cy: int = 360, front_w: int = 230):
    """'Processing' state — the Z-stack stays in place. A soft glowing halo
    pulses behind it (3 pulses) and 'tokens' stream inward from random points
    around the canvas, getting absorbed into the stack. Conveys active processing
    without a literal rotation."""
    faces = s.cubemap
    rng = np.random.default_rng(7)
    n_tokens = 80
    angles  = rng.uniform(0, 2 * np.pi, n_tokens)
    phases  = rng.uniform(0, 1, n_tokens)
    speeds  = rng.uniform(0.5, 1.0, n_tokens)
    # absorbed radius — once a token gets this close, it disappears into the stack
    R_absorb = front_w * 0.52
    # spawn radius
    R_spawn = front_w * 1.9

    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)

        # Soft halo pulse (1.5 pulses over the scene)
        pulse = 0.5 - 0.5 * np.cos(t * np.pi * 3.0)
        halo_layer = np.zeros_like(f)
        halo_r = int(front_w * 0.55 + 70 * pulse)
        glow_intensity = pulse * 0.55
        glow_col = tuple(int(c * glow_intensity) for c in (140, 200, 255))
        cv2.circle(halo_layer, (cx, cy), halo_r, glow_col, -1)
        halo_layer = cv2.GaussianBlur(halo_layer, (0, 0), sigmaX=42)
        f = cv2.add(f, halo_layer)

        # Tokens streaming inward
        for k in range(n_tokens):
            phase = (t * speeds[k] + phases[k]) % 1.0
            radius = R_spawn - phase * (R_spawn - R_absorb)
            if radius < R_absorb:
                continue
            ang = angles[k]
            sx_ = int(cx + radius * np.cos(ang))
            sy_ = int(cy + radius * np.sin(ang))
            a = phase                                     # brighter as it nears the stack
            r_dot = 1 + int(a * 2)
            color = tuple(int(c * a) for c in (200, 220, 255))
            cv2.circle(f, (sx_, sy_), r_dot, color, -1)
            # short motion trail behind it
            trail_r = radius + 12
            tx_ = int(cx + trail_r * np.cos(ang))
            ty_ = int(cy + trail_r * np.sin(ang))
            tcol = tuple(int(c * a * 0.4) for c in (160, 200, 255))
            cv2.line(f, (sx_, sy_), (tx_, ty_), tcol, 1, lineType=cv2.LINE_AA)

        # Stack — barely-perceptible scale pulse synced to the halo
        pulse_scale = 1.0 + 0.022 * np.sin(t * np.pi * 3.0)
        cur_w = int(front_w * pulse_scale)
        f = draw_zstack(f, faces, cx, cy, front_w=cur_w)

        save_frame(f, frames_dir, start + i)


# Back-compat alias used by p1_full_test (renamed semantically).
scene_stack_spin = scene_stack_processing


# ────────────────────────────────────────────────────────────────────
# Scene 2: 4 mini cubemap-stacks pop out to the 4 edges
# ────────────────────────────────────────────────────────────────────
def scene_pop_to_modalities(s, frames_dir, start, dur, *,
                             cx: int = 640, cy: int = 360,
                             central_front_w: int = 230,
                             edge_front_w: int = 120):
    """4 mini cubemap-stacks (one per modality) emerge from the central RGB stack
    and travel to their edge anchor positions, scaling down as they arrive."""
    rgb_faces = s.cubemap
    mod_cubes = build_modality_cubemaps(s, face_w=256)
    # stagger so the four don't move in perfect lockstep
    staggers = {"depth": 0.00, "metric": 0.08, "normals": 0.16, "sky": 0.24}

    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)
        # central RGB stack — shrinks and fades fully out by the end of pop
        cf_t = ease_in_out(t)
        central_w = int(central_front_w * (1.0 - 0.45 * cf_t))
        central_a = max(0.0, 1.0 - 1.05 * cf_t)
        if central_a > 0.02:
            f = draw_zstack(f, rgb_faces, cx, cy, front_w=central_w, alpha=central_a)
        # 4 modality stacks
        for name in MODALITY_ORDER:
            st = staggers[name]
            ke = max(0.0, min(1.0, (t - st) / max(1e-3, 1.0 - st)))
            if ke <= 0: continue
            e = ease_in_out(ke)
            tx, ty = EDGE_POSITIONS[name]
            mx = int(cx + (tx - cx) * e); my = int(cy + (ty - cy) * e)
            cur_w = int(central_front_w * 0.85 + (edge_front_w - central_front_w * 0.85) * e)
            cube_faces = mod_cubes[name]      # (6, fw, fw, 3) uint8
            # alpha rises with arrival
            a = min(1.0, e * 1.5)
            f = draw_zstack(f, cube_faces, mx, my,
                              front_w=cur_w, alpha=a)
            # tiny modality label fades in once arrived
            if e > 0.75:
                la = (e - 0.75) / 0.25
                lab = MODALITY_DISPLAY[name]
                col = MODALITY_COLORS[name]
                label_pos = (mx, my + cur_w // 2 + 22)
                f = draw_text(f, lab, label_pos, size=14, color=col,
                                bold=True, anchor="mm", alpha=la)
        save_frame(f, frames_dir, start + i)


# ────────────────────────────────────────────────────────────────────
# Scene 3: each mini cubemap-stack unfolds into its ERP strip
# ────────────────────────────────────────────────────────────────────
ERP_STRIP_W = 320
ERP_STRIP_H = ERP_STRIP_W // 2          # 2:1 aspect


# ────────────────────────────────────────────────────────────────────
# Cubemap geometry — face UV ↔ 3D ray ↔ ERP UV (truthful projection)
# ────────────────────────────────────────────────────────────────────
# Cubemap face order in the project: [F, R, B, L, U, D].
# F looks +Z, R +X, B -Z, L -X, U +Y, D -Y.
# We use a right-handed world frame with +Y up, image y down inside each face.
def _face_uv_to_dir(face_idx: int, uu: np.ndarray, vv: np.ndarray):
    """Per-face UV in [0,1] → 3D ray direction (unnormalised)."""
    a = 2 * uu - 1; b = -(2 * vv - 1)
    if face_idx == 0:   return a,             b,            np.ones_like(a)  # F  +Z
    if face_idx == 1:   return np.ones_like(a), b,           -a              # R  +X
    if face_idx == 2:   return -a,            b,           -np.ones_like(a)  # B  -Z
    if face_idx == 3:   return -np.ones_like(a), b,         a                # L  -X
    if face_idx == 4:   return a,             np.ones_like(a), -b            # U  +Y
    if face_idx == 5:   return a,             -np.ones_like(a),  b           # D  -Y
    raise ValueError(face_idx)


def _face_uv_to_erp_uv(face_idx: int, uu: np.ndarray, vv: np.ndarray):
    """Each face-UV sample → its (eu, ev) location on the ERP — the truthful
    projection. Wraps in longitude (B, U, D) are left to fall out as a split
    near the lon=±π seam, which is what physically happens."""
    dx, dy, dz = _face_uv_to_dir(face_idx, uu, vv)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    lon = np.arctan2(dx, dz)                       # [-π, π]
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))        # [-π/2, π/2]
    eu = (lon + np.pi) / (2 * np.pi)
    ev = (np.pi / 2 - lat) / np.pi
    return eu, ev


# Cross-layout offsets (units of one face-size cell), centred so F sits over the anchor.
CROSS_OFFSETS = {
    0: (0,  0),    # F (centre)
    1: (1,  0),    # R (right)
    2: (2,  0),    # B (far right — but in ERP it wraps to the other side)
    3: (-1, 0),    # L (left)
    4: (0, -1),    # U (above)
    5: (0,  1),    # D (below)
}


def _erp_strip_box(name: str, *, scale: float = 1.0) -> tuple[int, int, int, int]:
    """Return the (x, y, w, h) of the modality's ERP strip at its edge anchor."""
    cx, cy = EDGE_POSITIONS[name]
    w = int(ERP_STRIP_W * scale); h = int(ERP_STRIP_H * scale)
    x = cx - w // 2; y = cy - h // 2
    return x, y, w, h


def scene_unfold_to_erps(s, frames_dir, start, dur, *,
                          cx: int = 640, cy: int = 360,
                          central_front_w: int = 230,
                          edge_front_w: int = 120):
    """Each mini cubemap-stack splays its faces then settles into an ERP strip.
    Implemented as a cross-fade from the Z-stack to the ERP image at the same anchor."""
    cube_face_w_for_render = 128
    mod_cubes = build_modality_cubemaps(s, face_w=cube_face_w_for_render)
    mod_erps  = build_modality_erps(s)

    # Per-face sample grid. Each face has Nu×Nv points; each point morphs from
    # a position on the Z-stack → unfolded-cube cross → ERP (via real projection).
    Nu, Nv = 44, 44
    cross_face_size = 34
    u_lin = np.linspace(0.0, 1.0, Nu)
    v_lin = np.linspace(0.0, 1.0, Nv)
    uu, vv = np.meshgrid(u_lin, v_lin)
    u_pix = (uu * (cube_face_w_for_render - 1)).astype(np.int32)
    v_pix = (vv * (cube_face_w_for_render - 1)).astype(np.int32)

    # Per-face precomputed cross and ERP offsets (relative to the corner anchor)
    face_info = []
    for k in range(6):
        ox, oy = CROSS_OFFSETS[k]
        cross_dx = (ox + (uu - 0.5)) * cross_face_size
        cross_dy = (oy + (vv - 0.5)) * cross_face_size
        eu, ev = _face_uv_to_erp_uv(k, uu, vv)
        erp_dx = (eu - 0.5) * ERP_STRIP_W
        erp_dy = (ev - 0.5) * ERP_STRIP_H
        face_info.append((cross_dx, cross_dy, erp_dx, erp_dy))

    # Z-stack offsets matching scene_pop's mini-stack final state
    z_dx_per_k, z_dy_per_k = 11, -11
    z_decay = 0.95

    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)

        # Phases:
        #   stack→cross  : t ∈ [0.05, 0.35]     (Z-stack opens into the cube cross)
        #   cross→ERP    : t ∈ [0.30, 0.82]     (each face deforms into its ERP region)
        #   cleanup      : t ∈ [0.78, 1.00]     (warped output cross-fades into clean ERP)
        stack_to_cross = ease_in_out(min(1.0, max(0.0, (t - 0.05) / 0.30)))
        cross_to_erp   = ease_in_out(min(1.0, max(0.0, (t - 0.30) / 0.52)))
        cleanup        = ease_in_out(min(1.0, max(0.0, (t - 0.78) / 0.22)))

        for name in MODALITY_ORDER:
            ax, ay = EDGE_POSITIONS[name]
            cube_faces = mod_cubes[name]            # (6, fw, fw, 3) uint8
            erp_img    = mod_erps[name]

            # ── Clean ERP underneath fades in during cleanup ──
            if cleanup > 0.02:
                ex, ey, ew_, eh_ = _erp_strip_box(name)
                pnl = np.full((eh_ + 8, ew_ + 8, 3), (22, 24, 30), dtype=np.uint8)
                cv2.rectangle(pnl, (0, 0), (ew_ + 7, eh_ + 7), (60, 65, 80), 1)
                cv2.rectangle(pnl, (0, 0), (4, eh_ + 7), MODALITY_COLORS[name], -1)
                f = paste(f, pnl, ex - 4, ey - 4, alpha=cleanup)
                strip = cv2.resize(erp_img, (ew_, eh_), interpolation=cv2.INTER_AREA)
                f = paste(f, strip, ex, ey, alpha=cleanup)
                f = draw_text(f, MODALITY_DISPLAY[name], (ex + 4, ey - 22),
                                size=14, color=MODALITY_COLORS[name],
                                bold=True, alpha=cleanup)

            # ── Warped cube faces (rendered back-to-front for the Z-stack to read) ──
            warped_alpha = 1.0 - cleanup
            if warped_alpha < 0.02:
                continue

            for k in reversed(range(6)):
                cross_dx_k, cross_dy_k, erp_dx_k, erp_dy_k = face_info[k]

                # Z-stack offsets (Square of size z_size, shifted by k*offset)
                z_size = edge_front_w * (z_decay ** k)
                z_dx = (uu - 0.5) * z_size + k * z_dx_per_k
                z_dy = (vv - 0.5) * z_size + k * z_dy_per_k

                # Two-stage interpolation
                s1_dx = (1 - stack_to_cross) * z_dx + stack_to_cross * cross_dx_k
                s1_dy = (1 - stack_to_cross) * z_dy + stack_to_cross * cross_dy_k
                cur_dx = (1 - cross_to_erp) * s1_dx + cross_to_erp * erp_dx_k
                cur_dy = (1 - cross_to_erp) * s1_dy + cross_to_erp * erp_dy_k

                px = (ax + cur_dx).astype(np.int32).ravel()
                py = (ay + cur_dy).astype(np.int32).ravel()

                face_img = cube_faces[k]
                if face_img.ndim == 3 and face_img.shape[0] == 3:
                    face_img = (np.transpose(face_img, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                colors = face_img[v_pix, u_pix].reshape(-1, 3)

                # shade back cards while still in Z-stack form (so depth reads); fades out as we move to cross
                shade = 1.0 - (1.0 - stack_to_cross) * (k / 5.0) * 0.45
                if shade < 0.999:
                    colors = (colors.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)

                # Painter splat (back-to-front via outer k loop)
                r = 2
                for ddy in range(-r, r + 1):
                    for ddx in range(-r, r + 1):
                        if ddx * ddx + ddy * ddy > r * r: continue
                        yy = py + ddy; xx = px + ddx
                        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                        f[yy[ok], xx[ok]] = colors[ok]

        save_frame(f, frames_dir, start + i)


# ────────────────────────────────────────────────────────────────────
# Scene 4: the 4 ERPs converge to the centre, blend → point cloud bloom
# ────────────────────────────────────────────────────────────────────
def scene_blend_to_dark(s, frames_dir, start, dur, *,
                         cx: int = 640, cy: int = 360):
    """4 ERPs slide to the centre, blend, then fade fully to black.
    No PC bloom here — the PC scene handles the entire reveal."""
    mod_erps = build_modality_erps(s)
    target_w = 420; target_h = target_w // 2
    target_x = cx - target_w // 2; target_y = cy - target_h // 2

    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)
        be = ease_in_out(min(1.0, t / 0.55))
        acc = np.zeros((H, W, 3), dtype=np.float32)
        count = np.zeros((H, W, 1), dtype=np.float32)
        for name in MODALITY_ORDER:
            sx, sy, sw, sh = _erp_strip_box(name)
            ex = target_x; ey = target_y; ew = target_w; eh = target_h
            cur_x = int(sx + (ex - sx) * be)
            cur_y = int(sy + (ey - sy) * be)
            cur_w = int(sw + (ew - sw) * be)
            cur_h = int(sh + (eh - sh) * be)
            strip = cv2.resize(mod_erps[name], (cur_w, cur_h),
                                interpolation=cv2.INTER_AREA)
            x0 = max(0, cur_x); y0 = max(0, cur_y)
            x1 = min(W, cur_x + cur_w); y1 = min(H, cur_y + cur_h)
            if x1 <= x0 or y1 <= y0:
                continue
            sx0 = x0 - cur_x; sy0 = y0 - cur_y
            acc[y0:y1, x0:x1] += strip[sy0:sy0 + (y1 - y0),
                                         sx0:sx0 + (x1 - x0)].astype(np.float32)
            count[y0:y1, x0:x1] += 1
        mask = (count[..., 0] > 0)
        blended = np.zeros_like(acc)
        blended[mask] = acc[mask] / count[mask]
        # darken-to-black during the last 40% of the scene
        dark_t = max(0.0, min(1.0, (t - 0.55) / 0.45))
        scale = 1.0 - ease_in_out(dark_t)
        blended = (blended.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
        f = paste(f, blended, 0, 0, alpha=1.0)
        save_frame(f, frames_dir, start + i)


# Back-compat alias
scene_blend_to_pc = scene_blend_to_dark
