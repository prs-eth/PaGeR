"""Variant D — Prism: stack feeds a glowing prism; 4 colored beams refract out, each landing in a head panel."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, ease_in_out, ease_out, paste, resize,
                     write_video, fresh_frames_dir, save_frame,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT)
from p1_shared import (W, H, FPS, HEAD_LABELS, HEAD_SUBS, HEAD_COLORS,
                        make_head_panels, run_prologue, scene_pc_orbit)

HERO = "eth_campus_plaza"
STACK_X, STACK_TOP_Y, THUMB_W = 110, 80, 90
STACK_RIGHT = STACK_X + THUMB_W
STACK_H = 6 * THUMB_W + 5 * 4
STACK_CY = STACK_TOP_Y + STACK_H // 2


def draw_stack(f, faces):
    for k in range(6):
        face = (np.transpose(faces[k], (1, 2, 0)) * 255).astype(np.uint8)
        face = cv2.resize(face, (THUMB_W, THUMB_W), interpolation=cv2.INTER_AREA)
        y = STACK_TOP_Y + k * (THUMB_W + 4)
        f = paste(f, face, STACK_X, y)
        cv2.rectangle(f, (STACK_X, y), (STACK_X + THUMB_W, y + THUMB_W),
                      (110, 115, 130), 1)
    return f


def draw_beam(canvas, p1, p2, color, intensity=1.0, thickness=3):
    """Draw a soft glowing line: bright core + blurred halo, additively blended."""
    H_, W_ = canvas.shape[:2]
    layer = np.zeros_like(canvas)
    # halo (thick, blurred)
    cv2.line(layer, p1, p2, color, thickness * 6, lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=6)
    # core (thin, sharp)
    cv2.line(layer, p1, p2, (255, 255, 255), max(1, thickness - 1), lineType=cv2.LINE_AA)
    cv2.line(layer, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    out = canvas.astype(np.float32) + layer.astype(np.float32) * intensity
    return out.clip(0, 255).astype(np.uint8)


def scene_prism(s, frames_dir, start, dur):
    panels = make_head_panels(s)
    faces = s.cubemap
    # prism position (center)
    prism_cx = 540
    prism_cy = STACK_CY
    prism_r = 90
    prism_pts = np.array([
        [prism_cx,            prism_cy - prism_r],
        [prism_cx - prism_r,  prism_cy + prism_r // 2],
        [prism_cx + prism_r,  prism_cy + prism_r // 2],
    ], np.int32)
    # 4 panel positions (right side, 2x2)
    pw, ph = 360, 175
    gap_x, gap_y = 18, 18
    grid_left = W - 2 * pw - gap_x - 50
    grid_top = (H - 2 * ph - gap_y) // 2 - 10
    targets = [
        (grid_left,                grid_top),
        (grid_left + pw + gap_x,   grid_top),
        (grid_left,                grid_top + ph + gap_y),
        (grid_left + pw + gap_x,   grid_top + ph + gap_y),
    ]
    # exit points on the prism's right edge (top, upper-mid, lower-mid, bottom)
    exits = [
        (prism_cx + 35, prism_cy - prism_r // 2),
        (prism_cx + prism_r - 10, prism_cy - 10),
        (prism_cx + prism_r - 10, prism_cy + 20),
        (prism_cx + 35, prism_cy + prism_r // 2),
    ]
    # beam endpoints (left edge mid of each panel)
    beam_ends = [(p[0], p[1] + ph // 2) for p in targets]

    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        f = draw_stack(f, faces)

        # input beam from stack to prism (always on)
        in_beam_t = min(1.0, t * 3)
        in_end = (int(STACK_RIGHT + 8 + (prism_cx - prism_r - 8 - (STACK_RIGHT + 8)) * in_beam_t),
                   STACK_CY)
        f = draw_beam(f, (STACK_RIGHT + 8, STACK_CY), in_end, (220, 225, 235),
                       intensity=0.8, thickness=3)

        # prism outline (appears after input beam arrives)
        if t > 0.18:
            pa = min(1.0, (t - 0.18) / 0.12)
            tmp = f.copy()
            cv2.polylines(tmp, [prism_pts], True, (200, 205, 220), 2, lineType=cv2.LINE_AA)
            f = cv2.addWeighted(f, 1.0 - pa, tmp, pa, 0)
            f = draw_text(f, "ViT-Giant", (prism_cx, prism_cy + prism_r + 22),
                          size=14, color=TEXT_LIGHT, bold=True, anchor="mm", alpha=pa)

        # 4 refracted beams + panels
        for k in range(4):
            ks = 0.30 + k * 0.06
            ke = max(0.0, min(1.0, (t - ks) / 0.40))
            if ke <= 0: continue
            ex, ey = exits[k]
            tx, ty = beam_ends[k]
            cur_end = (int(ex + (tx - ex) * ease_in_out(ke)),
                        int(ey + (ty - ey) * ease_in_out(ke)))
            f = draw_beam(f, (ex, ey), cur_end, HEAD_COLORS[k],
                            intensity=1.0, thickness=2)
            # panel after the beam reaches
            if ke > 0.7:
                pa = (ke - 0.7) / 0.3
                x, y = targets[k]
                cv2.rectangle(f, (x, y), (x + pw, y + ph), (22, 24, 30), -1)
                cv2.rectangle(f, (x, y), (x + pw, y + ph), (60, 65, 80), 1)
                cv2.rectangle(f, (x, y), (x + 5, y + ph), HEAD_COLORS[k], -1)
                cont = cv2.resize(panels[k], (pw - 14, ph - 36),
                                    interpolation=cv2.INTER_AREA)
                f = paste(f, cont, x + 7, y + 28, alpha=pa)
                f = draw_text(f, HEAD_LABELS[k], (x + 14, y + 6), size=14,
                              color=TEXT_LIGHT, bold=True, alpha=pa)
                f = draw_text(f, HEAD_SUBS[k], (x + pw - 14, y + 9), size=12,
                              color=TEXT_DIM, anchor="rt", alpha=pa * 0.9)

        f = draw_text(f, "4.  One backbone, four refracted outputs",
                      (60, H - 50), size=22, color=TEXT_DIM,
                      alpha=min(1.0, t * 2))
        save_frame(f, frames_dir, start + i)


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1d_prism")
    cursor = run_prologue(s, frames_dir)
    n_branch = 140
    scene_prism(s, frames_dir, cursor, n_branch); cursor += n_branch
    n_hold = 18
    last = cursor - 1
    import shutil
    for i in range(n_hold):
        shutil.copy(frames_dir / f"f_{last:05d}.png",
                    frames_dir / f"f_{cursor + i:05d}.png")
    cursor += n_hold
    n_pc = 168
    scene_pc_orbit(s, frames_dir, cursor, n_pc,
                    initial_d=2.0, intro_blackin=14, title_start_t=0.55)
    cursor += n_pc
    write_video(frames_dir, OUT / "p1d_prism.mp4", fps=FPS)
    print(f"✓ wrote p1d_prism.mp4  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
