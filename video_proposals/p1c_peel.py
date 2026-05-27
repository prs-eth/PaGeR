"""Variant C — Layer Peel: 4 cards peel off the cubemap stack like rotating pages, landing in 4 quadrants."""
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


def draw_stack(f, faces, alpha=1.0):
    for k in range(6):
        face = (np.transpose(faces[k], (1, 2, 0)) * 255).astype(np.uint8)
        face = cv2.resize(face, (THUMB_W, THUMB_W), interpolation=cv2.INTER_AREA)
        y = STACK_TOP_Y + k * (THUMB_W + 4)
        f = paste(f, face, STACK_X, y, alpha=alpha)
        cv2.rectangle(f, (STACK_X, y), (STACK_X + THUMB_W, y + THUMB_W),
                      (110, 115, 130), 1)
    return f


def card_warp(card: np.ndarray, theta: float, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cheap pseudo-3D rotation around the vertical axis (squash x by cos(theta))."""
    h, w = card.shape[:2]
    sx = max(0.02, abs(np.cos(theta))) * scale
    sy = scale
    new_w = max(1, int(round(w * sx))); new_h = max(1, int(round(h * sy)))
    warped = cv2.resize(card, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # mirror if back-facing
    if np.cos(theta) < 0:
        warped = warped[:, ::-1]
    # shade by edge orientation (front-facing = brightest)
    shade = max(0.55, 0.55 + 0.45 * abs(np.cos(theta)))
    warped = (warped.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)
    return warped, np.array([new_w, new_h])


def scene_peel(s, frames_dir, start, dur):
    panels = make_head_panels(s)
    faces = s.cubemap
    # target landing positions (2x2 grid on the right)
    cw_target, ch_target = 360, 200
    gap_x, gap_y = 18, 24
    grid_left = W - 2 * cw_target - gap_x - 50
    grid_top = (H - 2 * ch_target - gap_y) // 2 - 5
    targets = [
        (grid_left,                grid_top),
        (grid_left + cw_target + gap_x,   grid_top),
        (grid_left,                grid_top + ch_target + gap_y),
        (grid_left + cw_target + gap_x,   grid_top + ch_target + gap_y),
    ]
    # source position (above stack, single page-shaped panel)
    src_w, src_h = 220, 130
    src_x = STACK_RIGHT + 40
    src_y = STACK_CY - src_h // 2
    # stagger each card's peel start
    card_starts = [0.00, 0.12, 0.24, 0.36]
    card_dur = 0.45

    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)
        f = draw_stack(f, faces)
        # connector arrow from stack to source-card position
        cv2.line(f, (STACK_RIGHT + 8, STACK_CY),
                 (src_x - 8, STACK_CY),
                 (180, 185, 195), 2)
        f = draw_text(f, "ViT-Giant", (STACK_RIGHT + 22, STACK_CY - 22),
                      size=14, color=TEXT_LIGHT, bold=True)
        # render the 4 peeling cards
        for k in range(4):
            ks = card_starts[k]
            ke = max(0.0, min(1.0, (t - ks) / card_dur))
            if ke <= 0: continue
            cont = panels[k]
            # the card's current position interpolates from source to target center
            target_cx = targets[k][0] + cw_target // 2
            target_cy = targets[k][1] + ch_target // 2
            src_cx = src_x + src_w // 2
            src_cy = src_y + src_h // 2
            ease = ease_in_out(ke)
            cur_cx = int(src_cx + (target_cx - src_cx) * ease)
            cur_cy = int(src_cy + (target_cy - src_cy) * ease)
            cur_w = int(src_w + (cw_target - src_w) * ease)
            cur_h = int(src_h + (ch_target - src_h) * ease)
            # rotate the card around y-axis: 0 → 2π over the journey
            theta = ke * (np.pi * 1.5) + (k * 0.6)
            card_thumb = cv2.resize(cont, (cur_w, cur_h), interpolation=cv2.INTER_AREA)
            warped, (ww, wh) = card_warp(card_thumb, theta)
            f = paste(f, warped, cur_cx - ww // 2, cur_cy - wh // 2, alpha=min(1.0, ke * 2))
            # at the end, settle into the panel frame
            if ke >= 0.95:
                a = (ke - 0.95) / 0.05
                tx, ty = targets[k]
                cv2.rectangle(f, (tx, ty), (tx + cw_target, ty + ch_target),
                              (60, 65, 80), 1)
                cv2.rectangle(f, (tx, ty), (tx + 5, ty + ch_target), HEAD_COLORS[k], -1)
                f = draw_text(f, HEAD_LABELS[k], (tx + 14, ty + 6), size=14,
                              color=TEXT_LIGHT, bold=True, alpha=a)
                f = draw_text(f, HEAD_SUBS[k], (tx + cw_target - 14, ty + 9),
                              size=12, color=TEXT_DIM, anchor="rt", alpha=a)
        f = draw_text(f, "4.  Four heads peel off a shared backbone",
                      (60, H - 50), size=22, color=TEXT_DIM,
                      alpha=min(1.0, t * 2))
        save_frame(f, frames_dir, start + i)


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1c_peel")
    cursor = run_prologue(s, frames_dir)
    n_branch = 140
    scene_peel(s, frames_dir, cursor, n_branch); cursor += n_branch
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
    write_video(frames_dir, OUT / "p1c_peel.mp4", fps=FPS)
    print(f"✓ wrote p1c_peel.mp4  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
