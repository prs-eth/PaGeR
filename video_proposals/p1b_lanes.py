"""Variant B — Stacked Lanes: 4 horizontal lanes painted left-to-right by a moving wave."""
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


def scene_lanes(s, frames_dir, start, dur):
    panels = make_head_panels(s)
    # 4 lanes — top to bottom on the right side
    lane_x0 = STACK_RIGHT + 90
    lane_x1 = W - 50
    lane_w = lane_x1 - lane_x0
    lane_top = 80
    lane_h = (H - lane_top - 80 - 3 * 14) // 4
    gap = 14
    positions = [(lane_x0, lane_top + k * (lane_h + gap)) for k in range(4)]

    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        # redraw stack
        faces = s.cubemap
        for k in range(6):
            face = (np.transpose(faces[k], (1, 2, 0)) * 255).astype(np.uint8)
            face = cv2.resize(face, (THUMB_W, THUMB_W), interpolation=cv2.INTER_AREA)
            y = STACK_TOP_Y + k * (THUMB_W + 4)
            f = paste(f, face, STACK_X, y)
            cv2.rectangle(f, (STACK_X, y), (STACK_X + THUMB_W, y + THUMB_W),
                          (110, 115, 130), 1)
        # ViT-Giant label above the lane area
        f = draw_text(f, "ViT-Giant", (lane_x0 + lane_w // 2, 50),
                      size=16, color=TEXT_LIGHT, bold=True, anchor="mm")
        f = draw_text(f, "→ four heads", (lane_x0 + lane_w // 2, 72),
                      size=12, color=TEXT_DIM, anchor="mm")
        # for each lane: draw frame, then progressively reveal panel left→right via a sweep
        for k, (lx, ly) in enumerate(positions):
            # bezier-ish connector from stack edge to lane left
            cv2.line(f, (STACK_RIGHT + 8, STACK_CY),
                     (lane_x0 - 14, ly + lane_h // 2),
                     HEAD_COLORS[k], 1)
            # lane background
            cv2.rectangle(f, (lx, ly), (lx + lane_w, ly + lane_h), (22, 24, 30), -1)
            cv2.rectangle(f, (lx, ly), (lx + lane_w, ly + lane_h), (60, 65, 80), 1)
            cv2.rectangle(f, (lx, ly), (lx + 5, ly + lane_h), HEAD_COLORS[k], -1)
            f = draw_text(f, HEAD_LABELS[k], (lx + 14, ly + 4), size=13,
                          color=TEXT_LIGHT, bold=True)
            f = draw_text(f, HEAD_SUBS[k], (lx + lane_w - 12, ly + 6), size=11,
                          color=TEXT_DIM, anchor="rt")
            # progressively-painted content: stagger lanes a touch
            stagger = k * 0.06
            sweep = max(0.0, min(1.0, (e - stagger) / max(1e-3, 0.85 - stagger)))
            if sweep <= 0: continue
            content_w = lane_w - 14
            content_h = lane_h - 26
            cont = cv2.resize(panels[k], (content_w, content_h), interpolation=cv2.INTER_AREA)
            reveal_w = int(content_w * sweep)
            if reveal_w > 0:
                px = lx + 7; py = ly + 22
                f[py:py + content_h, px:px + reveal_w] = cont[:, :reveal_w]
                # sweep edge glow
                if reveal_w < content_w:
                    cv2.line(f, (px + reveal_w, py), (px + reveal_w, py + content_h),
                             HEAD_COLORS[k], 2)
        f = draw_text(f, "4.  Four heads, painted in parallel",
                      (60, H - 50), size=22, color=TEXT_DIM,
                      alpha=min(1.0, e * 2))
        save_frame(f, frames_dir, start + i)


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1b_lanes")
    cursor = run_prologue(s, frames_dir)
    n_branch = 130
    scene_lanes(s, frames_dir, cursor, n_branch); cursor += n_branch
    # short hold
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
    write_video(frames_dir, OUT / "p1b_lanes.mp4", fps=FPS)
    print(f"✓ wrote p1b_lanes.mp4  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
