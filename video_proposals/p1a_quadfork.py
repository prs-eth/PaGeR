"""Variant A — Quadfork: stack on the left, 4 data lines fork out to 4 head panels on the right."""
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

STACK_X = 110
STACK_TOP_Y = 80
THUMB_W = 90
# total stack height
STACK_H = 6 * THUMB_W + 5 * 4
STACK_CY = STACK_TOP_Y + STACK_H // 2
STACK_RIGHT = STACK_X + THUMB_W


def scene_quadfork(s, frames_dir, start, dur):
    panels = make_head_panels(s)
    pw, ph = 360, 175                       # 2x2 grid panels
    gap_x, gap_y = 18, 18
    grid_left = W - 2 * pw - gap_x - 50
    grid_top = (H - 2 * ph - gap_y) // 2 - 10
    positions = [
        (grid_left,                grid_top),
        (grid_left + pw + gap_x,   grid_top),
        (grid_left,                grid_top + ph + gap_y),
        (grid_left + pw + gap_x,   grid_top + ph + gap_y),
    ]
    junction = (grid_left - 50, STACK_CY)
    # endpoints: left edge of each panel
    ends = [(p[0], p[1] + ph // 2) for p in positions]

    for i in range(dur):
        t = i / max(dur - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        # redraw the stack (held position)
        faces = s.cubemap
        for k in range(6):
            face = (np.transpose(faces[k], (1, 2, 0)) * 255).astype(np.uint8)
            face = cv2.resize(face, (THUMB_W, THUMB_W), interpolation=cv2.INTER_AREA)
            y = STACK_TOP_Y + k * (THUMB_W + 4)
            f = paste(f, face, STACK_X, y, alpha=1.0)
            cv2.rectangle(f, (STACK_X, y), (STACK_X + THUMB_W, y + THUMB_W),
                          (110, 115, 130), 1)
        # backbone "trunk" line from stack to junction
        trunk_t = min(1.0, e * 3.0)
        trunk_end_x = int(STACK_RIGHT + 8 + (junction[0] - (STACK_RIGHT + 8)) * trunk_t)
        cv2.line(f, (STACK_RIGHT + 8, STACK_CY), (trunk_end_x, STACK_CY),
                 (180, 185, 195), 2)
        # ViT label above the trunk
        f = draw_text(f, "ViT-Giant", ((STACK_RIGHT + junction[0]) // 2, STACK_CY - 20),
                      size=14, color=TEXT_LIGHT, bold=True, anchor="mm",
                      alpha=min(1.0, e * 2))
        # junction dot
        if e > 0.18:
            cv2.circle(f, junction, 5, (220, 220, 220), -1)
        # 4 fork lines + panels
        for k in range(4):
            start_e = 0.18 + k * 0.05
            ke = max(0.0, min(1.0, (e - start_e) / 0.4))
            if ke <= 0: continue
            ex, ey = ends[k]
            # bezier-ish: 3-segment path junction → control1 → endpoint
            mid_x = (junction[0] + ex) // 2
            mid_y = junction[1] + int((ey - junction[1]) * ke)
            # draw two segments
            cv2.line(f, junction, (mid_x, junction[1]),
                     HEAD_COLORS[k], 2)
            cv2.line(f, (mid_x, junction[1]), (mid_x, mid_y),
                     HEAD_COLORS[k], 2)
            if ke > 0.5:
                # arrive at panel
                arrive = (ex, ey)
                cv2.line(f, (mid_x, ey), arrive, HEAD_COLORS[k], 2)
                # token dots along the path (animated)
                phase = (t * 2 + k * 0.13) % 1.0
                dx = int(mid_x + (ex - mid_x) * phase)
                cv2.circle(f, (dx, ey), 3, HEAD_COLORS[k], -1)
            # panel content
            panel_a = max(0.0, min(1.0, (ke - 0.45) * 2.5))
            if panel_a <= 0: continue
            x, y = positions[k]
            cv2.rectangle(f, (x, y), (x + pw, y + ph), (22, 24, 30), -1)
            cv2.rectangle(f, (x, y), (x + pw, y + ph), (60, 65, 80), 1)
            cv2.rectangle(f, (x, y), (x + 5, y + ph), HEAD_COLORS[k], -1)
            # content
            cont = cv2.resize(panels[k], (pw - 14, ph - 36), interpolation=cv2.INTER_AREA)
            f = paste(f, cont, x + 7, y + 28, alpha=panel_a)
            f = draw_text(f, HEAD_LABELS[k], (x + 14, y + 6), size=14,
                          color=TEXT_LIGHT, bold=True, alpha=panel_a)
            f = draw_text(f, HEAD_SUBS[k], (x + pw - 14, y + 9), size=12,
                          color=TEXT_DIM, anchor="rt", alpha=panel_a * 0.9)
        # caption
        f = draw_text(f, "4.  Four heads share a single backbone",
                      (60, H - 50), size=22, color=TEXT_DIM,
                      alpha=min(1.0, e * 2))
        save_frame(f, frames_dir, start + i)


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1a_quadfork")
    cursor = run_prologue(s, frames_dir)
    print(f"  prologue done: cursor={cursor}")
    n_branch = 120
    scene_quadfork(s, frames_dir, cursor, n_branch); cursor += n_branch
    print(f"  branch done: cursor={cursor}")
    # hold the branch frame to let the eye settle
    n_hold = 18
    last = cursor - 1
    for i in range(n_hold):
        os_path = frames_dir / f"f_{last:05d}.png"
        new_path = frames_dir / f"f_{cursor + i:05d}.png"
        # symlink/copy: simpler — re-render? Just copy bytes.
        import shutil; shutil.copy(os_path, new_path)
    cursor += n_hold
    # PC orbit (longer — user requested)
    n_pc = 168
    scene_pc_orbit(s, frames_dir, cursor, n_pc,
                    initial_d=2.0, intro_blackin=14, title_start_t=0.55)
    cursor += n_pc
    write_video(frames_dir, OUT / "p1a_quadfork.mp4", fps=FPS)
    print(f"✓ wrote {OUT / 'p1a_quadfork.mp4'}  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
