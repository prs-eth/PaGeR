"""Variant E — Carousel: a single large hero panel cycles through the 4 head outputs; thumbnail tray tracks the active one."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, ease_in_out, ease_out, blend,
                     paste, resize, write_video, fresh_frames_dir, save_frame,
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


def scene_carousel(s, frames_dir, start, dur):
    panels = make_head_panels(s)
    faces = s.cubemap
    # Hero panel on the right
    hero_x = STACK_RIGHT + 90
    hero_y = 70
    hero_w = W - hero_x - 60
    hero_h = H - hero_y - 160          # leave space for the tray
    # Thumbnail tray (4 small previews of the heads)
    tray_h = 80
    tray_y = hero_y + hero_h + 24
    tray_pad = 20
    tray_w_each = (hero_w - 3 * tray_pad) // 4
    tray_positions = [(hero_x + k * (tray_w_each + tray_pad), tray_y) for k in range(4)]

    # cycle: each head gets ~1.0s on stage, with cross-fades
    seg = 1.0 / 4                      # fraction of dur per head
    cross = 0.10                        # fraction of dur for cross-fade

    for i in range(dur):
        t = i / max(dur - 1, 1)
        f = blank(W, H)
        f = draw_stack(f, faces)
        # arrow from stack to hero
        cv2.line(f, (STACK_RIGHT + 8, STACK_CY),
                 (hero_x - 14, hero_y + hero_h // 2),
                 (180, 185, 195), 2)
        cv2.line(f, (STACK_RIGHT + 32, STACK_CY - 24),
                 (STACK_RIGHT + 60, STACK_CY - 24),
                 (90, 95, 110), 1)
        f = draw_text(f, "ViT-Giant", (STACK_RIGHT + 24, STACK_CY - 42),
                      size=14, color=TEXT_LIGHT, bold=True)

        # determine current head (k) and blend with next
        idx = min(3, int(t / seg))
        local_t = (t - idx * seg) / seg
        next_idx = min(3, idx + 1)
        # cross-fade in the final 'cross' fraction of each segment
        a = 0.0
        if local_t > (1 - cross) and idx < 3:
            a = (local_t - (1 - cross)) / cross

        # Hero panel rendering
        cv2.rectangle(f, (hero_x, hero_y), (hero_x + hero_w, hero_y + hero_h),
                      (22, 24, 30), -1)
        cv2.rectangle(f, (hero_x, hero_y), (hero_x + hero_w, hero_y + hero_h),
                      (60, 65, 80), 1)
        # left accent stripe (current head)
        cv2.rectangle(f, (hero_x, hero_y), (hero_x + 6, hero_y + hero_h),
                      HEAD_COLORS[idx], -1)
        # content (cross-fade)
        cur = cv2.resize(panels[idx], (hero_w - 16, hero_h - 44),
                          interpolation=cv2.INTER_AREA)
        if a > 0:
            nxt = cv2.resize(panels[next_idx], (hero_w - 16, hero_h - 44),
                              interpolation=cv2.INTER_AREA)
            shown = blend(cur, nxt, a)
            stripe_col = tuple(int(c1 * (1 - a) + c2 * a)
                                for c1, c2 in zip(HEAD_COLORS[idx], HEAD_COLORS[next_idx]))
            cv2.rectangle(f, (hero_x, hero_y), (hero_x + 6, hero_y + hero_h),
                          stripe_col, -1)
        else:
            shown = cur
        f = paste(f, shown, hero_x + 8, hero_y + 36)
        # label (slide-in)
        label_idx = next_idx if a > 0.5 else idx
        f = draw_text(f, HEAD_LABELS[label_idx],
                      (hero_x + 18, hero_y + 8), size=20,
                      color=TEXT_LIGHT, bold=True)
        f = draw_text(f, HEAD_SUBS[label_idx],
                      (hero_x + hero_w - 16, hero_y + 12), size=14,
                      color=TEXT_DIM, anchor="rt")

        # Thumbnail tray
        for k in range(4):
            tx, ty = tray_positions[k]
            cv2.rectangle(f, (tx, ty), (tx + tray_w_each, ty + tray_h),
                          (18, 20, 26), -1)
            active = (k == idx and a < 0.5) or (k == next_idx and a >= 0.5)
            border_col = HEAD_COLORS[k] if active else (60, 65, 80)
            cv2.rectangle(f, (tx, ty), (tx + tray_w_each, ty + tray_h),
                          border_col, 2 if active else 1)
            # mini thumb of the head output
            mini = cv2.resize(panels[k], (tray_w_each - 8, tray_h - 24),
                              interpolation=cv2.INTER_AREA)
            f = paste(f, mini, tx + 4, ty + 20,
                       alpha=1.0 if active else 0.55)
            f = draw_text(f, HEAD_LABELS[k], (tx + 6, ty + 2),
                          size=11, color=TEXT_LIGHT if active else TEXT_DIM,
                          bold=active)

        f = draw_text(f, "4.  Four predictions, one forward pass",
                      (60, H - 50), size=22, color=TEXT_DIM,
                      alpha=min(1.0, t * 2))
        save_frame(f, frames_dir, start + i)


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1e_carousel")
    cursor = run_prologue(s, frames_dir)
    # ~1s per head + cross-fades = 4*24 + some pad
    n_branch = 130
    scene_carousel(s, frames_dir, cursor, n_branch); cursor += n_branch
    n_hold = 12
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
    write_video(frames_dir, OUT / "p1e_carousel.mp4", fps=FPS)
    print(f"✓ wrote p1e_carousel.mp4  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
