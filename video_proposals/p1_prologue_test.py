"""Prologue-only render for review: intro → wrap → splay → Z-axis cubemap stack, then hold."""
from __future__ import annotations
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vutils import write_video, fresh_frames_dir, load_sample, OUT
from p1_shared import W, H, FPS, run_prologue

HERO = "eth_campus_plaza"


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1_prologue_test")
    cursor = run_prologue(s, frames_dir)
    # hold the final frame for ~2 seconds so the Z-stack reads clearly
    last = cursor - 1
    n_hold = 48
    for i in range(n_hold):
        shutil.copy(frames_dir / f"f_{last:05d}.png",
                    frames_dir / f"f_{cursor + i:05d}.png")
    cursor += n_hold
    write_video(frames_dir, OUT / "p1_prologue_test.mp4", fps=FPS)
    print(f"✓ wrote p1_prologue_test.mp4  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
