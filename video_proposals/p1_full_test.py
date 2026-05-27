"""Full p1 sequence (revised): prologue → spin → 4 cubemap-stack outputs → 4 ERPs → point cloud."""
from __future__ import annotations
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vutils import write_video, fresh_frames_dir, load_sample, OUT
from p1_shared import W, H, FPS, run_prologue, scene_pc_orbit
from p1_sequence import (scene_stack_processing, scene_pop_to_modalities,
                          scene_unfold_to_erps, scene_blend_to_dark)

HERO = "eth_campus_plaza"


def main():
    s = load_sample(HERO)
    frames_dir = fresh_frames_dir("p1_full_test")
    cursor = run_prologue(s, frames_dir)
    print(f"  prologue done @ frame {cursor}")
    # processing: stack in place with pulse + tokens streaming in
    n = 60
    scene_stack_processing(s, frames_dir, cursor, n); cursor += n
    print(f"  process done @ frame {cursor}")
    # pop: 4 modality cubemap-stacks emerge at 4 corners
    n = 60
    scene_pop_to_modalities(s, frames_dir, cursor, n); cursor += n
    print(f"  pop done     @ frame {cursor}")
    # unfold: Z-stack splays into row, row melts into ERP
    n = 90
    scene_unfold_to_erps(s, frames_dir, cursor, n); cursor += n
    print(f"  unfold done  @ frame {cursor}")
    # blend (no PC bloom here — PC scene handles the full reveal)
    n = 66
    scene_blend_to_dark(s, frames_dir, cursor, n); cursor += n
    print(f"  blend done   @ frame {cursor}")
    # single long PC reveal: bloom (close camera) → pull back + orbit, PaGeR title in/out
    n = 240
    scene_pc_orbit(s, frames_dir, cursor, n,
                    initial_d=0.6, intro_blackin=18,
                    title_in_t=0.30, title_full_t=0.48,
                    title_out_t=0.88, title_gone_t=1.00)
    cursor += n
    print(f"  pc done      @ frame {cursor}")
    write_video(frames_dir, OUT / "p1_full_test.mp4", fps=FPS)
    print(f"✓ wrote p1_full_test.mp4  ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
