"""Proposal 5: Tour of Worlds — fast-cut gallery across multiple panoramas.
Music-driven rhythm; each scene = ERP → depth flash → normals flash → point cloud orbit."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from vutils import (load_sample, blank, draw_text, blend, ease_in_out, ease_out,
                     paste, resize, project_pointcloud, autoframe_distance, center_points,
                     write_video, fresh_frames_dir, save_frame,
                     BG_DARK, TEXT_LIGHT, TEXT_DIM, ACCENT, OUT)

W, H, FPS = 1280, 720, 24

SCENES = [
    "medieval_kitchen",
    "eth_campus_plaza",
    "church_meeting_room",
    "zurich_street_corner",
    "livingroom_synth",
]


def render_block(s, frames_dir, start: int) -> int:
    """One scene block: erp zoom (16f) → depth (12f) → normals (12f) → pc orbit (40f)."""
    cursor = start

    P = center_points(s.points_xyz)
    d = autoframe_distance(P, fov_deg=55)

    def stretch(img):
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    erp_full = stretch(s.rgb)
    depth_full = stretch(s.depth_viz)
    norm_full = stretch(s.normals_viz)

    # 1. ERP zoom (slow zoom-in)
    n1 = 16
    for i in range(n1):
        t = i / max(n1 - 1, 1)
        sc = 1.0 + 0.10 * t
        cw = int(W * sc); ch = int(H * sc)
        big = cv2.resize(s.rgb, (cw, ch))
        crop = big[(ch - H) // 2:(ch - H) // 2 + H, (cw - W) // 2:(cw - W) // 2 + W]
        f = crop
        # tag with the modality
        f = draw_text(f, "RGB", (40, H - 50), size=22, color=TEXT_LIGHT,
                      bold=True, alpha=0.9)
        f = draw_text(f, s.name.replace("_", " "), (W - 40, H - 50),
                      size=18, color=TEXT_DIM, anchor="rt", alpha=0.85)
        save_frame(f, frames_dir, cursor + i)
    cursor += n1

    # 2. depth flash (cross-fade rgb→depth)
    n2 = 12
    for i in range(n2):
        t = i / max(n2 - 1, 1); e = ease_in_out(t)
        f = blend(erp_full, depth_full, e)
        f = draw_text(f, "DEPTH", (40, H - 50), size=22, color=ACCENT,
                       bold=True, alpha=0.9)
        save_frame(f, frames_dir, cursor + i)
    cursor += n2

    # 3. normals flash
    n3 = 12
    for i in range(n3):
        t = i / max(n3 - 1, 1); e = ease_in_out(t)
        f = blend(depth_full, norm_full, e)
        f = draw_text(f, "NORMALS", (40, H - 50), size=22, color=(140, 200, 255),
                       bold=True, alpha=0.9)
        save_frame(f, frames_dir, cursor + i)
    cursor += n3

    # 4. point cloud orbit
    n4 = 40
    for i in range(n4):
        t = i / max(n4 - 1, 1)
        yaw = -0.5 + ease_out(t) * 1.6
        pitch = 0.08 + 0.06 * np.sin(t * np.pi)
        img = project_pointcloud(P, s.points_rgb, canvas_w=W, canvas_h=H,
                                  yaw=yaw, pitch=pitch, distance=d, splat=2)
        # transition-in from normals: fade for first 6 frames
        if i < 6:
            a = i / 6
            img = blend(norm_full, img, a)
        img = draw_text(img, "POINT CLOUD", (40, H - 50), size=22, color=TEXT_LIGHT,
                          bold=True, alpha=0.9)
        save_frame(img, frames_dir, cursor + i)
    cursor += n4

    return cursor


def main():
    frames_dir = fresh_frames_dir("p5_tour")
    cursor = 0

    # ── opening sting (1.25s)
    n = 30
    for i in range(n):
        t = i / max(n - 1, 1); e = ease_out(t)
        f = blank(W, H)
        f = draw_text(f, "PaGeR", (W // 2, H // 2 - 8), size=72,
                       color=TEXT_LIGHT, bold=True, anchor="mm", alpha=e)
        f = draw_text(f, "from any panorama, anywhere",
                       (W // 2, H // 2 + 44), size=22, color=TEXT_DIM,
                       anchor="mm", alpha=e)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    # ── scenes 1-2
    for name in SCENES[:2]:
        s = load_sample(name)
        cursor = render_block(s, frames_dir, cursor)

    # ── mid card (~2s)
    n = 48
    for i in range(n):
        t = i / max(n - 1, 1); e = ease_in_out(t)
        a = 1.0 - abs(2 * t - 1)
        f = blank(W, H)
        # tiny pipeline schematic
        ex0 = W // 2 - 280; ey0 = H // 2 - 28
        cv2.rectangle(f, (ex0, ey0), (ex0 + 60, ey0 + 40), (200, 200, 200), 1)
        f = draw_text(f, "ERP", (ex0 + 30, ey0 + 12), size=14, color=TEXT_LIGHT, anchor="mm", alpha=a)
        cv2.arrowedLine(f, (ex0 + 60, ey0 + 20), (ex0 + 130, ey0 + 20), (180, 180, 180), 1, tipLength=0.2)
        cv2.rectangle(f, (ex0 + 130, ey0 - 10), (ex0 + 260, ey0 + 50), (220, 220, 220), 1)
        f = draw_text(f, "ViT-Giant", (ex0 + 195, ey0 + 14), size=14, color=TEXT_LIGHT, anchor="mm", alpha=a)
        # 4 head outputs
        for k, lab in enumerate(["depth", "normals", "sky", "scale"]):
            yy = ey0 - 30 + k * 20
            cv2.arrowedLine(f, (ex0 + 260, ey0 + 20), (ex0 + 360, yy), (180, 180, 180), 1, tipLength=0.18)
            f = draw_text(f, lab, (ex0 + 370, yy - 6), size=12, color=TEXT_DIM, alpha=a)
        f = draw_text(f, "one forward pass · depth + normals + sky + scale",
                       (W // 2, H // 2 + 90), size=18, color=TEXT_DIM,
                       anchor="mm", alpha=a)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    # ── scenes 3-5
    for name in SCENES[2:]:
        s = load_sample(name)
        cursor = render_block(s, frames_dir, cursor)

    # ── end card (~2s)
    n = 60
    for i in range(n):
        t = i / max(n - 1, 1); e = ease_in_out(t)
        f = blank(W, H)
        a = min(1.0, t * 4)
        f = draw_text(f, "PaGeR", (W // 2, H // 2 - 20), size=78,
                       color=TEXT_LIGHT, bold=True, anchor="mm", alpha=a)
        f = draw_text(f, "Panoramic Geometry Reconstruction",
                       (W // 2, H // 2 + 30), size=22, color=TEXT_DIM,
                       anchor="mm", alpha=a)
        f = draw_text(f, "code · weights · datasets — github.com/prs-eth/PaGeR",
                       (W // 2, H - 60), size=18, color=TEXT_DIM,
                       anchor="mm", alpha=a * 0.85)
        save_frame(f, frames_dir, cursor + i)
    cursor += n

    write_video(frames_dir, OUT / "p5_tour.mp4", fps=FPS)
    print(f"✓ wrote {OUT / 'p5_tour.mp4'} ({cursor} frames, {cursor / FPS:.1f}s)")


if __name__ == "__main__":
    main()
