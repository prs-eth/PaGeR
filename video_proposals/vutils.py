"""Shared video utilities: panorama loader, point-cloud projection, ERP-sphere wrap, ffmpeg writer."""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "video_proposals" / "cache"
OUT = ROOT / "video_proposals" / "out"
FRAMES = ROOT / "video_proposals" / "frames"
ASSETS_EX = ROOT / "assets" / "examples"

# global look
BG_DARK = (8, 9, 14)
BG_WHITE = (250, 250, 248)
ACCENT = (255, 220, 140)         # warm accent
TEXT_LIGHT = (235, 235, 230)
TEXT_DIM = (140, 145, 160)
TEXT_DARK = (24, 26, 32)

EXAMPLES = ["livingroom_synth", "church_meeting_room", "eth_campus_plaza",
            "zurich_street_corner", "medieval_kitchen"]


@dataclass
class Sample:
    name: str
    rgb: np.ndarray            # (H,W,3) uint8
    depth_viz: np.ndarray      # (H,W,3) uint8
    normals_viz: np.ndarray    # (H,W,3) uint8
    sky_mask: np.ndarray       # (H,W) uint8
    cubemap: np.ndarray        # (6,3,h,w) float32 [0,1]
    depth_raw: np.ndarray      # (H,W) float32 meters
    points_xyz: np.ndarray     # (N,3) float32
    points_rgb: np.ndarray     # (N,3) float32 in [0,1]
    points_normals: np.ndarray # (N,3) float32 in [0,1]
    scene: str


def load_sample(name: str) -> Sample:
    d = CACHE / name
    return Sample(
        name=name,
        rgb=np.array(Image.open(d / "rgb.png")),
        depth_viz=np.array(Image.open(d / "depth_viz.png")),
        normals_viz=np.array(Image.open(d / "normals_viz.png")),
        sky_mask=np.array(Image.open(d / "sky_mask.png")),
        cubemap=np.load(d / "cubemap.npy"),
        depth_raw=np.load(d / "depth_raw.npy"),
        points_xyz=np.load(d / "points_xyz.npy"),
        points_rgb=np.load(d / "points_rgb.npy"),
        points_normals=np.load(d / "points_normals.npy"),
        scene=(d / "scene.txt").read_text().strip(),
    )


# ────────────────────────────────────────────────────────────────────
# Drawing helpers
# ────────────────────────────────────────────────────────────────────
def blank(w: int, h: int, color=BG_DARK) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8); img[:] = color; return img


def vignette(img: np.ndarray, strength: float = 0.55) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    fall = 1.0 - np.clip(r * strength, 0, 1) ** 2
    return (img.astype(np.float32) * fall[..., None]).clip(0, 255).astype(np.uint8)


_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold
        else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    ) + ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    for p in candidates:
        if Path(p).exists():
            try:
                f = ImageFont.truetype(p, size); _FONT_CACHE[key] = f; return f
            except Exception:
                pass
    f = ImageFont.load_default(); _FONT_CACHE[key] = f; return f


def draw_text(img: np.ndarray, text: str, xy, size: int = 28, color=TEXT_LIGHT,
              bold: bool = False, anchor: str = "lt", alpha: float = 1.0) -> np.ndarray:
    pil = Image.fromarray(img); draw = ImageDraw.Draw(pil, "RGBA")
    rgba = (*color, int(255 * alpha))
    draw.text(xy, text, font=_font(size, bold), fill=rgba, anchor=anchor)
    return np.array(pil)


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear cross-fade a→b."""
    t = float(np.clip(t, 0, 1))
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).clip(0, 255).astype(np.uint8)


def ease_in_out(t: float) -> float:
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


def paste(canvas: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0) -> np.ndarray:
    h, w = src.shape[:2]; H, W = canvas.shape[:2]
    x0 = max(0, x); y0 = max(0, y); x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0: return canvas
    sx0, sy0 = x0 - x, y0 - y; sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    piece = src[sy0:sy1, sx0:sx1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region * (1 - alpha) + piece * alpha).clip(0, 255).astype(np.uint8)
    return canvas


def resize(img: np.ndarray, w: int | None = None, h: int | None = None,
           keep_aspect: bool = True) -> np.ndarray:
    H, W = img.shape[:2]
    if w is None and h is None: return img
    if keep_aspect:
        if w is None: w = max(1, int(round(W * (h / H))))
        elif h is None: h = max(1, int(round(H * (w / W))))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w * h < W * H else cv2.INTER_LINEAR)


# ────────────────────────────────────────────────────────────────────
# Point cloud projection (orthographic-perspective splatter)
# ────────────────────────────────────────────────────────────────────
def project_pointcloud(points: np.ndarray, colors: np.ndarray, *,
                       canvas_w: int, canvas_h: int,
                       yaw: float = 0.0, pitch: float = 0.0,
                       distance: float = 6.0, fov_deg: float = 55.0,
                       splat: int = 2, bg=(8, 9, 14), gamma: float = 0.9) -> np.ndarray:
    """Render the point cloud as a tiny splatted image. Vectorised numpy + cv2."""
    # camera looks at origin, positioned at +Z*distance, then rotate by (yaw, pitch).
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # Rotate points: yaw around Y, pitch around X.
    P = points.copy()
    # yaw (around Y)
    xz = np.stack([cy * P[:, 0] + sy * P[:, 2], P[:, 1], -sy * P[:, 0] + cy * P[:, 2]], axis=1)
    # pitch (around X)
    P = np.stack([xz[:, 0], cp * xz[:, 1] - sp * xz[:, 2], sp * xz[:, 1] + cp * xz[:, 2]], axis=1)
    # translate so camera at +z=distance looks at origin → put camera at (0,0,distance), points stay.
    z = distance - P[:, 2]                    # depth from camera
    mask = z > 0.05
    P = P[mask]; col = colors[mask]; z = z[mask]
    if len(P) == 0:
        return np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    f = canvas_h * 0.5 / np.tan(np.deg2rad(fov_deg) * 0.5)
    u = canvas_w * 0.5 + f * P[:, 0] / z
    v = canvas_h * 0.5 - f * P[:, 1] / z
    ui = u.astype(np.int32); vi = v.astype(np.int32)
    inb = (ui >= 0) & (ui < canvas_w) & (vi >= 0) & (vi < canvas_h)
    ui = ui[inb]; vi = vi[inb]; col = col[inb]; z = z[inb]

    # painter's algorithm with depth buffer for splat=1; for splat>1 we draw far-to-near.
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    if splat <= 1:
        # near→far overwrite: sort descending by z, write so closest end up last
        order = np.argsort(-z)
        canvas[vi[order], ui[order]] = (col[order] * 255).astype(np.uint8)
    else:
        # tiny splats; sort far→near so near overwrites
        order = np.argsort(-z)
        ui_s, vi_s, col_s = ui[order], vi[order], col[order]
        # rasterise into a float buffer for slight smoothing
        buf = canvas.astype(np.float32)
        r = splat
        # build a soft kernel
        ks = 2 * r + 1
        yy, xx = np.mgrid[-r:r+1, -r:r+1]
        kernel = np.exp(-(xx**2 + yy**2) / max(1e-6, (r * 0.9)**2))
        kernel = (kernel / kernel.max())
        # iterate splat offsets and accumulate via maximum (so near wins per channel)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                w = kernel[dy + r, dx + r]
                if w < 0.05:
                    continue
                yy2 = vi_s + dy; xx2 = ui_s + dx
                ok = (yy2 >= 0) & (yy2 < canvas_h) & (xx2 >= 0) & (xx2 < canvas_w)
                yy2 = yy2[ok]; xx2 = xx2[ok]; cc = col_s[ok]
                pix = (cc * 255 * w + buf[yy2, xx2] * (1 - w))
                buf[yy2, xx2] = pix
        canvas = buf.clip(0, 255).astype(np.uint8)
    if gamma != 1.0:
        canvas = (np.power(canvas.astype(np.float32) / 255.0, gamma) * 255).astype(np.uint8)
    return canvas


def autoframe_distance(points: np.ndarray, fov_deg: float = 55.0, margin: float = 1.15) -> float:
    """Pick a camera distance that frames the cloud nicely."""
    if len(points) == 0: return 5.0
    center = np.median(points, axis=0)
    r = np.linalg.norm(points - center, axis=1)
    r95 = float(np.percentile(r, 95))
    return r95 / np.tan(np.deg2rad(fov_deg) * 0.5) * margin


def center_points(points: np.ndarray) -> np.ndarray:
    """Recenter points around the optical center."""
    c = np.median(points, axis=0)
    return (points - c).astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# ERP → sphere preview (orthographic stub for a "looking at the panorama as a globe")
# ────────────────────────────────────────────────────────────────────
def render_sphere_morph(erp: np.ndarray, alpha: float, *,
                         canvas_w: int, canvas_h: int,
                         flat_aspect: float = 1.5, R: float = 1.0,
                         scale_flat: float = 350.0, scale_sphere: float = 245.0,
                         cy_flat: float = 302.0, cy_sphere: float = 360.0,
                         texture_yaw_flat: float = np.pi,
                         splat: int = 2, Nu: int = 600, Nv: int = 300,
                         bg=(8, 9, 14)) -> np.ndarray:
    """Forward-warp a dense grid of ERP samples through a smooth geometric morph
    between a flat rectangle (alpha=0) and a sphere (alpha=1). True 3D deformation:
    each (u, v) sample interpolates between its flat XY position and its sphere XYZ
    position, then is orthographically projected. Painter's algorithm handles occlusion
    when the back of the sphere appears at higher alpha.

    Default parameters are tuned so alpha=0 matches the prologue intro's final frame
    (ERP centred at y=302, width≈1050, rolled by 50%) and alpha=1 matches the splay
    scene's initial sphere (radius_frac≈0.34, yaw=0, centred at canvas centre).
    """
    H_e, W_e = erp.shape[:2]
    a = float(np.clip(alpha, 0.0, 1.0))
    scale = (1 - a) * scale_flat + a * scale_sphere
    cy = (1 - a) * cy_flat + a * cy_sphere
    cx = canvas_w * 0.5

    u_lin = np.linspace(0.0, 1.0, Nu, endpoint=False)
    v_lin = np.linspace(0.0, 1.0, Nv)
    uu, vv = np.meshgrid(u_lin, v_lin)

    # flat geometry (rectangle in world units)
    flat_W = 2.0 * R * flat_aspect
    flat_H = flat_W / 2.0
    fx = (uu - 0.5) * flat_W
    fy = (0.5 - vv) * flat_H
    fz = np.zeros_like(uu)

    # sphere geometry
    lon = (uu - 0.5) * 2 * np.pi
    lat = (0.5 - vv) * np.pi
    cos_lat = np.cos(lat)
    sx = R * np.sin(lon) * cos_lat
    sy = R * np.sin(lat)
    sz = R * np.cos(lon) * cos_lat

    # interpolate the 3D position
    X = (1 - a) * fx + a * sx
    Y = (1 - a) * fy + a * sy
    Z = (1 - a) * fz + a * sz

    # orthographic projection
    px = (X * scale + cx).astype(np.int32)
    py = (-Y * scale + cy).astype(np.int32)

    # texture roll: at alpha=0 the ERP is rolled to match the intro's end state;
    # the roll smoothly unwinds to 0 as alpha→1 so the geometry & texture align on the sphere
    u_offset = (1 - a) * texture_yaw_flat / (2 * np.pi)
    u_sampled = (uu + u_offset) % 1.0
    u_pix = (u_sampled * (W_e - 1)).astype(np.int32) % W_e
    v_pix = (vv * (H_e - 1)).clip(0, H_e - 1).astype(np.int32)
    colors = erp[v_pix, u_pix]

    # painter's algorithm: paint far → near so the front of the sphere overwrites the back
    order = np.argsort(Z.ravel())
    px_f = px.ravel()[order]; py_f = py.ravel()[order]
    col_f = colors.reshape(-1, 3)[order]

    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    r = splat
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            yy = py_f + dy; xx = px_f + dx
            ok = (yy >= 0) & (yy < canvas_h) & (xx >= 0) & (xx < canvas_w)
            canvas[yy[ok], xx[ok]] = col_f[ok]
    return canvas


def erp_to_sphere(erp: np.ndarray, *, canvas_w: int, canvas_h: int,
                  yaw: float = 0.0, radius_frac: float = 0.42,
                  bg=(8, 9, 14)) -> np.ndarray:
    """Render the ERP as a textured sphere (orthographic, ignores backface)."""
    H, W = erp.shape[:2]
    R = int(min(canvas_w, canvas_h) * radius_frac)
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[-R:R, -R:R].astype(np.float32)
    rr = np.sqrt(xx * xx + yy * yy)
    inside = rr < R
    z = np.zeros_like(xx); z[inside] = np.sqrt(R * R - rr[inside] ** 2)
    # surface normal of the unit sphere -> longitude/latitude
    # (image y goes down; flip so sky lands on top of the rendered sphere)
    nx = xx / R; ny = -yy / R; nz = z / R
    lon = np.arctan2(nx, nz) + yaw            # [-pi, pi]
    lat = np.arcsin(ny.clip(-1, 1))           # [-pi/2, pi/2]
    u = (lon + np.pi) / (2 * np.pi)
    v = 0.5 + lat / np.pi
    u = (u * (W - 1)).astype(np.int32) % W
    v = ((1 - v) * (H - 1)).clip(0, H - 1).astype(np.int32)
    samp = erp[v, u]
    cy, cx = canvas_h // 2, canvas_w // 2
    region = canvas[cy - R:cy + R, cx - R:cx + R].copy()
    region[inside] = samp[inside]
    # soft shade
    shade = (0.55 + 0.45 * nz).clip(0, 1)
    region = (region.astype(np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8)
    canvas[cy - R:cy + R, cx - R:cx + R] = region
    return canvas


# ────────────────────────────────────────────────────────────────────
# ffmpeg writer
# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# Modality visualizers (added for the iterated p1 variants)
# ────────────────────────────────────────────────────────────────────
def make_coarse_metric_depth(depth_raw: np.ndarray,
                              sky_mask: np.ndarray | None = None,
                              downsample: int = 10, blur_sigma: float = 4.0,
                              cmap_name: str = "inferno") -> np.ndarray:
    """Paper-style coarse metric-depth visualisation: log-space inferno, heavily smoothed."""
    H, W = depth_raw.shape
    d = depth_raw.astype(np.float32).copy()
    if sky_mask is not None:
        # set sky to a soft max so it reads as the warm "far" colour
        sky = sky_mask.astype(np.float32) / 255.0
        finite_top = float(np.percentile(d[d < d.max()] if (d < d.max()).any() else d, 99))
        d = np.clip(d, 0, finite_top)
        d[sky > 0.5] = finite_top
    d_log = np.log(np.maximum(d, 0.05))
    sw = max(8, W // downsample); sh = max(8, H // downsample)
    small = cv2.resize(d_log, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=blur_sigma)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    lo = float(np.percentile(big, 2)); hi = float(np.percentile(big, 98))
    n = ((big - lo) / max(1e-3, hi - lo)).clip(0, 1)
    cm = matplotlib.colormaps[cmap_name]
    return (cm(n)[..., :3] * 255).astype(np.uint8)


def make_sky_viz(sky_mask: np.ndarray, bg=(18, 20, 26),
                 fg=(200, 230, 255)) -> np.ndarray:
    """Cyan-tinted sky mask on dark."""
    m = sky_mask.astype(np.float32) / 255.0
    H, W = sky_mask.shape
    out = np.empty((H, W, 3), dtype=np.float32)
    for i, (b, f) in enumerate(zip(bg, fg)):
        out[..., i] = b * (1 - m) + f * m
    return out.clip(0, 255).astype(np.uint8)


def make_cubemap_stack(faces: np.ndarray, thumb_w: int = 90, gap: int = 4,
                       shadow: bool = True) -> np.ndarray:
    """6 cubemap faces stacked vertically as labelled thumbnails — F,R,B,L,U,D top to bottom."""
    imgs = [(np.transpose(f, (1, 2, 0)) * 255).astype(np.uint8) for f in faces]
    th = thumb_w
    pad = 6
    total_h = 6 * th + 5 * gap + 2 * pad
    total_w = thumb_w + 2 * pad + (8 if shadow else 0)
    canvas = np.full((total_h, total_w, 3), BG_DARK, dtype=np.uint8)
    y = pad
    for k, im in enumerate(imgs):
        thumb = cv2.resize(im, (th, th), interpolation=cv2.INTER_AREA)
        if shadow:
            # subtle drop shadow
            cv2.rectangle(canvas, (pad + 3, y + 3), (pad + th + 3, y + th + 3),
                          (0, 0, 0), -1)
        canvas[y:y + th, pad:pad + th] = thumb
        cv2.rectangle(canvas, (pad, y), (pad + th, y + th), (90, 95, 110), 1)
        y += th + gap
    return canvas


def write_video(frames_dir: Path, out_path: Path, fps: int = 24,
                pattern: str = "f_%05d.png", crf: int = 18) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", "medium",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def fresh_frames_dir(name: str) -> Path:
    d = FRAMES / name
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_frame(frame: np.ndarray, frames_dir: Path, idx: int) -> None:
    Image.fromarray(frame).save(frames_dir / f"f_{idx:05d}.png")
