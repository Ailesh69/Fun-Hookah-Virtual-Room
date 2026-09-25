"""Stationary hookah sprites drawn with OpenCV primitives.

Each design returns (BGRA sprite, (port_x, port_y)). The port point is the
tip of the hose nozzle on the side of the stem - the simulator draws a
dynamic hose from there to wherever the user's hand is.

Rendering: layered primitives with per-pixel shading -
- Glass base with radial gradient body, tinted liquid, meniscus line,
  top-left specular highlight, contact shadow, dark rim
- Metallic stem/collar with cylindrical shading and narrow specular streak
- Multiple ornament rings for silhouette variety
- Tapered clay bowl with foil (poked holes) on top
- Three glowing embers with dark base, warm bloom, hot core, ash flecks
- Side hose port (nozzle nub)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _paint(img: np.ndarray, mask: np.ndarray, color_bgr) -> None:
    a = mask[..., None].astype(np.float32)
    color = np.asarray(color_bgr, np.float32)
    img[..., :3] = (img[..., :3].astype(np.float32) * (1 - a) + color * a).astype(np.uint8)
    cur_a = img[..., 3:4].astype(np.float32) / 255.0
    img[..., 3] = np.clip((cur_a + a * (1 - cur_a))[..., 0] * 255, 0, 255).astype(np.uint8)


def _paint_rgb(img: np.ndarray, mask: np.ndarray, color_layer: np.ndarray) -> None:
    a = mask[..., None].astype(np.float32)
    img[..., :3] = (img[..., :3].astype(np.float32) * (1 - a) + color_layer * a).astype(np.uint8)
    cur_a = img[..., 3:4].astype(np.float32) / 255.0
    img[..., 3] = np.clip((cur_a + a * (1 - cur_a))[..., 0] * 255, 0, 255).astype(np.uint8)


def _ellipse_mask(shape, cx, cy, rx, ry) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return (((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2) <= 1.0


def _radial_shade(shape, cx, cy, rx, ry) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    return np.clip(1 - d, 0, 1)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def draw_glass_bulb(img, cx, cy, rx, ry, tint):
    shape = img.shape[:2]
    h, w = shape
    mask = _ellipse_mask(shape, cx, cy, rx, ry).astype(np.float32)
    if mask.sum() == 0:
        return

    tint_np = np.asarray(tint, np.float32)
    bright = np.clip(tint_np * 1.4, 0, 255)
    dark = tint_np * 0.15

    shade = _radial_shade(shape, cx, cy, rx, ry)
    body = dark + (bright - dark) * shade[..., None]
    _paint_rgb(img, mask * 0.95, np.clip(body, 0, 255))

    # Liquid gradient in the bottom half
    yy = np.mgrid[:h, :w][0].astype(np.float32)
    liquid_top = cy - int(ry * 0.05)
    liq_grad = np.clip((yy - liquid_top) / max(1, (cy + ry - liquid_top)), 0, 1)
    liq_mask = mask * liq_grad * 0.55
    _paint(img, liq_mask, tint_np * 0.55)

    # Meniscus line
    band = np.zeros(shape, np.float32)
    y1 = max(0, liquid_top - 1)
    y2 = min(h, liquid_top + 2)
    band[y1:y2, :] = 1.0
    _paint(img, band * mask * 0.7, (255, 255, 255))

    # Rim outline
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360,
                (int(tint_np[0] * 0.18), int(tint_np[1] * 0.18),
                 int(tint_np[2] * 0.18), 255), 2, cv2.LINE_AA)

    # Top-left specular crescent
    spec = np.zeros(shape, np.float32)
    cv2.ellipse(spec, (cx - int(rx * 0.45), cy - int(ry * 0.45)),
                (max(2, int(rx * 0.28)), max(2, int(ry * 0.12))),
                -25, 0, 360, 1.0, -1, cv2.LINE_AA)
    spec = cv2.GaussianBlur(spec, (0, 0), 3) * mask * 0.95
    _paint(img, spec, (255, 255, 255))

    # Bottom-right rim highlight
    spec2 = np.zeros(shape, np.float32)
    cv2.ellipse(spec2, (cx + int(rx * 0.55), cy + int(ry * 0.4)),
                (max(2, int(rx * 0.12)), max(2, int(ry * 0.04))),
                15, 0, 360, 1.0, -1, cv2.LINE_AA)
    spec2 = cv2.GaussianBlur(spec2, (0, 0), 2) * mask * 0.6
    _paint(img, spec2, (255, 255, 255))

    # Contact shadow inside base bottom
    shad = np.zeros(shape, np.float32)
    cv2.ellipse(shad, (cx, cy + int(ry * 0.75)),
                (int(rx * 0.85), max(2, int(ry * 0.22))),
                0, 0, 360, 1.0, -1, cv2.LINE_AA)
    shad = cv2.GaussianBlur(shad, (0, 0), 5) * mask * 0.45
    _paint(img, shad, (0, 0, 0))


def draw_metal_column(img, cx, y_top, y_bot, half_w, base_bgr, hl_bgr):
    h, w = img.shape[:2]
    y1 = max(0, y_top)
    y2 = min(h, y_bot)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w + 1)
    if x1 >= x2 or y1 >= y2:
        return

    xs = np.arange(x1, x2)
    u = (xs - (cx - half_w)) / (2 * half_w)
    theta = (u - 0.3) * np.pi * 1.15
    shade = np.clip(np.cos(theta), -0.85, 1.0)

    base = np.asarray(base_bgr, np.float32)
    hl = np.asarray(hl_bgr, np.float32)
    col = np.where(shade[:, None] >= 0,
                   base + (hl - base) * shade[:, None],
                   base * (1 + shade[:, None] * 0.8))

    streak = np.exp(-((u - 0.3) / 0.05) ** 2) * 55
    col = np.clip(col + streak[:, None], 0, 255).astype(np.uint8)

    img[y1:y2, x1:x2, :3] = col[None, :, :]
    img[y1:y2, x1:x2, 3] = 255


def draw_disc(img, cx, cy, rx, ry, base_bgr, hl_bgr):
    shape = img.shape[:2]
    mask = _ellipse_mask(shape, cx, cy, rx, ry).astype(np.float32)
    if mask.sum() == 0:
        return
    yy = np.mgrid[:shape[0], :shape[1]][0].astype(np.float32)
    v = np.clip((cy + ry - yy) / max(1, 2 * ry), 0, 1)
    base = np.asarray(base_bgr, np.float32)
    hl = np.asarray(hl_bgr, np.float32)
    color = base + (hl - base) * v[..., None]
    _paint_rgb(img, mask, np.clip(color, 0, 255))
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360,
                (int(base_bgr[0] * 0.35), int(base_bgr[1] * 0.35),
                 int(base_bgr[2] * 0.35), 255), 1, cv2.LINE_AA)


def draw_bowl(img, cx, cy, half_w, half_h, base_bgr, hl_bgr):
    top_y = cy - half_h
    bot_y = cy + half_h
    top_hw = half_w
    bot_hw = int(half_w * 0.55)

    body = np.array([
        [cx - top_hw, top_y],
        [cx + top_hw, top_y],
        [cx + bot_hw, bot_y],
        [cx - bot_hw, bot_y],
    ], np.int32)
    cv2.fillPoly(img, [body], (*base_bgr, 255), cv2.LINE_AA)

    shadow_poly = np.array([
        [cx, top_y],
        [cx + top_hw, top_y],
        [cx + bot_hw, bot_y],
        [cx, bot_y],
    ], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [shadow_poly],
                 (int(base_bgr[0] * 0.55), int(base_bgr[1] * 0.55),
                  int(base_bgr[2] * 0.55), 255), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)

    # Open top rim
    cv2.ellipse(img, (cx, top_y), (top_hw, max(2, int(half_h * 0.22))),
                0, 0, 360, (*hl_bgr, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, top_y), (top_hw, max(2, int(half_h * 0.22))),
                0, 0, 360,
                (int(base_bgr[0] * 0.35), int(base_bgr[1] * 0.35),
                 int(base_bgr[2] * 0.35), 255), 2, cv2.LINE_AA)
    # Bottom cap
    cv2.ellipse(img, (cx, bot_y), (bot_hw, max(2, int(half_h * 0.15))),
                0, 0, 360,
                (int(base_bgr[0] * 0.5), int(base_bgr[1] * 0.5),
                 int(base_bgr[2] * 0.5), 255), -1, cv2.LINE_AA)


def draw_foil(img, cx, cy, rx, ry):
    """Aluminum foil disc on top of the bowl with visible poked holes."""
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (205, 205, 215, 255), -1, cv2.LINE_AA)
    # Slight top-lit gradient
    hi = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(hi, (cx, cy - max(1, ry // 3)), (int(rx * 0.75), max(1, ry // 2)),
                0, 0, 360, 1.0, -1, cv2.LINE_AA)
    hi = cv2.GaussianBlur(hi, (0, 0), 2) * 0.4
    _paint(img, hi, (245, 245, 250))
    # Poked holes
    rng = np.random.default_rng(11)
    for _ in range(14):
        hx = cx + int(rng.integers(-rx + 3, rx - 3))
        hy = cy + (int(rng.integers(-ry + 1, max(-ry + 2, ry - 1))) if ry > 2 else 0)
        cv2.circle(img, (hx, hy), 1, (25, 25, 30, 255), -1)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (100, 100, 110, 255), 1, cv2.LINE_AA)


def draw_coals(img, cx, cy, spread_x, size):
    shape = img.shape[:2]
    coals = [(cx - spread_x, cy), (cx, cy - 1), (cx + spread_x, cy)]

    for (x, y) in coals:
        cv2.circle(img, (x, y), size, (25, 25, 25, 255), -1, cv2.LINE_AA)

    glow = np.zeros(shape, np.float32)
    for (x, y) in coals:
        cv2.circle(glow, (x, y), size + 4, 1.0, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    glow = np.clip(glow, 0, 1) * 0.9
    _paint(img, glow, (40, 120, 255))

    for (x, y) in coals:
        cv2.circle(img, (x, y), max(1, size // 3), (140, 210, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), max(1, size // 5), (200, 240, 255, 255), -1, cv2.LINE_AA)

    rng = np.random.default_rng(3)
    for (x, y) in coals:
        for _ in range(2):
            ox = int(rng.integers(-size // 2, max(-size // 2 + 1, size // 2)))
            oy = int(rng.integers(-size // 2, max(-size // 2 + 1, size // 2)))
            cv2.circle(img, (x + ox, y + oy), 1, (220, 220, 240, 255), -1, cv2.LINE_AA)


def draw_port(img, x, y, base_bgr, hl_bgr):
    """Small horizontal nozzle sticking out from the side of the stem."""
    cv2.ellipse(img, (x, y), (11, 7), 0, 0, 360, (*base_bgr, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (11, 7), 0, 0, 360,
                (int(base_bgr[0] * 0.3), int(base_bgr[1] * 0.3),
                 int(base_bgr[2] * 0.3), 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x + 4, y - 2), 3, (*hl_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x + 8, y), 2, (15, 15, 15, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Full assembly (no hose - hose is drawn dynamically by the simulator)
# ---------------------------------------------------------------------------

def build_hookah(w, h, glass_tint, metal_base, metal_hl) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = _blank(w, h)
    cx = w // 2

    coal_cy = int(h * 0.06)
    bowl_top = int(h * 0.09)
    bowl_bot = int(h * 0.18)
    plate_y = int(h * 0.20)
    stem_top = int(h * 0.21)
    collar_top = int(h * 0.54)
    collar_bot = int(h * 0.61)
    base_cy = int(h * 0.79)
    base_ry = int(h * 0.19)
    base_rx = int(w * 0.44)

    port_x = cx + int(w * 0.08)
    port_y = int(h * 0.50)

    # Ground shadow
    shad = np.zeros((h, w), np.float32)
    cv2.ellipse(shad, (cx, base_cy + base_ry + 10),
                (int(base_rx * 0.9), 9), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    shad = cv2.GaussianBlur(shad, (0, 0), 6) * 0.6
    _paint(img, shad, (0, 0, 0))

    # Glass base
    draw_glass_bulb(img, cx, base_cy, base_rx, base_ry, glass_tint)

    # Collar between base and stem
    draw_metal_column(img, cx, collar_top, collar_bot,
                      int(w * 0.095), metal_base, metal_hl)
    cv2.line(img, (cx - int(w * 0.095), collar_bot - 2),
             (cx + int(w * 0.095), collar_bot - 2),
             (int(metal_base[0] * 0.35), int(metal_base[1] * 0.35),
              int(metal_base[2] * 0.35), 255), 2, cv2.LINE_AA)

    # Main stem
    draw_metal_column(img, cx, stem_top, collar_top,
                      int(w * 0.055), metal_base, metal_hl)

    # Ornament rings
    for frac in (0.22, 0.32, 0.42, 0.52):
        ry = int(h * (0.21 + frac * 0.33))
        draw_metal_column(img, cx, ry, ry + 7,
                          int(w * 0.08), metal_base, metal_hl)

    # Charcoal plate
    draw_disc(img, cx, plate_y, int(w * 0.20),
              max(3, int(h * 0.018)), metal_base, metal_hl)

    # Clay bowl
    draw_bowl(img, cx, (bowl_top + bowl_bot) // 2,
              int(w * 0.09), (bowl_bot - bowl_top) // 2,
              (58, 68, 88), (110, 130, 160))

    # Foil disc
    draw_foil(img, cx, bowl_top + 2, int(w * 0.09), max(2, int(h * 0.012)))

    # Coals
    draw_coals(img, cx, coal_cy + 4, int(w * 0.05),
               max(5, int(w * 0.024)))

    # Hose port
    draw_port(img, port_x, port_y, metal_base, metal_hl)

    return img, (port_x + int(w * 0.025), port_y)


# ---------------------------------------------------------------------------
# Color palettes (BGR)
# ---------------------------------------------------------------------------

PALETTES = [
    # (name, glass_tint, metal_base, metal_hl, hose_base, hose_hl)
    ("Classic Amber",
     (20, 110, 200),   (35, 85, 145),  (120, 210, 255),
     (25, 25, 30),     (90, 90, 100)),
    ("Sapphire",
     (190, 100, 30),   (150, 150, 160), (240, 240, 250),
     (60, 35, 20),     (170, 120, 60)),
    ("Royal Ruby",
     (45, 30, 190),    (30, 90, 155),   (90, 200, 255),
     (20, 20, 40),     (90, 65, 100)),
    ("Emerald",
     (80, 165, 55),    (140, 140, 145), (235, 235, 240),
     (35, 55, 30),     (120, 165, 90)),
    ("Midnight",
     (45, 40, 55),     (55, 55, 62),    (155, 155, 165),
     (15, 15, 22),     (75, 75, 88)),
]


def _factory(pal):
    def make(w, h):
        return build_hookah(w, h, pal[1], pal[2], pal[3])
    return make


DESIGNS = [(p[0], _factory(p)) for p in PALETTES]
