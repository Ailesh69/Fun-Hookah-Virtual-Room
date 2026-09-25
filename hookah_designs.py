"""Realistic-looking hookah sprites drawn with OpenCV primitives.

Each design returns (BGRA sprite, (smoke_origin_x, smoke_origin_y)).
The smoke origin is the top of the coals, where smoke visibly rises from.

Rendering approach: layered primitives with per-pixel shading -
glass base with radial gradient, liquid line, top-left specular; metal
stem with cylindrical shading and a narrow specular streak; clay bowl
with side shadow; multi-layer bloom for glowing embers; sagging hose
with dark underlayer plus offset highlight; small metal mouthpiece.
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
    """Blend a solid BGR color into img (BGRA) using a float (H,W) alpha mask."""
    a = mask[..., None].astype(np.float32)
    color = np.asarray(color_bgr, np.float32)
    img[..., :3] = (img[..., :3].astype(np.float32) * (1 - a) + color * a).astype(np.uint8)
    cur_a = img[..., 3:4].astype(np.float32) / 255.0
    img[..., 3] = np.clip((cur_a + a * (1 - cur_a))[..., 0] * 255, 0, 255).astype(np.uint8)


def _paint_rgb(img: np.ndarray, mask: np.ndarray, color_layer: np.ndarray) -> None:
    """Blend a per-pixel color layer (H,W,3) into img using a float mask."""
    a = mask[..., None].astype(np.float32)
    img[..., :3] = (img[..., :3].astype(np.float32) * (1 - a) + color_layer * a).astype(np.uint8)
    cur_a = img[..., 3:4].astype(np.float32) / 255.0
    img[..., 3] = np.clip((cur_a + a * (1 - cur_a))[..., 0] * 255, 0, 255).astype(np.uint8)


def _ellipse_mask(shape, cx, cy, rx, ry) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return (((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2) <= 1.0


def _radial_shade(shape, cx, cy, rx, ry) -> np.ndarray:
    """1 at center, 0 at ellipse edge (linear falloff)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    return np.clip(1 - d, 0, 1)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def draw_glass_bulb(img: np.ndarray, cx: int, cy: int, rx: int, ry: int,
                    tint: Tuple[int, int, int]) -> None:
    shape = img.shape[:2]
    h, w = shape
    mask = _ellipse_mask(shape, cx, cy, rx, ry).astype(np.float32)
    if mask.sum() == 0:
        return

    tint_np = np.asarray(tint, np.float32)
    bright = np.clip(tint_np * 1.35, 0, 255)
    dark = tint_np * 0.15

    # Body: radial gradient sphere
    shade = _radial_shade(shape, cx, cy, rx, ry)
    body = dark + (bright - dark) * shade[..., None]
    _paint_rgb(img, mask * 0.95, np.clip(body, 0, 255))

    # Liquid: soft vertical gradient darkening toward the bottom
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

    # Specular highlight top-left
    spec = np.zeros(shape, np.float32)
    cv2.ellipse(spec, (cx - int(rx * 0.45), cy - int(ry * 0.45)),
                (max(2, int(rx * 0.28)), max(2, int(ry * 0.12))),
                -25, 0, 360, 1.0, -1, cv2.LINE_AA)
    spec = cv2.GaussianBlur(spec, (0, 0), 3) * mask * 0.9
    _paint(img, spec, (255, 255, 255))

    # Small bottom-right rim reflection
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


def draw_metal_column(img: np.ndarray, cx: int, y_top: int, y_bot: int,
                      half_w: int, base_bgr, hl_bgr) -> None:
    """Vertical metallic cylinder with cylindrical shading + specular streak."""
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


def draw_disc(img: np.ndarray, cx: int, cy: int, rx: int, ry: int,
              base_bgr, hl_bgr) -> None:
    """Flat elliptical disc (charcoal plate) with top-lit gradient."""
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


def draw_bowl(img: np.ndarray, cx: int, cy: int, half_w: int, half_h: int,
              base_bgr, hl_bgr) -> None:
    """Tapered clay bowl with rim and side shadow."""
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

    # Side shadow (right half)
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

    # Top opening (rim) - lit inside then dark rim line
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


def draw_coals(img: np.ndarray, cx: int, cy: int, spread_x: int, size: int) -> None:
    """Three glowing coals: dark base, warm bloom, bright core, ash flecks."""
    shape = img.shape[:2]
    coals = [(cx - spread_x, cy), (cx, cy - 1), (cx + spread_x, cy)]

    for (x, y) in coals:
        cv2.circle(img, (x, y), size, (25, 25, 25, 255), -1, cv2.LINE_AA)

    glow = np.zeros(shape, np.float32)
    for (x, y) in coals:
        cv2.circle(glow, (x, y), size + 3, 1.0, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 6)
    glow = np.clip(glow, 0, 1) * 0.85
    _paint(img, glow, (40, 120, 255))

    for (x, y) in coals:
        cv2.circle(img, (x, y), max(1, size // 3), (120, 200, 255, 255), -1, cv2.LINE_AA)

    rng = np.random.default_rng(3)
    for (x, y) in coals:
        for _ in range(2):
            ox = int(rng.integers(-size // 2, max(-size // 2 + 1, size // 2)))
            oy = int(rng.integers(-size // 2, max(-size // 2 + 1, size // 2)))
            cv2.circle(img, (x + ox, y + oy), 1, (220, 220, 240, 255), -1, cv2.LINE_AA)


def draw_hose(img: np.ndarray, pts, base_bgr, hl_bgr, thickness: int = 12) -> None:
    """Draping hose: shadow underlayer + main body + offset highlight."""
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (0, 0, 0, 255), thickness + 4, cv2.LINE_AA)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (*base_bgr, 255), thickness, cv2.LINE_AA)
    off = max(1, thickness // 4)
    for i in range(len(pts) - 1):
        p1 = (pts[i][0], pts[i][1] - off)
        p2 = (pts[i + 1][0], pts[i + 1][1] - off)
        cv2.line(img, p1, p2, (*hl_bgr, 255), max(1, thickness // 3), cv2.LINE_AA)


def draw_mouthpiece(img: np.ndarray, x: int, y: int, base_bgr, hl_bgr) -> None:
    cv2.circle(img, (x, y), 11, (*base_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 11,
               (int(base_bgr[0] * 0.3), int(base_bgr[1] * 0.3),
                int(base_bgr[2] * 0.3), 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x - 3, y - 3), 3, (*hl_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 4, (15, 15, 15, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Full assembly
# ---------------------------------------------------------------------------

def build_hookah(w: int, h: int,
                 glass_tint, metal_base, metal_hl,
                 hose_color, hose_hl) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = _blank(w, h)
    cx = w // 2

    coal_y = int(h * 0.06)
    bowl_top = int(h * 0.085)
    bowl_bot = int(h * 0.185)
    plate_y = int(h * 0.205)
    col_top = int(h * 0.215)
    collar_top = int(h * 0.49)
    collar_bot = int(h * 0.55)
    base_cy = int(h * 0.73)
    base_ry = int(h * 0.21)
    base_rx = int(w * 0.42)

    # Ground shadow
    shad = np.zeros((h, w), np.float32)
    cv2.ellipse(shad, (cx, base_cy + base_ry + 8),
                (int(base_rx * 0.9), 8), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    shad = cv2.GaussianBlur(shad, (0, 0), 5) * 0.55
    _paint(img, shad, (0, 0, 0))

    # Glass base
    draw_glass_bulb(img, cx, base_cy, base_rx, base_ry, glass_tint)

    # Collar joining base to stem
    draw_metal_column(img, cx, collar_top, collar_bot,
                      int(w * 0.09), metal_base, metal_hl)
    cv2.line(img, (cx - int(w * 0.09), collar_bot - 2),
             (cx + int(w * 0.09), collar_bot - 2),
             (int(metal_base[0] * 0.35), int(metal_base[1] * 0.35),
              int(metal_base[2] * 0.35), 255), 2, cv2.LINE_AA)

    # Main stem
    draw_metal_column(img, cx, col_top, collar_top,
                      int(w * 0.05), metal_base, metal_hl)

    # Ornament rings on stem
    for frac in (0.26, 0.35, 0.44):
        ry = int(h * (0.215 + frac * 0.28))
        draw_metal_column(img, cx, ry, ry + 6,
                          int(w * 0.075), metal_base, metal_hl)

    # Charcoal plate
    draw_disc(img, cx, plate_y, int(w * 0.17),
              max(3, int(h * 0.018)), metal_base, metal_hl)

    # Clay bowl
    draw_bowl(img, cx, (bowl_top + bowl_bot) // 2,
              int(w * 0.085), (bowl_bot - bowl_top) // 2,
              (58, 68, 88), (110, 130, 160))

    # Foil on top of bowl (a subtle grey disc under coals)
    cv2.ellipse(img, (cx, bowl_top + 2),
                (int(w * 0.085), max(2, int(h * 0.012))),
                0, 0, 360, (180, 180, 190, 255), -1, cv2.LINE_AA)

    # Coals
    draw_coals(img, cx, coal_y + 4, int(w * 0.045),
               max(4, int(w * 0.022)))

    # Hose port
    port_y = int(h * 0.47)
    port_x = cx + int(w * 0.06)
    cv2.circle(img, (port_x, port_y), 7, (*metal_base, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (port_x, port_y), 7,
               (int(metal_base[0] * 0.3), int(metal_base[1] * 0.3),
                int(metal_base[2] * 0.3), 255), 1, cv2.LINE_AA)
    cv2.circle(img, (port_x - 2, port_y - 2), 2, (*metal_hl, 255), -1, cv2.LINE_AA)

    # Draping hose
    end_x = min(w - 18, port_x + int(w * 0.34))
    end_y = min(h - 20, port_y + int(h * 0.30))
    pts = []
    start = (port_x + 7, port_y)
    for t in np.linspace(0, 1, 32):
        sag = np.sin(t * np.pi) * (h * 0.05)
        x = int(start[0] * (1 - t) + end_x * t)
        y = int(start[1] * (1 - t) + end_y * t + sag)
        pts.append((x, y))
    draw_hose(img, pts, hose_color, hose_hl, thickness=max(8, int(w * 0.045)))
    draw_mouthpiece(img, pts[-1][0], pts[-1][1], metal_base, metal_hl)

    return img, (cx, coal_y - 3)


# ---------------------------------------------------------------------------
# Designs (color palettes) - BGR
# ---------------------------------------------------------------------------

def classic_amber(w, h):
    return build_hookah(
        w, h,
        glass_tint=(20, 110, 200),      # amber whiskey
        metal_base=(35, 85, 145),       # antique brass
        metal_hl=(120, 210, 255),       # bright gold
        hose_color=(25, 25, 30),        # dark leather
        hose_hl=(90, 90, 100),
    )


def sapphire(w, h):
    return build_hookah(
        w, h,
        glass_tint=(190, 100, 30),      # deep sapphire blue
        metal_base=(150, 150, 160),     # chrome
        metal_hl=(240, 240, 250),
        hose_color=(60, 35, 20),
        hose_hl=(170, 120, 60),
    )


def royal_ruby(w, h):
    return build_hookah(
        w, h,
        glass_tint=(45, 30, 190),       # ruby
        metal_base=(30, 90, 155),       # rose gold
        metal_hl=(90, 200, 255),
        hose_color=(20, 20, 40),
        hose_hl=(90, 65, 100),
    )


def emerald(w, h):
    return build_hookah(
        w, h,
        glass_tint=(80, 165, 55),       # emerald green
        metal_base=(140, 140, 145),     # silver
        metal_hl=(235, 235, 240),
        hose_color=(35, 55, 30),
        hose_hl=(120, 165, 90),
    )


def midnight(w, h):
    return build_hookah(
        w, h,
        glass_tint=(45, 40, 55),        # smoked glass with purple hint
        metal_base=(55, 55, 62),        # gunmetal
        metal_hl=(155, 155, 165),
        hose_color=(15, 15, 22),
        hose_hl=(75, 75, 88),
    )


DESIGNS = [
    ("Classic Amber", classic_amber),
    ("Sapphire", sapphire),
    ("Royal Ruby", royal_ruby),
    ("Emerald", emerald),
    ("Midnight", midnight),
]
