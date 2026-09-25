"""Hookah designs drawn programmatically with OpenCV primitives.

Each design returns a BGRA sprite (numpy array) sized to `width` x `height`.
The mouthpiece tip (where smoke comes out) is at (width // 2, small y near top)
by convention so the caller can anchor smoke correctly.
"""

import cv2
import numpy as np


def _blank(w, h):
    return np.zeros((h, w, 4), dtype=np.uint8)


def _ellipse(img, center, axes, color, thickness=-1):
    cv2.ellipse(img, center, axes, 0, 0, 360, color, thickness, cv2.LINE_AA)


def _gradient_fill(img, cx, cy, rx, ry, inner_color, outer_color):
    """Cheap radial gradient inside an ellipse."""
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dx = (xx - cx) / max(rx, 1)
    dy = (yy - cy) / max(ry, 1)
    dist = np.clip(np.sqrt(dx * dx + dy * dy), 0, 1)
    mask = dist <= 1.0
    for c in range(3):
        img[..., c] = np.where(
            mask,
            (inner_color[c] * (1 - dist) + outer_color[c] * dist).astype(np.uint8),
            img[..., c],
        )
    img[..., 3] = np.where(mask, 255, img[..., 3])


def _draw_base_hookah(w, h, base_color, accent_color, glow_color, name):
    """Common hookah silhouette: base bowl, stem, top bowl, hose. Colored per design."""
    img = _blank(w, h)
    cx = w // 2

    # y-anchors (top -> bottom)
    top_bowl_y = int(h * 0.14)
    stem_top_y = int(h * 0.22)
    stem_bot_y = int(h * 0.55)
    base_center_y = int(h * 0.74)

    # ---- Base (big bulb) ----
    base_rx = int(w * 0.42)
    base_ry = int(h * 0.22)
    _gradient_fill(img, cx, base_center_y, base_rx, base_ry, glow_color, base_color)
    _ellipse(img, (cx, base_center_y), (base_rx, base_ry), (*accent_color, 255), 3)
    # Water shimmer inside base
    cv2.ellipse(img, (cx - base_rx // 3, base_center_y - base_ry // 3),
                (base_rx // 4, base_ry // 6), 0, 0, 360,
                (255, 255, 255, 120), -1, cv2.LINE_AA)

    # ---- Stem ----
    stem_w = max(6, int(w * 0.06))
    cv2.rectangle(img, (cx - stem_w // 2, stem_top_y),
                  (cx + stem_w // 2, stem_bot_y),
                  (*base_color, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - stem_w // 2, stem_top_y),
                  (cx + stem_w // 2, stem_bot_y),
                  (*accent_color, 255), 2, cv2.LINE_AA)
    # Stem decorations (rings)
    for i in range(3):
        ry = stem_top_y + int((stem_bot_y - stem_top_y) * (0.25 + 0.25 * i))
        cv2.line(img, (cx - stem_w, ry), (cx + stem_w, ry), (*accent_color, 255), 2, cv2.LINE_AA)

    # ---- Top bowl (where coals sit) ----
    bowl_rx = int(w * 0.18)
    bowl_ry = int(h * 0.06)
    _ellipse(img, (cx, top_bowl_y + bowl_ry), (bowl_rx, bowl_ry), (*base_color, 255))
    _ellipse(img, (cx, top_bowl_y + bowl_ry), (bowl_rx, bowl_ry), (*accent_color, 255), 3)
    # Coals (glowing)
    for dx in (-bowl_rx // 2, 0, bowl_rx // 2):
        cv2.circle(img, (cx + dx, top_bowl_y), max(3, bowl_ry // 2),
                   (40, 80, 255, 255), -1, cv2.LINE_AA)  # orange-red in BGR
        cv2.circle(img, (cx + dx, top_bowl_y), max(2, bowl_ry // 3),
                   (80, 180, 255, 255), -1, cv2.LINE_AA)

    # ---- Hose (curly on the right) ----
    pts = []
    hose_start = (cx + int(w * 0.05), int(h * 0.62))
    for t in np.linspace(0, 1, 40):
        x = int(hose_start[0] + w * 0.35 * t + w * 0.06 * np.sin(t * np.pi * 3))
        y = int(hose_start[1] + h * 0.18 * t)
        x = min(x, w - 6)
        y = min(y, h - 6)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (*accent_color, 255), 6, cv2.LINE_AA)
        cv2.line(img, pts[i], pts[i + 1], (*base_color, 255), 3, cv2.LINE_AA)
    # Mouthpiece tip
    cv2.circle(img, pts[-1], 8, (*accent_color, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pts[-1], 5, (30, 30, 30, 255), -1, cv2.LINE_AA)

    # ---- Nameplate ----
    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    tx = cx - tw // 2
    ty = base_center_y + 4
    cv2.putText(img, name, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (255, 255, 255, 255), 1, cv2.LINE_AA)

    return img, (cx, top_bowl_y - 4)  # sprite + smoke origin (top of coals)


def classic_gold(w, h):
    img, tip = _draw_base_hookah(
        w, h,
        base_color=(30, 110, 180),      # bronze-ish (BGR)
        accent_color=(0, 200, 255),     # gold
        glow_color=(60, 160, 220),
        name="Classic Gold",
    )
    return img, tip


def neon_dream(w, h):
    img, tip = _draw_base_hookah(
        w, h,
        base_color=(180, 30, 200),      # magenta
        accent_color=(255, 200, 0),     # cyan-ish
        glow_color=(220, 80, 255),
        name="Neon Dream",
    )
    # Extra neon halo
    cv2.circle(img, (w // 2, int(h * 0.74)), int(w * 0.46),
               (255, 200, 0, 90), 2, cv2.LINE_AA)
    return img, tip


def royal_ruby(w, h):
    img, tip = _draw_base_hookah(
        w, h,
        base_color=(40, 40, 180),       # deep red
        accent_color=(0, 215, 255),     # gold accents
        glow_color=(80, 60, 220),
        name="Royal Ruby",
    )
    return img, tip


def emerald_mist(w, h):
    img, tip = _draw_base_hookah(
        w, h,
        base_color=(90, 160, 40),       # emerald
        accent_color=(200, 255, 200),
        glow_color=(140, 210, 80),
        name="Emerald Mist",
    )
    return img, tip


def cosmic_purple(w, h):
    img, tip = _draw_base_hookah(
        w, h,
        base_color=(120, 40, 90),
        accent_color=(255, 255, 255),
        glow_color=(200, 100, 160),
        name="Cosmic Purple",
    )
    # sparkle stars on base
    cx, cy = w // 2, int(h * 0.74)
    rng = np.random.default_rng(7)
    for _ in range(18):
        sx = cx + int(rng.integers(-int(w * 0.35), int(w * 0.35)))
        sy = cy + int(rng.integers(-int(h * 0.15), int(h * 0.15)))
        cv2.circle(img, (sx, sy), 1, (255, 255, 255, 255), -1, cv2.LINE_AA)
    return img, tip


DESIGNS = [
    ("Classic Gold", classic_gold),
    ("Neon Dream", neon_dream),
    ("Royal Ruby", royal_ruby),
    ("Emerald Mist", emerald_mist),
    ("Cosmic Purple", cosmic_purple),
]
