"""Fun Hookah Virtual Room - webcam simulator you can actually smoke from.

The hookah stands stationary at the bottom of the frame. Its hose curves out
to wherever your hand is (detected via skin-color tracking). Bring the
mouthpiece up to your mouth and smoke starts pouring out - move it away and
the smoke stops.

Controls:
    1..5   Switch hookah design
    S      Save a screenshot to ./screenshots/
    F      Toggle FPS overlay
    D      Toggle hand/mouth debug markers
    H      Toggle help overlay
    M      Mirror hookah to the other side of the frame
    Q/ESC  Quit
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hookah_designs import DESIGNS, PALETTES


# ---------------------------------------------------------------------------
# Smoke particle system - denser, longer-lived, more opaque than before
# ---------------------------------------------------------------------------

@dataclass
class SmokeParticle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    life: float
    max_life: float
    tint: Tuple[int, int, int] = (245, 245, 245)


class SmokeSystem:
    def __init__(self, max_particles: int = 600):
        self.particles: List[SmokeParticle] = []
        self.max_particles = max_particles
        self._rng = np.random.default_rng()

    def emit(self, x: int, y: int, count: int = 5, strength: float = 1.0,
             tint: Tuple[int, int, int] = (245, 245, 245)) -> None:
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                return
            angle = self._rng.uniform(-np.pi / 2 - 0.7, -np.pi / 2 + 0.7)
            speed = self._rng.uniform(45, 95) * strength
            life = self._rng.uniform(2.2, 3.6)
            self.particles.append(SmokeParticle(
                x=float(x + self._rng.uniform(-6, 6)),
                y=float(y + self._rng.uniform(-4, 4)),
                vx=float(np.cos(angle) * speed + self._rng.uniform(-18, 18)),
                vy=float(np.sin(angle) * speed),
                radius=float(self._rng.uniform(16, 26) * strength),
                life=life,
                max_life=life,
                tint=tint,
            ))

    def update(self, dt: float) -> None:
        alive: List[SmokeParticle] = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt + np.sin(p.life * 3.0) * 8 * dt
            p.y += p.vy * dt
            p.vy *= 0.988
            p.vx *= 0.985
            p.radius += 18 * dt
            alive.append(p)
        self.particles = alive

    def render(self, frame: np.ndarray) -> None:
        if not self.particles:
            return
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        mask_acc = np.zeros((h, w), dtype=np.float32)
        for p in self.particles:
            if p.x < -80 or p.x > w + 80 or p.y < -80 or p.y > h + 80:
                continue
            alpha = max(0.0, min(1.0, p.life / p.max_life)) * 0.85
            r = int(p.radius)
            cv2.circle(overlay, (int(p.x), int(p.y)), r, p.tint, -1, cv2.LINE_AA)
            cv2.circle(mask_acc, (int(p.x), int(p.y)), r, alpha, -1, cv2.LINE_AA)

        mask_acc = cv2.GaussianBlur(mask_acc, (0, 0), sigmaX=11, sigmaY=11)
        mask_acc = np.clip(mask_acc, 0, 0.95)
        mask_3 = cv2.merge([mask_acc, mask_acc, mask_acc])
        np.copyto(frame, (frame.astype(np.float32) * (1 - mask_3)
                          + overlay.astype(np.float32) * mask_3).astype(np.uint8))


# ---------------------------------------------------------------------------
# Overlay + drawing helpers
# ---------------------------------------------------------------------------

def overlay_bgra(background, sprite_bgra, x, y):
    bh, bw = background.shape[:2]
    sh, sw = sprite_bgra.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + sw, bw), min(y + sh, bh)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    sprite = sprite_bgra[sy1:sy2, sx1:sx2]
    alpha = sprite[..., 3:4].astype(np.float32) / 255.0
    bg = background[y1:y2, x1:x2].astype(np.float32)
    fg = sprite[..., :3].astype(np.float32)
    background[y1:y2, x1:x2] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)


def scale_sprite(sprite, new_w, new_h):
    return cv2.resize(sprite, (max(1, new_w), max(1, new_h)),
                      interpolation=cv2.INTER_AREA)


def draw_dynamic_hose(frame, port, hand, base_color, hl_color, thickness=14):
    p0 = np.array(port, dtype=float)
    p2 = np.array(hand, dtype=float)
    dist = float(np.linalg.norm(p2 - p0))
    if dist < 6:
        return
    mid = (p0 + p2) / 2.0
    sag = min(dist * 0.28, 70.0)
    p1 = mid + np.array([0.0, sag])

    pts = []
    for t in np.linspace(0, 1, 40):
        p = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        pts.append((int(p[0]), int(p[1])))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], (10, 10, 10), thickness + 4, cv2.LINE_AA)
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], base_color, thickness, cv2.LINE_AA)
    off = max(1, thickness // 4)
    for i in range(len(pts) - 1):
        p1_h = (pts[i][0], pts[i][1] - off)
        p2_h = (pts[i + 1][0], pts[i + 1][1] - off)
        cv2.line(frame, p1_h, p2_h, hl_color, max(1, thickness // 3), cv2.LINE_AA)


def draw_mouthpiece_at(frame, x, y, base_color, hl_color):
    cv2.circle(frame, (x, y), 20, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 17, base_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 17,
               tuple(int(c * 0.3) for c in base_color), 2, cv2.LINE_AA)
    cv2.circle(frame, (x - 5, y - 5), 5, hl_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 7, (12, 12, 12), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Detection: face, hand (skin color), mouth estimate
# ---------------------------------------------------------------------------

def load_face_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "Your installed OpenCV build is missing CascadeClassifier "
            f"(cv2 {getattr(cv2, '__version__', '?')}). This usually means pip "
            "installed a broken opencv-python 5.0.0 pre-release. Reinstall a "
            "stable 4.x build:\n\n"
            "    pip install --force-reinstall \"opencv-python==4.10.0.84\"\n"
        )
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {path}")
    return cascade


def detect_hand_skin(frame_bgr: np.ndarray,
                     face_bbox: Optional[Tuple[int, int, int, int]],
                     prev_hand: Optional[Tuple[int, int]] = None
                     ) -> Optional[Tuple[int, int]]:
    """Find the biggest skin-colored blob outside the face; return its centroid."""
    scale = 0.4
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)

    hsv_mask = cv2.inRange(hsv, (0, 30, 60), (25, 200, 255)) | \
        cv2.inRange(hsv, (155, 30, 60), (180, 200, 255))
    ycrcb_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    if face_bbox is not None:
        fx, fy, fw, fh = [int(v * scale) for v in face_bbox]
        pad_w = int(fw * 0.30)
        pad_top = int(fh * 0.40)
        pad_bot = int(fh * 0.10)
        cv2.rectangle(mask,
                      (max(0, fx - pad_w), max(0, fy - pad_top)),
                      (min(mask.shape[1], fx + fw + pad_w),
                       min(mask.shape[0], fy + fh + pad_bot)),
                      0, -1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    scored = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'] / scale)
        cy = int(M['m01'] / M['m00'] / scale)
        score = area / (scale ** 2)
        if prev_hand is not None:
            d = float(np.hypot(cx - prev_hand[0], cy - prev_hand[1]))
            score -= d * 40
        scored.append((score, (cx, cy)))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def estimate_mouth(face_bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    fx, fy, fw, fh = face_bbox
    return (fx + fw // 2, fy + int(fh * 0.78))


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(frame, design_name, design_idx, num_designs,
             face_ok, hand_ok, smoking, show_fps, fps, help_on):
    h, w = frame.shape[:2]

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 46), (25, 25, 30), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, dst=frame)

    cv2.putText(frame, "  Fun Hookah Virtual Room", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

    info = f"{design_name}  [{design_idx + 1}/{num_designs}]"
    (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, info, (w - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def pill(x, y, label, value, on_color):
        base = frame.copy()
        cv2.rectangle(base, (x, y - 22), (x + 190, y + 8), (20, 20, 20), -1)
        cv2.addWeighted(base, 0.6, frame, 0.4, 0, dst=frame)
        cv2.putText(frame, label, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x + 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    on_color, 1, cv2.LINE_AA)

    pill(10, h - 20, "face:", "detected" if face_ok else "not found",
         (100, 255, 100) if face_ok else (140, 140, 140))
    pill(210, h - 20, "hand:", "tracked" if hand_ok else "not found",
         (100, 255, 100) if hand_ok else (140, 140, 140))
    pill(410, h - 20, "smoke:", "PUFFING" if smoking else "waiting",
         (100, 220, 255) if smoking else (120, 120, 120))

    if show_fps:
        cv2.putText(frame, f"{fps:5.1f} fps", (w - 110, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    if help_on:
        lines = [
            "1-5 change design    M mirror hookah    S screenshot    F fps    D debug    H help    Q quit",
            "Bring the golden mouthpiece up to your mouth to start smoking. Move it away to stop.",
        ]
        panel = frame.copy()
        cv2.rectangle(panel, (0, 48), (w, 48 + 44), (20, 20, 20), -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, dst=frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 68 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (210, 255, 255), 1, cv2.LINE_AA)


def draw_face_marker(frame, face, smoking):
    fx, fy, fw, fh = face
    color = (0, 220, 120) if smoking else (140, 140, 140)
    thickness = 2
    L = max(10, fw // 6)
    for (px, py, dx, dy) in [
        (fx, fy, 1, 1), (fx + fw, fy, -1, 1),
        (fx, fy + fh, 1, -1), (fx + fw, fy + fh, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * L), color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
    if not cap.isOpened():
        raise RuntimeError("Could not open any webcam. Is one connected?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def build_sprite_cache():
    ref_w, ref_h = 340, 620
    return [(name, fac(ref_w, ref_h)[0], fac(ref_w, ref_h)[1])
            for name, fac in DESIGNS]


def main() -> int:
    cap = open_camera(0)
    cascade = load_face_cascade()
    smoke = SmokeSystem()
    sprites = build_sprite_cache()

    design_idx = 0
    hand_pos: Optional[Tuple[int, int]] = None
    hand_last_seen = 0.0
    hand_hold_time = 0.6  # keep drawing hose at last hand position for this long
    mirror = False        # if True, hookah on the right
    show_fps = True
    show_debug = False
    help_on = True

    last_time = time.time()
    fps_ema = 0.0

    Path("screenshots").mkdir(exist_ok=True)
    window = "Fun Hookah Virtual Room"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; exiting.", file=sys.stderr)
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = min(0.1, now - last_time)
        last_time = now
        inst_fps = 1.0 / max(dt, 1e-3)
        fps_ema = inst_fps if fps_ema == 0 else fps_ema * 0.9 + inst_fps * 0.1

        # ---- Faces ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces_raw = cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        faces = faces_raw.tolist() if len(faces_raw) else []
        primary_face = max(faces, key=lambda f: f[2] * f[3]) if faces else None

        # ---- Hand ----
        detected = detect_hand_skin(frame, primary_face, hand_pos)
        if detected is not None:
            if hand_pos is None:
                hand_pos = detected
            else:
                hand_pos = (int(hand_pos[0] * 0.5 + detected[0] * 0.5),
                            int(hand_pos[1] * 0.5 + detected[1] * 0.5))
            hand_last_seen = now
        elif hand_pos is not None and now - hand_last_seen > hand_hold_time:
            hand_pos = None

        # ---- Sprite placement ----
        name, base_sprite, port_ref = sprites[design_idx]
        target_h = int(h * 0.62)
        aspect = base_sprite.shape[1] / base_sprite.shape[0]
        target_w = int(target_h * aspect)
        sprite = scale_sprite(base_sprite, target_w, target_h)

        if mirror:
            sprite = cv2.flip(sprite, 1)
            port_px_x = base_sprite.shape[1] - port_ref[0]
        else:
            port_px_x = port_ref[0]

        margin = 24
        if mirror:
            sx = w - target_w - margin
        else:
            sx = margin
        sy = h - target_h - 12

        overlay_bgra(frame, sprite, sx, sy)

        # Port position in frame
        scale_x = target_w / base_sprite.shape[1]
        scale_y = target_h / base_sprite.shape[0]
        port_x = sx + int(port_px_x * scale_x)
        port_y = sy + int(port_ref[1] * scale_y)

        # ---- Hose target ----
        pal = PALETTES[design_idx]
        hose_base = pal[4]
        hose_hl = pal[5]
        metal_base = pal[2]
        metal_hl = pal[3]

        if hand_pos is not None:
            hose_end = hand_pos
        else:
            # Idle drape - hang toward the center of the frame
            hose_end = (port_x + (int(target_w * 0.4) * (-1 if mirror else 1)),
                        min(h - 30, sy + target_h - 30))

        draw_dynamic_hose(frame, (port_x, port_y), hose_end,
                          hose_base, hose_hl, thickness=15)
        draw_mouthpiece_at(frame, hose_end[0], hose_end[1], metal_base, metal_hl)

        # ---- Smoke gating ----
        smoking = False
        mouth = None
        if primary_face is not None and hand_pos is not None:
            mouth = estimate_mouth(primary_face)
            dist = float(np.hypot(hand_pos[0] - mouth[0], hand_pos[1] - mouth[1]))
            fw = primary_face[2]
            threshold = fw * 0.65
            if dist < threshold:
                smoking = True
                closeness = 1.0 - dist / threshold
                strength = 1.0 + closeness * 0.6
                count = int(5 + closeness * 5)
                smoke.emit(mouth[0], mouth[1] + 4, count=count,
                           strength=strength, tint=(245, 245, 250))

        smoke.update(dt)
        smoke.render(frame)

        # ---- Face marker ----
        if primary_face is not None:
            draw_face_marker(frame, primary_face, smoking)

        # ---- Debug ----
        if show_debug:
            if hand_pos is not None:
                cv2.circle(frame, hand_pos, 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "hand", (hand_pos[0] + 12, hand_pos[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if mouth is not None:
                cv2.circle(frame, mouth, 8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "mouth", (mouth[0] + 10, mouth[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (port_x, port_y), 5, (255, 0, 255), 2, cv2.LINE_AA)

        # ---- HUD ----
        draw_hud(frame, name, design_idx, len(sprites),
                 face_ok=primary_face is not None,
                 hand_ok=hand_pos is not None,
                 smoking=smoking, show_fps=show_fps, fps=fps_ema,
                 help_on=help_on)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            fname = time.strftime("screenshots/hookah_%Y%m%d_%H%M%S.png")
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('h'):
            help_on = not help_on
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('m'):
            mirror = not mirror
        elif ord('1') <= key <= ord('9'):
            n = key - ord('0') - 1
            if 0 <= n < len(sprites):
                design_idx = n

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
