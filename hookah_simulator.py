"""Fun Hookah Virtual Room - an OpenCV camera simulator.

Detects faces from the webcam, drops a hookah beside each friend, and puffs
animated smoke rings out of the mouthpiece. Pass the hookah around by pressing
P, switch designs with 1-5, and hit SPACE for a big puff.

Controls:
    1..5   Switch hookah design
    SPACE  Big puff (extra smoke burst)
    P      Pass the hookah to the next detected face
    A      Auto-pass mode on/off (rotate every few seconds)
    F      Toggle FPS overlay
    S      Save a screenshot to ./screenshots/
    H      Toggle help overlay
    Q/ESC  Quit
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from hookah_designs import DESIGNS


# ---------------------------------------------------------------------------
# Smoke particle system
# ---------------------------------------------------------------------------

@dataclass
class SmokeParticle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    life: float          # seconds remaining
    max_life: float
    tint: Tuple[int, int, int] = (230, 230, 230)


class SmokeSystem:
    def __init__(self, max_particles: int = 400):
        self.particles: List[SmokeParticle] = []
        self.max_particles = max_particles
        self._rng = np.random.default_rng()

    def emit(self, x: int, y: int, count: int = 3, strength: float = 1.0,
             tint: Tuple[int, int, int] = (230, 230, 230)) -> None:
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                return
            angle = self._rng.uniform(-np.pi / 2 - 0.5, -np.pi / 2 + 0.5)
            speed = self._rng.uniform(30, 70) * strength
            life = self._rng.uniform(1.4, 2.4)
            self.particles.append(SmokeParticle(
                x=float(x + self._rng.uniform(-4, 4)),
                y=float(y + self._rng.uniform(-2, 2)),
                vx=np.cos(angle) * speed + self._rng.uniform(-10, 10),
                vy=np.sin(angle) * speed,
                radius=self._rng.uniform(8, 16) * strength,
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
            # Rise, swirl, expand
            p.x += p.vx * dt + np.sin(p.life * 3) * 6 * dt
            p.y += p.vy * dt
            p.vy *= 0.985            # slow the rise
            p.vx *= 0.98
            p.radius += 14 * dt
            alive.append(p)
        self.particles = alive

    def render(self, frame: np.ndarray) -> None:
        if not self.particles:
            return
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        mask_acc = np.zeros((h, w), dtype=np.float32)
        for p in self.particles:
            if p.x < -50 or p.x > w + 50 or p.y < -50 or p.y > h + 50:
                continue
            alpha = max(0.0, min(1.0, p.life / p.max_life)) * 0.55
            r = int(p.radius)
            cv2.circle(overlay, (int(p.x), int(p.y)), r, p.tint, -1, cv2.LINE_AA)
            cv2.circle(mask_acc, (int(p.x), int(p.y)), r, alpha, -1, cv2.LINE_AA)

        mask_acc = cv2.GaussianBlur(mask_acc, (0, 0), sigmaX=9, sigmaY=9)
        mask_acc = np.clip(mask_acc, 0, 0.75)
        mask_3 = cv2.merge([mask_acc, mask_acc, mask_acc])
        np.copyto(frame, (frame.astype(np.float32) * (1 - mask_3)
                          + overlay.astype(np.float32) * mask_3).astype(np.uint8))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def overlay_bgra(background: np.ndarray, sprite_bgra: np.ndarray,
                 x: int, y: int) -> None:
    """Alpha-blend a BGRA sprite onto a BGR background at top-left (x, y)."""
    bh, bw = background.shape[:2]
    sh, sw = sprite_bgra.shape[:2]

    # Clip
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


def scale_sprite(sprite: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    return cv2.resize(sprite, (max(1, new_w), max(1, new_h)),
                      interpolation=cv2.INTER_AREA)


def load_face_cascade() -> cv2.CascadeClassifier:
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {path}")
    return cascade


def draw_hud(frame: np.ndarray, design_name: str, design_idx: int,
             num_designs: int, num_faces: int, holder_idx: int,
             auto_pass: bool, show_fps: bool, fps: float,
             help_on: bool) -> None:
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 44), (25, 25, 30), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, dst=frame)

    title = "  Fun Hookah Virtual Room  "
    cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                (0, 215, 255), 2, cv2.LINE_AA)

    info = f"Design [{design_idx + 1}/{num_designs}]: {design_name}"
    (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, info, (w - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom bar
    status_lines = []
    holder_txt = "—"
    if num_faces > 0:
        holder_txt = f"Friend #{holder_idx + 1}"
    status_lines.append(f"Faces: {num_faces}    Holder: {holder_txt}    "
                        f"Auto-pass: {'ON' if auto_pass else 'off'}")
    if show_fps:
        status_lines.append(f"FPS: {fps:5.1f}")

    for i, line in enumerate(status_lines):
        cv2.putText(frame, line, (10, h - 12 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)

    if help_on:
        help_lines = [
            "1..5 switch design   SPACE puff   P pass   A auto-pass",
            "S screenshot   F fps   H help   Q quit",
        ]
        panel_h = 18 * len(help_lines) + 16
        panel = frame.copy()
        cv2.rectangle(panel, (0, 46), (w, 46 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, dst=frame)
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (10, 46 + 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 255, 255), 1, cv2.LINE_AA)


def draw_face_marker(frame: np.ndarray, face: Tuple[int, int, int, int],
                     is_holder: bool) -> None:
    x, y, w, h = face
    color = (0, 215, 255) if is_holder else (120, 120, 120)
    thickness = 2 if is_holder else 1
    # Corner brackets, not a full box
    L = max(10, w // 6)
    for (px, py, dx, dy) in [
        (x, y, 1, 1), (x + w, y, -1, 1),
        (x, y + h, 1, -1), (x + w, y + h, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * L), color, thickness, cv2.LINE_AA)
    if is_holder:
        cv2.putText(frame, "holding hookah", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        # Try a few more indices as a courtesy
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
    if not cap.isOpened():
        raise RuntimeError("Could not open any webcam. Is one connected?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def build_sprite_cache() -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
    """Pre-render every design at a reference size; we'll scale per-face."""
    cache = []
    ref_w, ref_h = 320, 520
    for name, factory in DESIGNS:
        sprite, tip = factory(ref_w, ref_h)
        cache.append((name, sprite, tip))
    return cache


def main() -> int:
    cap = open_camera(0)
    cascade = load_face_cascade()
    smoke = SmokeSystem()
    sprites = build_sprite_cache()

    design_idx = 0
    holder_idx = 0
    auto_pass = False
    show_fps = True
    help_on = True

    last_time = time.time()
    last_pass_time = last_time
    puff_until = 0.0
    fps_ema = 0.0

    Path("screenshots").mkdir(exist_ok=True)

    window = "Fun Hookah Virtual Room"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; exiting.", file=sys.stderr)
            break

        frame = cv2.flip(frame, 1)  # selfie view
        h, w = frame.shape[:2]

        now = time.time()
        dt = min(0.1, now - last_time)
        last_time = now
        inst_fps = 1.0 / max(dt, 1e-3)
        fps_ema = inst_fps if fps_ema == 0 else fps_ema * 0.9 + inst_fps * 0.1

        # Faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces_raw = cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # Left-to-right, stable ordering
        faces = sorted(faces_raw.tolist(), key=lambda f: f[0]) if len(faces_raw) else []

        if faces:
            if holder_idx >= len(faces):
                holder_idx = 0
            if auto_pass and now - last_pass_time > 3.5:
                holder_idx = (holder_idx + 1) % len(faces)
                last_pass_time = now
        else:
            holder_idx = 0

        # Draw hookah sprite for the holder + smoke
        name, base_sprite, base_tip = sprites[design_idx]
        smoke_origin = None

        for idx, (fx, fy, fw, fh) in enumerate(faces):
            is_holder = idx == holder_idx
            draw_face_marker(frame, (fx, fy, fw, fh), is_holder)
            if not is_holder:
                continue

            # Scale sprite proportional to face
            target_w = int(fw * 1.6)
            aspect = base_sprite.shape[0] / base_sprite.shape[1]
            target_h = int(target_w * aspect)
            sprite = scale_sprite(base_sprite, target_w, target_h)

            # Position: to the right of and slightly below the face
            sx = fx + fw - int(target_w * 0.15)
            sy = fy + int(fh * 0.35)
            # Keep on-screen
            sx = max(0, min(sx, w - target_w))
            sy = max(0, min(sy, h - target_h))

            overlay_bgra(frame, sprite, sx, sy)

            # Smoke origin: scale the base tip to the resized sprite, offset by sx/sy
            tip_x = int(base_tip[0] * target_w / base_sprite.shape[1])
            tip_y = int(base_tip[1] * target_h / base_sprite.shape[0])
            smoke_origin = (sx + tip_x, sy + tip_y)

        # Emit smoke continuously from active hookah
        if smoke_origin is not None:
            smoke.emit(smoke_origin[0], smoke_origin[1], count=3, strength=1.0)
            if now < puff_until:
                smoke.emit(smoke_origin[0], smoke_origin[1], count=8, strength=1.6,
                           tint=(240, 240, 250))
        smoke.update(dt)
        smoke.render(frame)

        draw_hud(frame, design_name=name, design_idx=design_idx,
                 num_designs=len(sprites), num_faces=len(faces),
                 holder_idx=holder_idx, auto_pass=auto_pass,
                 show_fps=show_fps, fps=fps_ema, help_on=help_on)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            puff_until = now + 0.5
        elif key == ord('p'):
            if faces:
                holder_idx = (holder_idx + 1) % len(faces)
                last_pass_time = now
        elif key == ord('a'):
            auto_pass = not auto_pass
            last_pass_time = now
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('h'):
            help_on = not help_on
        elif key == ord('s'):
            fname = time.strftime("screenshots/hookah_%Y%m%d_%H%M%S.png")
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
        elif ord('1') <= key <= ord('9'):
            n = key - ord('0') - 1
            if 0 <= n < len(sprites):
                design_idx = n

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
