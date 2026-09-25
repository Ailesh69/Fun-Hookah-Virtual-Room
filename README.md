# Fun Hookah Virtual Room

A playful OpenCV webcam app that puts a fancy virtual hookah next to your face,
lets you pick from several designs, blows animated smoke out of the mouthpiece,
and lets you pass the hookah around when more than one friend is on camera.

Everything is drawn programmatically with OpenCV — no image assets required.

## Features

- Real-time face detection with Haar cascades (multi-face supported).
- 5 built-in hookah designs: Classic Gold, Neon Dream, Royal Ruby, Emerald
  Mist, Cosmic Purple.
- Particle-based smoke that rises, drifts and dissipates.
- "Pass the hookah" mechanic — the active hookah follows the current holder;
  press `P` to hand it to the next detected face, or `A` for auto-rotate.
- Big puff on `SPACE`, screenshot on `S`.
- Selfie-mirrored view with an overlay HUD.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+ and a working webcam.

## Run

```bash
python hookah_simulator.py
```

## Controls

| Key | Action |
| --- | --- |
| `1`–`5` | Switch hookah design |
| `SPACE` | Big smoke puff |
| `P` | Pass the hookah to the next face |
| `A` | Toggle auto-pass (rotates every few seconds) |
| `S` | Save a screenshot into `./screenshots/` |
| `F` | Toggle FPS overlay |
| `H` | Toggle on-screen help |
| `Q` / `Esc` | Quit |

## How it works

- `hookah_designs.py` builds each hookah as a BGRA sprite using OpenCV
  primitives (ellipses, gradients, glowing coals, a curly hose). It also
  reports the mouthpiece tip so smoke can originate from the right spot.
- `hookah_simulator.py` grabs frames from the webcam, runs face detection,
  scales and alpha-blends the current design next to the "holder" face, and
  emits smoke particles that rise, swirl and fade.
- The smoke system draws all particles into an accumulation mask, blurs it,
  and blends the tinted overlay into the frame — cheap, but it looks smoky.

## Notes

- Face detection uses the OpenCV-bundled Haar cascade
  (`haarcascade_frontalface_default.xml`), which ships with `opencv-python`.
- If no webcam is detected on index 0, the app tries indices 1–3 before
  giving up.
- Nothing leaves your machine — this is purely local video processing.
