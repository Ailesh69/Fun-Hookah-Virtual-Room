# Fun Hookah Virtual Room

An OpenCV webcam app that puts a virtual hookah on your desk. The hose
follows your hand around the frame - lift the mouthpiece to your mouth and
smoke starts pouring out. Move it away and the smoke stops.

Nothing leaves your machine. No image assets - everything is drawn with
OpenCV primitives.

## What it does

- Real-time face detection (Haar cascade).
- Skin-color hand tracking (HSV + YCrCb, face region excluded).
- A stationary hookah at the bottom of the frame - glass base with tinted
  liquid, cylindrical-shaded metal stem, clay bowl with foil, glowing
  embers.
- A dynamic hose (quadratic bezier) that curves from the hookah's side port
  to your tracked hand, with a metal mouthpiece at the end.
- Smoke gate: proximity between the mouthpiece and your estimated mouth
  drives emission. No proximity, no smoke.
- Five color palettes to choose from.

## Setup

```bash
python -m venv venv
venv\Scripts\activate         # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python hookah_simulator.py
```

Requires Python 3.9+ and a working webcam. If pip somehow installs the
broken `opencv-python 5.0.0.93` pre-release, force a stable build:

```bash
pip install --force-reinstall "opencv-python==4.10.0.84"
```

## Controls

| Key       | Action                                       |
| --------- | -------------------------------------------- |
| `1`..`5`  | Switch hookah design                         |
| `M`       | Mirror hookah to the other side of the frame |
| `D`       | Toggle hand/mouth debug markers              |
| `F`       | Toggle FPS overlay                           |
| `H`       | Toggle on-screen help                        |
| `S`       | Save a screenshot to `./screenshots/`        |
| `Q`/`Esc` | Quit                                         |

## How the smoke gate works

Each frame the app:

1. Detects the largest face and estimates the mouth at roughly 78% down
   from the top of the face box.
2. Detects a skin-colored blob outside the face - the biggest one within
   reason wins, weighted toward its last known position for stability.
3. Measures the distance from the tracked hand (mouthpiece) to the mouth.
4. If that distance is less than about 65% of the face width, the app
   emits smoke particles centered on the mouth. Otherwise nothing puffs.

The smoke system itself uses larger, longer-lived, more opaque particles
than a typical fog effect - so puffs read clearly against a normal indoor
background.

## Tips for good tracking

- Sit under decent, even lighting. Skin-color detection struggles with
  colored bulbs or heavy shadows.
- Keep your hand fully in the frame and away from the face when you're
  not smoking - the hookah hose will follow it around.
- Wear sleeves. A big bare arm can win against a small hand for the
  "biggest skin blob" score.
- Press `D` if you want to see where the app thinks your hand and mouth
  are - useful for tuning distance / lighting.
