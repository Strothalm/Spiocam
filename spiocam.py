#!/usr/bin/env python3
"""Spiocam - Koerper- und Handgesten-Tracking per Webcam (720p).

Start:  python spiocam.py
Tasten: q/ESC = Ende | m = Spiegeln | p = Koerper an/aus | h = Haende an/aus
        f = FPS/Info an/aus | s = Screenshot
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gestures import Hand, GestureSmoother, classify

MODEL_DIR = Path(__file__).parent / "models"
MODELS = {
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    ),
    "pose": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30),
    (30, 32), (28, 32), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
]

GREEN, CYAN, ORANGE, WHITE = (80, 220, 80), (230, 200, 40), (40, 160, 255), (255, 255, 255)


def ensure_model(kind: str) -> str:
    name, url = MODELS[kind]
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"Lade Modell {name} (einmalig) ...")
        urllib.request.urlretrieve(url, path)
    return str(path)


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    # Unter Windows ist DirectShow meist schneller/stabiler, unter Linux V4L2.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else (
        cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    # MJPG ist bei aelteren Logitech-Kameras Voraussetzung fuer 720p@30.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {index} konnte nicht geoeffnet werden.")
    return cap


def draw_skeleton(img, pts, connections, color, radius=3, thickness=2, vis=None):
    h, w = img.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in pts]
    for a, b in connections:
        if a < len(xy) and b < len(xy):
            if vis and (vis[a] < 0.5 or vis[b] < 0.5):
                continue
            cv2.line(img, xy[a], xy[b], color, thickness, cv2.LINE_AA)
    for i, p in enumerate(xy):
        if vis and vis[i] < 0.5:
            continue
        cv2.circle(img, p, radius, WHITE, -1, cv2.LINE_AA)


def put_label(img, text, org, color):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="Kamera-Index (Standard 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--no-pose", action="store_true", help="Koerper-Tracking aus (spart CPU)")
    ap.add_argument("--no-mirror", action="store_true", help="Bild nicht spiegeln")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    print(f"Kamera laeuft mit {int(cap.get(3))}x{int(cap.get(4))}")

    hand_det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model("hand")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    pose_det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model("pose")),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    show_pose, show_hands, show_info = not args.no_pose, True, True
    mirror = not args.no_mirror
    smoother = GestureSmoother(window=7)
    t0, last_ts, fps = time.monotonic(), -1, 0.0
    prev = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Kamerabild mehr.")
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - t0) * 1000)
        ts = max(ts, last_ts + 1)  # Timestamps muessen streng steigen
        last_ts = ts

        if show_pose:
            res = pose_det.detect_for_video(mp_img, ts)
            for pose in res.pose_landmarks:
                draw_skeleton(frame, pose, POSE_CONNECTIONS, CYAN,
                              vis=[p.visibility for p in pose])

        if show_hands:
            res = hand_det.detect_for_video(mp_img, ts)
            present = set()
            for i, (lms, handed) in enumerate(zip(res.hand_landmarks, res.handedness)):
                label = handed[0].category_name
                # Beim gespiegelten Bild ist die MediaPipe-Zuordnung umgekehrt zur Person.
                key = f"{i}"
                present.add(key)
                hand = Hand([(p.x, p.y, p.z) for p in lms], label)
                g = smoother.update(key, classify(hand))
                draw_skeleton(frame, lms, HAND_CONNECTIONS, GREEN)
                x = int(min(p.x for p in lms) * frame.shape[1])
                y = int(min(p.y for p in lms) * frame.shape[0])
                put_label(frame, f"{label}: {g or '-'}", (max(x, 6), max(y - 12, 30)), ORANGE)
            smoother.drop_missing(present)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
        prev = now
        if show_info:
            cv2.putText(frame, f"{fps:4.1f} FPS  {frame.shape[1]}x{frame.shape[0]}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        cv2.imshow("Spiocam", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("m"):
            mirror = not mirror
        elif k == ord("p"):
            show_pose = not show_pose
        elif k == ord("h"):
            show_hands = not show_hands
        elif k == ord("f"):
            show_info = not show_info
        elif k == ord("s"):
            name = f"spiocam_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(name, frame)
            print("Gespeichert:", name)

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    pose_det.close()


if __name__ == "__main__":
    main()
