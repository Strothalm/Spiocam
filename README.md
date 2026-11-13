# Spiocam

Koerper- und Handgesten-Tracking per Webcam (ausgelegt auf 720p, z. B. aeltere Logitech).

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python spiocam.py
```
Die Modelle (MediaPipe) werden beim ersten Start automatisch nach `models/` geladen.

## Bedienung
`q`/`ESC` Ende · `m` Spiegeln · `p` Koerper an/aus · `h` Haende an/aus · `f` Info · `s` Screenshot

Optionen: `--camera 1`, `--width/--height/--fps`, `--no-pose`, `--no-mirror`, `--max-hands`

## Gesten
Aktuell: Offene Hand, Vier, Faust, Daumen hoch/runter, Peace, Zeigen, Rock, Drei, OK, Pinch.
Neue Geste = eine Funktion mit `@gesture("Name")` in `gestures.py`.

Tests: `python tests/test_gestures.py`
