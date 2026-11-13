"""Gestenerkennung auf Basis der 21 MediaPipe-Hand-Landmarks.

Neue Geste hinzufuegen: einfach eine Funktion mit @gesture("Name") schreiben.
Sie bekommt ein `Hand`-Objekt und gibt True/False zurueck. Die Reihenfolge
der Registrierung = Prioritaet (die erste passende Geste gewinnt).
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

# Landmark-Indizes (MediaPipe Hands)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}


def _dist(a, b) -> float:
    return math.dist((a[0], a[1], a[2]), (b[0], b[1], b[2]))


@dataclass
class Hand:
    """Eine erkannte Hand. `lm` = 21 Punkte als (x, y, z), normalisiert (0..1)."""

    lm: list[tuple[float, float, float]]
    label: str  # "Left" / "Right" (aus Sicht der Person)

    @property
    def size(self) -> float:
        """Handgroesse als Referenz (Handgelenk -> Mittelfinger-Grundgelenk)."""
        return max(_dist(self.lm[WRIST], self.lm[MIDDLE_MCP]), 1e-6)

    def finger_extended(self, name: str) -> bool:
        """Finger gilt als gestreckt, wenn die Spitze weiter vom Handgelenk
        entfernt ist als das mittlere Gelenk (drehungsinvariant)."""
        mcp, pip, tip = FINGERS[name]
        return _dist(self.lm[tip], self.lm[WRIST]) > _dist(self.lm[pip], self.lm[WRIST]) * 1.05

    @property
    def thumb_extended(self) -> bool:
        """Daumen gestreckt, wenn die Spitze deutlich vom Zeigefinger-Grundgelenk absteht."""
        return _dist(self.lm[THUMB_TIP], self.lm[INDEX_MCP]) > self.size * 0.9

    def extended(self) -> dict[str, bool]:
        d = {n: self.finger_extended(n) for n in FINGERS}
        d["thumb"] = self.thumb_extended
        return d

    def pinch_distance(self) -> float:
        """Abstand Daumen-/Zeigefingerspitze relativ zur Handgroesse."""
        return _dist(self.lm[THUMB_TIP], self.lm[INDEX_TIP]) / self.size

    def thumb_direction(self) -> str:
        """'up' / 'down' / 'side' - Richtung des Daumens im Bild (y waechst nach unten)."""
        dy = self.lm[THUMB_TIP][1] - self.lm[THUMB_MCP][1]
        dx = self.lm[THUMB_TIP][0] - self.lm[THUMB_MCP][0]
        if abs(dy) < abs(dx) * 1.2:
            return "side"
        return "up" if dy < 0 else "down"


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
GESTURES: list[tuple[str, callable]] = []


def gesture(name: str):
    def deco(fn):
        GESTURES.append((name, fn))
        return fn

    return deco


def _four_curled(h: Hand) -> bool:
    return not any(h.finger_extended(n) for n in FINGERS)


@gesture("OK")
def _ok(h: Hand) -> bool:
    e = h.extended()
    return h.pinch_distance() < 0.35 and e["middle"] and e["ring"] and e["pinky"]


@gesture("Pinch")
def _pinch(h: Hand) -> bool:
    e = h.extended()
    # Zeigefingerspitze muss "vorne" sein, sonst ist es eine Faust mit Daumen am Zeigefinger.
    index_out = _dist(h.lm[INDEX_TIP], h.lm[WRIST]) > h.size * 1.0
    return (h.pinch_distance() < 0.30 and index_out
            and not e["middle"] and not e["ring"] and not e["pinky"])


@gesture("Daumen hoch")
def _thumbs_up(h: Hand) -> bool:
    return _four_curled(h) and h.thumb_extended and h.thumb_direction() == "up"


@gesture("Daumen runter")
def _thumbs_down(h: Hand) -> bool:
    return _four_curled(h) and h.thumb_extended and h.thumb_direction() == "down"


@gesture("Faust")
def _fist(h: Hand) -> bool:
    return _four_curled(h) and not h.thumb_extended


@gesture("Rock")
def _rock(h: Hand) -> bool:
    e = h.extended()
    return e["index"] and e["pinky"] and not e["middle"] and not e["ring"]


@gesture("Peace")
def _peace(h: Hand) -> bool:
    e = h.extended()
    return e["index"] and e["middle"] and not e["ring"] and not e["pinky"]


@gesture("Drei")
def _three(h: Hand) -> bool:
    e = h.extended()
    return e["index"] and e["middle"] and e["ring"] and not e["pinky"]


@gesture("Zeigen")
def _point(h: Hand) -> bool:
    e = h.extended()
    return e["index"] and not e["middle"] and not e["ring"] and not e["pinky"]


@gesture("Offene Hand")
def _open(h: Hand) -> bool:
    e = h.extended()
    return all(e[n] for n in FINGERS) and e["thumb"]


@gesture("Vier")
def _four(h: Hand) -> bool:
    e = h.extended()
    return all(e[n] for n in FINGERS) and not e["thumb"]


def classify(hand: Hand) -> str | None:
    for name, fn in GESTURES:
        if fn(hand):
            return name
    return None


class GestureSmoother:
    """Mehrheitsentscheid ueber die letzten N Frames, damit das Label nicht flackert."""

    def __init__(self, window: int = 7):
        self.window = window
        self._hist: dict[str, deque] = {}

    def update(self, key: str, value: str | None) -> str | None:
        q = self._hist.setdefault(key, deque(maxlen=self.window))
        q.append(value)
        top, n = Counter(q).most_common(1)[0]
        return top if n >= (len(q) // 2 + 1) else None

    def drop_missing(self, present: set[str]) -> None:
        for k in list(self._hist):
            if k not in present:
                del self._hist[k]
