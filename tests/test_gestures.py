"""Tests mit synthetischen Landmarks (Hand zeigt nach oben, y waechst nach unten)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gestures import Hand, classify, GestureSmoother  # noqa: E402

MCP_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(ext, thumb="out", pinch=False):
    lm = [(0.5, 0.9, 0.0)] * 21
    lm = list(lm)
    lm[0] = (0.5, 0.9, 0)
    for n, b in BASE.items():
        x = MCP_X[n]
        lm[b] = (x, 0.70, 0)
        if ext[n]:
            lm[b + 1] = (x, 0.55, 0); lm[b + 2] = (x, 0.45, 0); lm[b + 3] = (x, 0.38, 0)
        else:
            lm[b + 1] = (x, 0.62, 0); lm[b + 2] = (x, 0.70, 0); lm[b + 3] = (x, 0.78, 0)
    lm[1] = (0.42, 0.85, 0); lm[2] = (0.38, 0.78, 0)
    if thumb == "out":
        lm[3] = (0.30, 0.72, 0); lm[4] = (0.24, 0.68, 0)
    elif thumb == "up":
        lm[3] = (0.36, 0.65, 0); lm[4] = (0.36, 0.52, 0)
    elif thumb == "down":
        lm[3] = (0.36, 0.90, 0); lm[4] = (0.36, 1.03, 0)
    else:  # tucked
        lm[3] = (0.43, 0.76, 0); lm[4] = (0.46, 0.74, 0)
    if pinch:
        lm[4] = (lm[8][0] + 0.01, lm[8][1] + 0.01, 0)
    return Hand(lm, "Right")


E = dict(index=True, middle=True, ring=True, pinky=True)
C = dict(index=False, middle=False, ring=False, pinky=False)


def check(name, hand):
    got = classify(hand)
    assert got == name, f"erwartet {name!r}, bekommen {got!r}"


def test_all():
    check("Offene Hand", make_hand(E, "out"))
    check("Vier", make_hand(E, "tucked"))
    check("Faust", make_hand(C, "tucked"))
    check("Daumen hoch", make_hand(C, "up"))
    check("Daumen runter", make_hand(C, "down"))
    check("Peace", make_hand({**C, "index": True, "middle": True}, "tucked"))
    check("Zeigen", make_hand({**C, "index": True}, "tucked"))
    check("Rock", make_hand({**C, "index": True, "pinky": True}, "tucked"))
    check("Drei", make_hand({**E, "pinky": False}, "tucked"))
    pinch = make_hand(C, "out")
    pinch.lm[7] = (0.44, 0.60, 0); pinch.lm[8] = (0.40, 0.56, 0); pinch.lm[4] = (0.41, 0.57, 0)
    check("Pinch", pinch)
    check("OK", make_hand({**E, "index": True}, "out", pinch=True))


def test_smoother():
    s = GestureSmoother(5)
    out = [s.update("0", g) for g in ["Faust", "Faust", "Peace", "Faust", "Faust"]]
    assert out[-1] == "Faust"


if __name__ == "__main__":
    test_all(); test_smoother(); print("OK")
