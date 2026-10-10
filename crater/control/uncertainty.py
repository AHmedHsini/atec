"""How old a fix is allowed to be before the circle grows. Owner: Hammoud.

Uncertainty is not a fake constant. It grows when the detector is unsure,
when the hazard is far (the ground-plane ray is grazing), and when the
packet itself is aging during a dropout.
"""

from __future__ import annotations

import math


def uncertainty_m(confidence: float, x: float, y: float, age_s: float) -> float:
    confidence = min(1.0, max(0.0, confidence))
    age_s = max(0.0, age_s)
    slant = math.hypot(x, y)
    return (1.0 - confidence) * 1.6 + 0.035 * slant + 0.55 * age_s
