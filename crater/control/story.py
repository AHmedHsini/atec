"""Crater chronicle: telemetry turned into an operator story. Owner: Hammoud.

The chronicle does not invent hazards. Every line is tied to a track id,
a blackout edge, or mission start. Chapters change only when the link or
the scene actually changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoryLine:
    time_s: float
    chapter: str
    text: str


class Chronicle:
    def __init__(self) -> None:
        self.lines: list[StoryLine] = []
        self.chapter = "Descent"
        self._seen: set[int] = set()
        self._ice_marked = False
        self._in_blackout = False
        self._started = False
        self._near_lander = False

    def start(self, time_s: float = 0.0) -> None:
        if self._started:
            return
        self._started = True
        self._emit(
            time_s,
            "Descent",
            "SCOUT-1 is on the crater floor. Site Alpha is the origin. "
            "The radio can carry coordinates, not pictures.",
        )

    def on_packet(self, time_s: float, tracks: list) -> None:
        if not tracks:
            return
        for track in tracks:
            if track.track_id in self._seen:
                continue
            self._seen.add(track.track_id)
            where = _where(track.x, track.y)
            self._emit(
                time_s,
                "First Survey",
                f"Track {track.track_id}, {_label(track.kind)}, {where}. "
                f"Confidence {track.confidence:.0%}.",
            )
            if track.kind == "ice" and not self._ice_marked:
                self._ice_marked = True
                self._emit(
                    time_s,
                    "The Find",
                    f"Bright return on track {track.track_id}. Mark it for a second pass. "
                    "The picture stays aboard; only this fix comes home.",
                )
            if track.kind == "rock" and not self._near_lander and (track.x ** 2 + track.y ** 2) ** 0.5 < 6.0:
                self._near_lander = True
                self._emit(
                    time_s,
                    "First Survey",
                    f"Rock on track {track.track_id} is inside 6 m of the lander. Hold the traverse.",
                )

    def on_link(self, time_s: float, blackout: bool) -> None:
        if blackout and not self._in_blackout:
            self._in_blackout = True
            self._emit(
                time_s,
                "Blackout",
                "Carrier gone. Regolith ate the high-frequency link. "
                "Last fixes stay on the board and start to age. Do not treat them as live.",
            )
        elif not blackout and self._in_blackout:
            self._in_blackout = False
            self._emit(
                time_s,
                "Signal Restored",
                "Carrier is back. Fresh coordinates replace the aged ones. The map is live again.",
            )

    def finish(self, time_s: float, delivered: int, sent: int) -> None:
        ratio = 0.0 if sent == 0 else delivered / sent
        self._emit(
            time_s,
            "Debrief",
            f"Pass complete. {len(self._seen)} tracks named. "
            f"The link delivered {delivered} of {sent} packets ({ratio:.0%}).",
        )

    def _emit(self, time_s: float, chapter: str, text: str) -> None:
        self.chapter = chapter
        self.lines.append(StoryLine(time_s, chapter, text))


def _label(kind: str) -> str:
    return {"rock": "a rock hazard", "crater": "a crater rim", "ice": "a bright ice-like patch"}.get(kind, kind)


def _where(x: float, y: float) -> str:
    forward = f"{abs(x):.1f} m {'forward' if x >= 0 else 'behind'}"
    if abs(y) < 0.15:
        lateral = "on the centerline"
    else:
        lateral = f"{abs(y):.1f} m {'right' if y > 0 else 'left'}"
    return f"{forward}, {lateral} of Site Alpha"
