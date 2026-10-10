from veil.tracks import EXPIRED, REJECTED, Tracker


def _pit(x, y):
    return {"cls": "pit", "x": x, "y": y, "radius": 0.3}


def test_a_sliding_detection_is_rejected_and_never_confirmed():
    tracker = Tracker()
    for step in range(40):
        travel = step * 0.05
        tracker.update([_pit(0.2, travel + 1.7)], travel)
        tracker.drop_stale()
    assert tracker.rejected()
    assert all(track.reason == "inconsistent_ground" for track in tracker.rejected())
    assert not tracker.confirmed()


def test_a_stable_detection_is_confirmed():
    tracker = Tracker()
    for step in range(30):
        tracker.update([_pit(0.35, 5.0)], step * 0.05)
        tracker.drop_stale()
    assert len(tracker.confirmed()) == 1
    assert tracker.confirmed()[0].outcome == "confirmed"
    assert not tracker.rejected()


def test_a_quiet_track_expires_instead_of_being_called_inconsistent():
    tracker = Tracker()
    tracker.update([_pit(0.1, 2.0)], 0.0)
    tracker.update([_pit(0.1, 2.0)], 0.1)
    for step in range(8):
        tracker.update([], 0.2 + step)
        tracker.drop_stale()
    assert len(tracker.expired()) == 1
    assert tracker.expired()[0].outcome == EXPIRED
    assert tracker.expired()[0].reason == "missed_detections"
    assert not tracker.rejected()


def test_rejection_survives_after_the_track_goes_quiet():
    tracker = Tracker()
    for step in range(40):
        travel = step * 0.05
        tracker.update([_pit(0.0, travel + 1.5)], travel)
    for step in range(8):
        tracker.update([], 2.0 + step)
        tracker.drop_stale()
    assert tracker.rejected()
    assert tracker.rejected()[0].outcome == REJECTED
    assert not tracker.expired() or tracker.expired()[0].outcome != REJECTED
