import time

from veil.link import LossyLink
from veil.onboard import Onboard
from veil.packet import encode


def _pose(seq: int) -> bytes:
    return encode(2, seq, seq * 10, 0, 0, 0, 100, 0.0, 1.0, 0.05, 0.0)


def test_seeded_loss_is_near_thirty_percent_and_counts_payload_bytes():
    link = LossyLink(loss=0.3, blackout=(1000, 1001), seed=26)
    try:
        for seq in range(1, 201):
            link.send(_pose(seq), 0.0)
    finally:
        link.close()
    assert link.sent == 200
    assert link.bytes_sent == 4000
    assert 0.2 < link.dropped / link.sent < 0.45


def test_closed_link_counts_a_wasted_send_separately():
    link = LossyLink(loss=0.0, blackout=(0.0, 5.0), seed=1)
    try:
        assert link.send(_pose(1), 1.0) is False
        assert link.wasted_blackout_bytes == 20
        assert link.dropped == 1
    finally:
        link.close()
    closed = LossyLink(loss=0.0, blackout=(0.0, 5.0), seed=1)
    try:
        assert closed.send(_pose(1), 1.0) is False
        assert closed.take() == []
    finally:
        closed.close()
    opened = LossyLink(loss=0.0, blackout=(50.0, 51.0), seed=1)
    try:
        assert opened.send(_pose(7), 0.0) is True
        found = []
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline and not found:
            found = opened.take()
            time.sleep(0.02)
        assert len(found) == 1
        assert found[0]["seq"] == 7
        opened.ingest = None
        acks = set()
        while time.perf_counter() < deadline and not acks:
            acks |= opened.poll_acks()
            time.sleep(0.02)
        assert 7 in acks
    finally:
        opened.close()


def test_burst_budget_and_ack_do_not_drop_an_unacknowledged_hazard():
    onboard = Onboard()
    for index in range(5):
        onboard.pending[f"h{index}"] = {"seq": index + 1, "value": 5 - index, "bytes": _pose(index + 1)}
    onboard.pending["pose"] = {"seq": 9, "value": 0.01, "bytes": _pose(9)}
    chosen = onboard.packets(80)
    assert len(chosen) == 4
    assert _pose(9) not in chosen
    onboard.ack({1})
    assert "h0" in onboard.finished
    assert "h1" in onboard.pending


def test_a_hazard_expires_when_the_link_stays_down():
    onboard = Onboard()
    onboard.pending_limit_s = 8.0
    onboard.pending["h1"] = {"seq": 1, "kind": "hazard", "track_id": 1, "value": 1.0, "bytes": _pose(1)}
    onboard.queued_at[1] = 0.0
    onboard._expire(8.0)
    assert onboard.lost_hazards == 1
    assert "h1" not in onboard.pending
