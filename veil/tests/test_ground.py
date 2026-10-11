from veil.ground import GroundMap


def _hazard(seq, t_ms, x):
    return {"type": 1, "seq": seq, "t_ms": t_ms, "track_id": 4, "cls": "pit", "x": x, "y": 5.0, "sigma": 0.1, "radius": 0.4, "agreement": 80, "flags": 0}


def _pose(seq, t_ms, y):
    return {"type": 2, "seq": seq, "t_ms": t_ms, "track_id": 0, "cls": "pose", "x": 0.0, "y": y, "sigma": 0.05, "radius": 0.0, "agreement": 100, "flags": 0}


def test_duplicate_and_older_packets_do_not_roll_back_the_map():
    ground = GroundMap()
    assert len(ground.ingest([_hazard(1, 1000, 0.3)], 1.0)) == 1
    assert ground.ingest([_hazard(1, 1000, 9.0)], 1.2) == []
    assert ground.hazards[4]["x"] == 0.3
    ground.ingest([_hazard(2, 500, 8.0)], 2.0)
    assert ground.hazards[4]["x"] == 0.3
    ground.ingest([_hazard(3, 2000, 0.35)], 2.5)
    assert ground.hazards[4]["x"] == 0.35
    ground.ingest([_hazard(3, 2500, 0.36)], 3.0)
    assert ground.hazards[4]["x"] == 0.36
    assert len(ground.hazards) == 1


def test_sequence_wraps_across_65535_and_duplicates_collapse():
    ground = GroundMap()
    assert len(ground.ingest([_hazard(65535, 1000, 0.30)], 1.0)) == 1
    assert ground.ingest([_hazard(65535, 1000, 9.0)], 1.1) == []
    assert ground.hazards[4]["x"] == 0.30
    ground.ingest([_hazard(0, 2000, 0.40)], 2.0)
    assert ground.hazards[4]["x"] == 0.40
    ground.ingest([_hazard(65534, 9000, 8.0)], 3.0)
    assert ground.hazards[4]["x"] == 0.40
    ground.ingest([_pose(65535, 1000, 1.0), _pose(0, 2000, 2.0), _pose(65534, 3000, 9.0)], 3.0)
    assert [pose["y"] for pose in ground.poses] == [1.0, 2.0]


def test_dead_reckoning_stops_and_uncertainty_grows_only_while_silent():
    ground = GroundMap()
    ground.ingest([_pose(1, 0, 0.0), _pose(2, 500, 1.0)], 0.5)
    coasting = ground.scout(1.0, link_open=False)
    assert coasting["predicted"] is True
    assert coasting["y"] > 1.0
    held = ground.scout(3.0, link_open=False)
    assert held["predicted"] is True
    assert held["y"] == 1.0
    ground.ingest([_hazard(3, 0, 0.2)], 0.0)
    silent = ground.view(2.0, link_open=False)[0]
    assert silent["sigma"] == 0.1
    assert silent["staleness_m"] == 0.1
    fresh = ground.view(2.0, link_open=True)[0]
    assert fresh["sigma"] == 0.1
    assert fresh["staleness_m"] == 0.1
    assert fresh["age_s"] == 2.0


def test_unknown_messages_do_not_move_the_map():
    ground = GroundMap()
    ground.ingest([{"type": 9, "seq": 1, "t_ms": 0, "cls": "pit", "track_id": 1, "x": 1, "y": 1, "sigma": 0.1, "radius": 0.1, "agreement": 1}], 0)
    assert ground.hazards == {}
    assert ground.poses == []
