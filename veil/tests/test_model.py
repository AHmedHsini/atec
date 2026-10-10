from pathlib import Path

import numpy as np
import pytest

from veil.dataset import NAMES, features
from veil.model import WEIGHTS, load
from veil.perceive import detect
from veil.sim import render_pose


def test_weights_load_locally_and_match_the_feature_width():
    net = load()
    assert WEIGHTS.exists()
    assert net.samples >= 1000
    assert net.val_accuracy >= 0.85
    sample = features(np.zeros((240, 320), np.uint8), 20, 40)
    assert sample.shape[0] == net.w1.shape[0]
    name, confidence = net.predict(sample)
    assert name in NAMES
    assert 0.0 <= confidence <= 1.0


def test_missing_or_broken_weights_are_refused(tmp_path, monkeypatch):
    import veil.model as model

    missing = tmp_path / "missing.npz"
    monkeypatch.setattr(model, "WEIGHTS", missing)
    monkeypatch.setattr(model, "_cached", None)
    with pytest.raises(FileNotFoundError):
        model.load()
    broken = tmp_path / "broken.npz"
    np.savez(broken, w1=np.zeros((2, 2), np.float32))
    monkeypatch.setattr(model, "WEIGHTS", broken)
    with pytest.raises(ValueError):
        model.load()


def test_empty_and_malformed_frames_produce_no_detections():
    assert detect(np.zeros((0, 0), np.uint8), (0.0, 0.0)) == []
    assert detect(np.zeros((10, 10, 3), np.uint8), (0.0, 0.0)) == []
    image = render_pose((0.0, 0.0), np.random.default_rng(0))
    found = detect(image, (0.0, 0.0))
    assert found
    assert {item["cls"] for item in found} <= set(NAMES)


def test_dashboard_does_not_pretend_the_picture_is_downlinked():
    page = Path(__file__).resolve().parents[1].joinpath("web", "index.html").read_text(encoding="utf-8")
    assert "STAYS ON ROVER" in page
    assert "Not a live feed to Earth" in page
