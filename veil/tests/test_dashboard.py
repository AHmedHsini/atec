from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "web" / "index.html").read_text(encoding="utf-8")


def test_dashboard_marks_the_camera_as_local_and_shows_age_and_sigma():
    assert "ONBOARD ONLY - NOT TRANSMITTED" in PAGE
    assert "age" in PAGE.lower()
    assert "sigma" in PAGE
    assert "staleness" in PAGE
    assert 'id="theme"' in PAGE or "data-theme" in PAGE
    assert "prefers-color-scheme" in PAGE
    assert "prefers-reduced-motion" in PAGE
