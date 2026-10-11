from veil.panorama import crop_view, load_strip
from pathlib import Path


def test_turning_the_strip_changes_the_view():
    image = load_strip(Path(__file__).resolve().parents[2] / "dataset" / "panorama" / "apollo11.jpg")
    left, _ = crop_view(image, 0)
    right, spec = crop_view(image, 90)
    assert left.shape == right.shape
    assert left.shape[1] < image.shape[1]
    assert spec.pov == "surface"
    assert abs(float(left.mean()) - float(right.mean())) > 1.0
    wrapped, _ = crop_view(image, 350)
    assert wrapped.shape == left.shape
