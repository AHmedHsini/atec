"""Onboard scout: perception, tracking, and landing-site localization. Owner: Ahmed."""

from crater.scout.camera import CameraModel
from crater.scout.detect import detect_hazards
from crater.scout.track import Tracker

__all__ = ["CameraModel", "detect_hazards", "Tracker"]
