"""Unit tests for Mode Decoupling and Isolation (ATM & Restricted Zone)."""

from pathlib import Path
import numpy as np
import pytest

from config import load_config
from detection_model import TrackedObject
from main import SurveillancePipeline, prompt_mode_selection


def test_atm_mode_isolation():
    """Verify ATM mode only loads atm_module and only evaluates ATM rules."""
    pipeline = SurveillancePipeline(mode="atm", gui_enabled=False)

    # Check module loading isolation
    assert pipeline.mode == "atm"
    assert pipeline.atm_module is not None
    assert pipeline.restricted_module is None

    # Check video source assignment (strictly webcam "0")
    assert pipeline.source == pipeline.config.video.sources.atm

    # Check window title
    assert "ATM Security Mode" in pipeline.config.video.display_window.title

    # Check zone registration (ATM mode has no zone overlay/registration)
    assert len(pipeline.zone_engine.zones) == 0
    assert "atm_interaction_zone" not in pipeline.zone_engine.zones
    assert "restricted_zone" not in pipeline.zone_engine.zones
    pipeline.atm_module.close()


def test_restricted_mode_isolation():
    """Verify Restricted mode only loads restricted_module and only evaluates intrusion rules."""
    pipeline = SurveillancePipeline(mode="restricted", gui_enabled=False)

    # Check module loading isolation
    assert pipeline.mode == "restricted"
    assert pipeline.restricted_module is not None
    assert pipeline.atm_module is None

    # Check video source assignment
    assert pipeline.source == pipeline.config.video.sources.restricted

    # Check window title
    assert "Restricted Zone Intrusion Mode" in pipeline.config.video.display_window.title

    # Check zone registration
    assert "restricted_zone" in pipeline.zone_engine.zones
    assert "atm_interaction_zone" not in pipeline.zone_engine.zones


def test_source_override():
    """Verify explicit source argument overrides default config source for any mode."""
    custom_src = "sample_videos/custom_test.mp4"
    pipeline = SurveillancePipeline(source=custom_src, mode="atm", gui_enabled=False)
    assert pipeline.source == custom_src
    pipeline.atm_module.close()
