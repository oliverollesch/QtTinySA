"""Firmware gating for scanraw option 3 (auto-repeat)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.devices import scanraw_auto_repeat_supported  # noqa: E402


def test_pre_177_does_not_auto_repeat():
    assert scanraw_auto_repeat_supported("v1.4-156-g4eb315d") is False
    assert scanraw_auto_repeat_supported("v1.4-175") is False


def test_177_and_later_auto_repeat():
    assert scanraw_auto_repeat_supported("v1.4-177-gabc") is True
    assert scanraw_auto_repeat_supported("v1.4-199-gde12ba2") is True


def test_missing_firmware_is_conservative():
    assert scanraw_auto_repeat_supported(None) is False
    assert scanraw_auto_repeat_supported("") is False
