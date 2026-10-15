"""The suite runs against the Sonix camera's profile.

Its fixtures are that camera's real measurements -- the homographies fitted through
it, the lens its ruler placed, the frames the simulator draws at its scale -- so
the lens numbers they are checked against must be its too. The camera switch itself
is tested in test_cameras.py, in fresh interpreters.
"""

import os
import time

import pytest

os.environ["ROBOARM_CAMERA"] = "sonix"


@pytest.fixture(autouse=True)
def arm_as_modelled(monkeypatch):
    """The simulator's arm goes exactly where the model says: the grips taught by hand on the real
    one (roboarm/x3aim.py) off, as the M3's suite has its own (m3pro/tests/test_m3pick.py). The
    tests of the taught aim turn it on."""
    from roboarm import x3aim
    monkeypatch.setattr(x3aim, "ON", False)


def _wait(condition, timeout: float = 5.0):
    """Poll `condition` until it holds; the web and bridge tests drive threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out waiting")
