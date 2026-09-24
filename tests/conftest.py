"""The suite runs against the Sonix camera's profile.

Its fixtures are that camera's real measurements -- the homographies fitted through
it, the lens its ruler placed, the frames the simulator draws at its scale -- so
the lens numbers they are checked against must be its too. The camera switch itself
is tested in test_cameras.py, in fresh interpreters.
"""

import os

os.environ["ROBOARM_CAMERA"] = "sonix"
