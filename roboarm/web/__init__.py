"""Browser control panel for the arm: live camera, detector views, manual and
automatic modes. `python tools/webapp.py` serves it; see that file for usage.

    session.py   everything the page can see or do, behind one lock and one job
    server.py    the HTTP layer -- JSON endpoints and an MJPEG stream, stdlib only
    sim.py       a stand-in arm and camera, so the page runs with no robot at all
    static/      the page itself, one file
"""
