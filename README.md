# RoboARM 2026

A browser control panel for the **Rosmaster X3 Plus** arm: live wrist camera,
manual joint and Cartesian control, and a one-click "find the cube and pick it
up" that runs on vision alone -- no tag on the object, no size typed in.

There are three ways to run it, and you will probably use all three:

| | What you need | Who can use it |
|---|---|---|
| **1. Simulator** | this repo + Python | everyone, at once |
| **2. Your laptop drives the robot** | this repo + Python + the robot's WiFi | **one person at a time** |
| **3. The robot's own page** | a browser | everyone, taking turns |

Start with the simulator. It renders the wrist camera's view through the real
calibrated table homography, so a sweep really does find the blocks and a pick
really does lift one. Almost everything you will write can be finished there.

---

## Get the code

    git clone https://github.com/mys3544/AMEC-roboARM.git
    cd AMEC-roboARM

Every command below is run from that folder. If your instructor handed you a
zip instead, unpack it and use that folder -- it is the same tree.

## Install (for 1 and 2)

You need **Python 3.10 or newer**. The project uses [uv](https://docs.astral.sh/uv/)
for everything -- no pip, no conda. Install uv once:

Windows (PowerShell):

    irm https://astral.sh/uv/install.ps1 | iex

macOS / Linux:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then, from the project folder:

    uv sync --frozen

That reads `uv.lock` and builds a `.venv` with exactly the versions everyone
else is using (about 300 MB, mostly OpenCV). It does **not** need the robot.

> If `uv sync --frozen` fails on your platform, run plain `uv sync` -- the lock
> was resolved on the robot and on Windows, so a Mac may need to re-resolve.

---

## 1. The simulator

    uv run python tools/webapp.py --sim --port 8090

Open <http://localhost:8090/>. There is a pretend arm and a pretend camera; no
hardware is touched. Moves take as long as they really would -- pass
`--sim-speed 0` to make them instant while you are debugging something else.

## 2. Driving the real robot from your laptop

**Read this part before you run it.**

The page, the detectors, the kinematics and every decision run on *your*
machine; only servo commands and camera frames cross the WiFi. That is what
makes it good for coursework -- and it is also why:

> **Only one laptop may drive the robot at a time.**
>
> The "one job at a time, everything else gets *busy*" interlock lives in the
> session on *your* machine (`roboarm/web/session.py`). The robot only
> serialises individual commands. Two laptops in `--robot` mode do not see each
> other, and their moves will interleave in the middle of a descent. Take your
> slot, and close the app when your slot ends.

Your instructor starts the bridge on the robot (once, for everyone):

    docker compose --profile bridge up -d bridge

Then, on your machine, with the robot's address:

    uv run python tools/webapp.py --robot http://192.168.73.210:8761 --port 8091

Open <http://localhost:8091/>. If the robot is on a different address, change
the URL -- nothing else needs editing.

Check it is alive before you move anything: the **arm** and **camera** pills at
the top of the page should be green, and the log should show the calibration
loading. `http://192.168.73.210:8761/health` in a browser tells you the same
thing without starting the app at all.

## 3. The robot's own page

Nothing to install: open <http://192.168.73.210:8080/> (when your instructor has
the `web` service up). Everyone's browser shares **one** session here, so the
interlock works properly -- one person drives, everyone else gets "busy" and can
watch. This is the safe way to demo to a group.

---

## Using the panel

Along the top are status pills: arm, battery, camera, calibration, vision
service, and the current job. Under them, the two big buttons -- **PICK & PLACE
THE CUBE**, which does the whole thing by itself, and **STOP / HOLD**, which is
always allowed and abandons whatever move is under way at its last legal step.

**Manual** mode gives you per-joint sliders, jog buttons, Cartesian nudges
(forward / left / up in millimetres), an absolute "go to x, y, z", the gripper,
and the `home` / `survey pose` presets. Manual moves are refused while an
automatic job is running -- that is deliberate.

**Automatic** mode runs the jobs: `sweep & detect` (yaw the base through a few
stations and range everything it sees), `pick & place`, `go to survey pose`, and
`capture background` (a photo of the empty table, which the colour detector
subtracts -- retake it if the lighting changes).

The **view** selector switches the live image between `raw`, `detect`, `mask`
and `edges`; the **detector** selector picks which rung of the ladder is used.
The name in the menu and the name the API takes are not always the same -- the
API value is the one in backticks:

* `auto` -- tags, colour and the neural service together, best first. Needs no
  tag, no background photo and no declared size. Use this unless you are
  studying one rung in particular.
* `markers` (the menu says *tags*) -- an ArUco marker. The most accurate. Given
  an empty-table photo it fuses the tag's position with the silhouette's
  measured size; without one the size comes from the tag's own magnification,
  and anything wider than its tag is refused.
* `colour` -- a coloured blob, read on the table plane. Wants a saturated
  object.
* `changes` -- whatever is on the table that was not in the empty-table photo.
  Run `capture background` first, or this one fails outright.
* `yolo` (the menu says *neural*) -- a prompt-free YOLOE-26 segmentation model
  (YOLOE built on YOLO26, `yoloe-26s-seg-pf`) in its own container on the
  robot. Its *label* is nonsense ("stop sign"); the outline is what counts. An
  outline that is the whole picture, or that lies inside another outline (the
  printed face of a cube), is not an object and is dropped. If the service is
  not answering it drops quietly to `markers` and says so in the log. Any
  Ultralytics segmentation model can be served instead via
  `ROBOARM_VISION_MODEL` in `compose.yaml`; the plain COCO `yolo26n-seg` only
  finds what resembles one of its 80 classes.

Leave **object mm** empty: every rung measures the object for itself. A size
typed there overrides the measured one on every rung -- the position always
stays the camera's.

Objects whose outline touches the edge of the frame are flagged `clipped` and
never picked: a cube cut in half by the frame edge ranges as a small cube far
away, and the arm would dive for a place that is not there.

## Talking to it from your own code

The page is a thin client over a small JSON API, and so can your script be:

    import json, urllib.request
    req = urllib.request.Request(
        "http://localhost:8090/api/auto/oneclick",
        data=json.dumps({"detector": "colour"}).encode(),
        headers={"Content-Type": "application/json"})
    print(json.load(urllib.request.urlopen(req)))

The full list is in the docstring at the top of `roboarm/web/server.py`. The
most useful ones:

    GET  /api/state              everything the page shows, as JSON
    GET  /api/log?since=N        log lines newer than N
    POST /api/arm/cartesian      {"dx": 5, "dy": 0, "dz": 0} or {"x": 160, "y": 0, "z": 40}
    POST /api/arm/hold           the stop button; always allowed
    POST /api/auto/start         {"job": "sweep" | "pick" | "place" | "background" | "survey"}
    POST /api/auto/oneclick      {"detector": "auto"}: find a cube, pick it, put it down

A request that cannot run right now comes back `409` with a reason, not an
exception -- check for it.

## Tests

    uv run python -m pytest -q

259 of them, and they all pass without a robot (the simulator is what most of
them drive). Run them before you ask why something is broken. Style:

    uv run ruff check roboarm/ tools/ vision_service/ tests/

---

## What is in here

    roboarm/          the library: arm control, kinematics, detection, grasping
      arm.py          every servo command, and every safety rule, goes through this
      kinematics.py   forward and inverse kinematics for the 5-DOF arm
      detect.py       the detector ladder: tags, colour, neural; and ranging
      grasp.py        approach, descend, close, lift -- the pick itself
      sweep.py        yaw the base, look, collect what is on the table
      web/            the panel: server.py (HTTP), session.py (all the judgement),
                      remote.py (drive a robot over the bridge), sim.py (no robot)
    tools/            runnable scripts; webapp.py is the panel, the rest are probes
                      and calibration routines that need the real hardware
    tests/            pytest suite
    vision_service/   the neural detector; runs in its own container on the robot
    docs/STATUS.md    the engineering log -- what worked, what failed, and why.
                      Written for whoever picks the project up next; read it when
                      you want to know why something is the way it is.

## When it goes wrong

**`ImportError` or a DLL error from `cv2`** -- you are running the system Python
instead of the project's. Always `uv run python ...`, never bare `python`.

**The page says the arm is not connected** -- the robot's factory app grabs the
serial port and the camera when it is running. Your instructor has to stop it;
you cannot fix this from your laptop.

**Everything answers "busy"** -- an automatic job is holding the arm. Wait for
it, or press STOP. If you are in `--robot` mode, it may be *someone else's*
laptop moving the arm; see the warning in section 2.

**A pick shoves the cube instead of grasping it** -- it is probably not sized
the way you think. Look at the `detect` view and the log line that gives the
range and the measured width before blaming the arm.

**Nothing at all responds and the arm is limp** -- check the battery pill.
Below 10.0 V (`MIN_BATTERY_V` in roboarm/config.py) the arm refuses to engage
at all and says so in the error; a pack sagging towards that misses commands
before it gets there. Tell your instructor rather than retrying.

## House rules

* One driver at a time on the real robot.
* Keep a hand near **STOP / HOLD**, and keep your other hand out of the
  workspace. The arm does not know you are there.
* Do not expose the robot's bridge port beyond the lab WiFi: it has no
  authentication by design, and anyone who can reach it can move the arm.
* Put the cube back, and leave the arm at `home` when you are done.
