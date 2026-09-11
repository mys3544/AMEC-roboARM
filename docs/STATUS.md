# Status, end of 2026-09-11 (evening: first tag-free pick attempt)

Written after the session that tried to pick an UNTAGGED cube. Read this bit
first, then the morning's entry below it, which still stands.

## The cube, and what the camera made of it

A plain **red 40 mm cube, no ArUco tag at all**, on the table straight ahead.
The tag-free reading worked: the colour rung put it at **164.2 mm fwd, +21.1 mm
left, 40.2 x 40.2 mm, oriented -55 deg, not clipped**, and that was confirmed
offline against the very frame the pick used (`out/sweep_p00.0.jpg`): the raw
silhouette measures 52.3 x 52.3 mm on the table plane, which at the 211.6 mm
lens height back-solves to a ~40 mm cube. So `range_block` is right on a real
untagged cube. **No tag was involved at any point.**

The neural rung read the same cube as **85 mm ("studio shot", "stop sign")** and
was `clipped=True`, so it was correctly listed and never chosen. Do not read
anything into the 85: the mask ran off the edge of the frame.

## The pick FAILED, and why -- the fingers, not the sums

The gripper shoved the cube from 164 mm out to 185 mm and then "closed on
nothing". **The user watched it happen and named the cause: the fingers were
still half closed as the arm came down, and caught the cube's side.**

That is exactly what the code did. The approach set the opening in the SAME
`move_to` as the arm joints, so J6 was travelling 30 -> 90 (a 70 mm gap closing
to 57 mm) while the arm descended. Two consequences, and the second is the nasty
one:

* a finger that is still closing as it arrives sweeps the object aside instead
  of straddling it; and
* while J6 is in transit the TOOL LENGTH is unknown exactly where it matters --
  `GRIPPER_TOOL_MM` grows 162 -> 180 mm over that travel, so a lagging gripper
  puts the real tips up to 18 mm from where the descent's own forward
  kinematics believe they are. The old comment in `grasp.pick()` admitted this
  ("the tips sit up to 18 mm higher or 10 mm lower ... while the fingers
  settle") and judged it harmless. It is not.

**FIXED in `grasp.pick()`**: the opening is now its own completed move, with
`verify=True`, BEFORE the arm sets off --

    arm.set_gripper(step.opening, speed_dps=GRIPPER_DPS)   # finishes here
    arm.move_to(_arm_only(step.hover), speed_dps=APPROACH_DPS, verify=False)

so the fingers are stationary from the approach until the grasp, and the tool
length is fixed during the descent. This is the "opening FIRST" that was tried
and rejected once for looking odd (a fully open gripper visibly closes before
the arm has gone anywhere). It looks odd; it is correct. 259 tests pass, ruff
clean. **NOT YET TRIED ON THE ROBOT** -- that is tomorrow's first job.

## The other number, deliberately left alone

The same run logged a reach miss that nothing corrected:

    reach pass: fingertip 20.2 mm off (short by -14.9 fwd, -0.5 left, -13.7 up)
    fingertip landed 14.9 mm from the aim point, 26.7 mm up

`grasp.pick()` descends open-loop (`passes=0`) whenever a reach offset is set
([grasp.py], the "ONE descent" comment), on the reasoning that the 20 mm
`REACH_OFFSET_M` already absorbs the model's readback shortfall. At this reach
it does not: the arm fell 14.9 mm short of a 184 mm aim and the descent never
corrected it.

**One variable at a time** -- the gripper fix went in alone, so tomorrow's log
says whether the 14.9 mm was the real problem or just a passenger behind the
fingers. If it is still there, the three candidates, in order of cost:

1. raise the reach offset from the panel (20 -> ~35 mm); no code change;
2. re-enable the correction passes with an offset set -- fixes the 13.7 mm
   HEIGHT error too, which no offset can, but the 2026-09-10 note says passes
   and offset fought each other, so re-tune the offset down if you do this;
3. `tools/touch_probe.py` -- the one check that measures the real fingertip
   against the camera's point instead of trading one model against another.

## Where everything was left

* **Arm HOME and holding** (J1..J5 = 90, J6 = 149), torque ON, table clear.
* **Battery 11.8 V**, down from 12.2 in the morning -- worth charging.
* Red cube loose on the table at about **180 mm fwd, +16 mm left**. At that
  range it reads `"partly out of frame -- its size is not known"` and is
  refused, so **slide it back to 155-165 mm before retrying**.
* Bridge and vision containers left UP on the robot. Local webapp stopped.
* `roboarm/grasp.py` changed on the LAPTOP only -- **not yet pushed to the
  robot** (it does not need to be for `--robot` mode, where the grasp logic runs
  on the laptop, but the robot's copy is now behind).

## Tomorrow, in order

1. Put the cube back at 155-165 mm.
2. `.venv/Scripts/python tools/webapp.py --robot http://192.168.73.210:8761 --port 8091`
3. Pick on the **colour** rung (POST `/api/auto/oneclick` with
   `{"detector": "colour"}`), and watch the fingers.
4. Read the reach numbers out of the log before changing anything else.

---

# Status, end of 2026-09-11 (morning)

Where things stand, so tomorrow starts from here rather than from memory.
Yesterday's handoff is folded in where it still applies.

## What changed today

* **Faster, same accuracy.** The one-click pick of the cube straight ahead went
  from 30.7 s to 22.0 s; with a two-station search it is 26.7 s. The fingertip
  still lands 6-7 mm from the aim point (it did before), holds and places.
  Where the time came from, all measured on the robot:
  * station-to-station base yaws at 30 deg/s instead of 20 (40 costs 0.8 mm);
  * no exposure wait after a yaw: a tag read from the very next frame is within
    1.7 mm of one read 1.2 s later. The 1.2 s wait stays for moves that change
    the picture (coming from home, or after a place);
  * transit moves at 30 deg/s, gripper at 60, place descent at 12; the pick
    descent stays at 8;
  * the gripper opens to the object's width DURING the approach, in one move;
  * the fingers open only enough to release, not fully;
  * the refine look is skipped when the base is already within 8 deg of the
    centred bearing (it moved the estimate 0.7 mm and cost 2 s);
  * the coverage figure is computed once per process and remembered, not at
    the start of every search (it was 2 s of IK over a grid). A warm-up thread
    was tried and dropped: it starved the frame loop.
  The job log now prints how long each stage took ("[search took 7.8 s]").
* **No tag, no declared size needed.** The default detector is now `auto`:
  tags + colour + the neural service, each object sized for itself.
  * A tag's own magnification gives the object's HEIGHT (m = H / (H - h)),
    with no nadir in it; the object is taken to be a cube, so that is its width
    too. The lab's 40 mm cube reads 38-42 mm this way from different stations.
    The one-click button no longer sends "40".
  * An untagged outline (colour blob, neural mask) is ranged from the corner
    farthest from the point under the lens: the two edges meeting there are
    edges of the top face and s * m long. See `detect.range_block()` for why
    two earlier readings were abandoned -- the near-face closed form is exact on
    paper and useless on a real mask.
  * The neural service now runs the PROMPT-FREE YOLOE (`yoloe-11s-seg-pf.pt`).
    Text prompts ("cube", "block", "box", ...) found nothing at all on the
    tag-covered cube; the prompt-free model boxes it well (as a "speed limit
    sign" -- the label is ignored). It is a segmentation model, and its mask
    polygon is sent to the client (`polygon` in the wire format) because a box
    only circumscribes the outline. On the real frame the mask ranged the cube
    at 42 mm, 5 mm from where the tag put it.
  * A panel on the laptop reaches the neural service THROUGH THE BRIDGE
    (`/vision/detect`, `/vision/health`); the compose-internal `vision` host is
    not reachable from outside.
  * Outlines that touch the picture's edge are flagged `clipped` and never
    picked: a 60 mm cube cut to a sliver used to range as a small cube and pass
    the in-frame test.
* **Different sizes.** Each target carries its measured height; the in-frame
  test uses it (a taller cube is thrown further out by parallax). Gripper
  opening follows the measured width, as before. Cubes of 22-55 mm are the
  supported range; the gripper cannot verify a hold below 22 mm.

## Not done: the outer rim (cubes 19-21 cm out)

The sweep measures 129..189 mm from the base; the arm grasps to 210. The fix
is the SECOND calibrated look, tilted out, which the code already carries but
which has never been fitted because it needs the ChArUco board FACE UP:

    (robot)  docker compose --profile bridge stop bridge      # it holds the camera
             docker compose run --rm --no-deps core python tools/calibrate_table.py --outer
             docker compose run --rm --no-deps core python tools/calibrate_table.py --outer --verify
             docker compose --profile bridge up -d bridge

Then the sweep rings both looks (primary ring first, then the outer ring),
backgrounds are filed per look (`table_background_outer_*.png`), and the refine
look tries the primary first. Nothing else needs changing.

## Not verified today

* A PALE cube with no tag has not been on the table. The colour rung cannot
  see one; the neural mask is what should find it. Try it and look at the
  `detect` view -- the label will be nonsense, the box and size are what count.
* The point under the lens (`kin.camera_nadir`) is only known to about 25 mm
  (from yesterday's yaw experiment). The readings above were chosen to depend
  on it weakly; the tag-height reading not at all.

## How to run (unchanged)

* robot: `docker compose --profile bridge up -d bridge` and
  `docker compose --profile vision up -d vision` (the vision container needs
  the CLIP fork installed -- it is in the requirements now; a rebuild picks
  it up, a recreated container does not).
* laptop: `.venv/Scripts/python tools/webapp.py --robot http://192.168.73.210:8761 --port 8091`
* tests: 259, all passing (`PYTHONPATH=. .venv/Scripts/python -m pytest -q`,
  ruff clean).

## Housekeeping

* Arm at the survey pose, holding; cube at about 169 fwd / +45 left.
  Bridge and vision containers left running. Battery 12.2 V this morning.
* Robot tree synced from the laptop by hash. Still no git.
