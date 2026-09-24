# Status, 2026-09-24 (the wrist camera is now a Logitech C930e)

The user replaced the stock Sonix wrist camera with a Logitech Webcam C930e.
Lab WiFi (robot .210). Branch `drop-pose`, uncommitted at the end, on top of the
09-22/23 work that was already uncommitted.

## The old camera is kept, one file away

`ROBOARM_CAMERA` (`sonix` | `c930e`, default `c930e`) keys everything
camera-specific: picture size and V4L2 settings, lens distortion, lens position,
reach offset, the depth engine, and `data/cameras/<name>/` for the calibration
and empty-table photos. The Sonix's calibration was copied to
`data/cameras/sonix/` on the robot BEFORE anything was refitted, and into git as
`cameras/sonix/table_homography.json`. Going back: `cp cameras/sonix.env .env`,
recreate bridge and vision, `ROBOARM_CAMERA=sonix` on the laptop
(`cameras/README.md`). The test suite is pinned to the Sonix (`tests/conftest.py`,
its fixtures are that camera's data); `tests/test_cameras.py` checks the switch.

## The C930e

* Mounted on arm_link4 (does not roll with J5), looking along the tool; the
  user's ruler: 85 mm from J4, 50 mm off the axis on top of the arm. Fitted from
  23 board views the optical centre is 39 mm off (50 is presumably the body),
  3 mm sideways -- `_LENS` uses 85 / 39 / 0.
* Its 4:3 sizes are crops. Measured at the survey pose: 640x480 sees 228 x
  172 mm, every 16:9 size 305 x 171, the whole 3:2 sensor 303 x 204 but only at
  1 fps uncompressed. So 848x480 MJPG (widest usable view, same px/mm as the
  640x480 the pipeline was built at). Continuous autofocus hunts on a plain
  table, so focus is fixed at 45 (sharpest from the survey pose), set on every
  open because the camera keeps it only until unplugged.
* Lens distortion undone on capture: a one-view homography was 1.4 mm mean /
  3.1 mm worst off, 0.8 / 0.95 undistorted. Focal length is not pinned by views
  that all look down (416 or 501 by model) -- do not read heights off it.
* Depth rung: new ONNX exported at 518x910 (throwaway `uv run --with torch`
  env on the laptop, weights from its HF cache), engine built with trtexec in the
  vision container, 54 MB. `depth.py` reads its input size from the engine and
  picks the engine by `ROBOARM_CAMERA` (compose passes it to vision).

## Three things found on the way, all fixed

1. **The board had moved 11 mm** to the robot's right since the Sonix was
   calibrated (paper off and on, or the robot nudged). Found by fingertip probes
   on five board corners and by the circle the lens draws round J1; after moving
   `BOARD_Y0` -185.5 -> -196.5 mm a +-60 deg yaw arc puts the J1 axis at
   (-0.3, +0.6) mm, i.e. on the robot's origin. Forward was right.
2. **Look poses were not repeatable.** The heavier camera leaves J2..J4 up to
   two degrees short of a command, on whichever side they came from (J3 read 22
   from below, 24 from above, same command), and the saved pose was the
   READBACK, so re-commanding it put the arm lower still: the homography was
   7..22 mm out on revisits. Now `calibrate_table.py` saves the COMMAND, and
   `Arm.move_to(repeatable=True)` -- used for every look pose in the session and
   the calibrate tool -- goes 4 deg past the goal on J2..J4 and comes down onto
   it. Nudging by the shortfall (tried first) fights the deadband and does not
   converge. Revisits from home, from a yawed station and from below: 0.5..1.2 mm
   mean, 1.8 mm worst (one case 3.4).
3. **Pooling yaws in a fit adds J1's error**: 11 mm worst pooled, ~1 mm from
   one yaw. The C930e sees 5..9 markers per frame, so every look is fitted at
   yaw 0 only now. Fits: primary 0.9 mm, outer 0.7, near 0.8 (worst corner).

## Live, with the paper back and five cubes

Sweep: 21 stations in 46 s, the five cubes listed and nothing else plannable.
One-click pick held (26 mm cube at 131/-33, re-look 2.4 mm from the sweep).
CLEAR THE TABLE: three more held (landed 2.2 / 1.6 / 8.5 mm from the aim), the
fourth -- 26 mm cube at 188/+69, 197 mm out -- closed on nothing twice with the
5 mm reach offset (6 mm, capped). Hovering the tips over the camera's estimate
showed the camera right to a few mm; the fingertip probes had shown the tips now
landing 4..8 mm FURTHER than their readback. Offset 0: held first time, passes
14.3 -> 4.2 -> 2.2 mm. So `REACH_OFFSET_M` is per camera: Sonix 5, C930e 0 --
one cube's worth of evidence; watch the grips and tune on the panel.

## Later: the 30 x 30 x 60 block (two 30 mm cubes glued), lying flat

CLEAR THE TABLE took it at the tenth attempt, pushing it 20..30 mm each time.
Two faults, both in detection:

1. **Tops were read as squares.** `range_block` took the short side of the
   depth rung's top face, solved the cube relation and returned a square, so
   the grasp (which closes across the narrow way, and treats a square as
   either way round) closed along the 60 mm half the time -- 5 mm to spare
   either side of the 70 mm opening. Now the depth rung's outlines are read
   as RECTANGLES (`detect._range_top_face`): the short side sets the height
   (a bar lying down is as tall as it is narrow), both sides shrink by the
   same magnification. Box-only detectors keep the square reading.
2. **The depth rung cut the block's top to a sliver.** Its edge threshold is
   8 MADs of the map's gradient; on a clear table that is 0.013, under the
   slope the model draws across a top face (~0.03), so the whole top was
   "edge" but for a 57 px strip -- missed from one look, 11 mm wide from the
   next. A plain floor at 0.05 fixed that frame and lost every cube in a busy
   one (robot parts in view stretch the normalisation; rims fell under 0.05).
   So `raised_regions` now runs a second pass at the floor, and a coarse
   region replaces the fine regions inside it only when there is at most one
   -- it mends slivers and finds tops the fine pass missed, never merges.
   Over 63 of the day's frames: nothing lost, 11 frames gained whole objects
   the old pass missed or cut (checked by eye), the block whole in every view.

Live after both: the block measured 26 mm wide, the fingers turned across it
(+78 deg, J5 105), held first time; a cube held too. The other two attempts
went into the DROP TRAY -- dropped cubes, and the blue block lying half out of
it -- at 92 and 108 mm from the drop point, just outside `DROP_ZONE_M` (80 mm).

## Open

* The drop tray's contents are targets beyond 80 mm of the drop point; widen
  `session.DROP_ZONE_M` (to ~120 mm) or end the ring short of the tray -- the
  user's call, it takes a piece of the right-hand workspace with it.
* The reach passes now swing +-8 mm (short, over, short) where they used to
  converge -- the same deadband. A repeatable approach inside `_reach_to` would
  probably fix it; not done, grasp moves are untouched.
* The CH340 dropped off USB once mid-session (bridge's serial thread died,
  every servo read empty, stale battery). Cable nudged while the table was set,
  or the hub: the C930e draws more than the Sonix on the same USB 2 hub and its
  own port logged a reset earlier. Restart the bridge if it recurs; watch it.

---

# Status, 2026-09-22 ("clear the table" picks as it goes; the tagged cube nobody could see)

Hotspot again (robot 172.20.78.81, laptop .150); the robot had rebooted, bridge
and vision restarted over paramiko. Branch `drop-pose`, uncommitted at the end.

## Morning: the round-by-round CLEAR THE TABLE, 330 s, "failed" on an empty table

Four cubes. Round 1 listed 4 raised, held 3; the white cube missed after a
side-station re-look (J1 71, refined 10 mm, passes 17.6 / 7.3 / 7.9, landed
6.6 mm off and 31 mm up, closed on nothing -- the 2026-09-21 signature). Round
2 re-found it 19 mm away and held it. Round 3 found a tagged cube at 163/+23
that rounds 1 and 2 had walked past, held it -- and the job then said FAILED
at the 3-round cap without a confirming sweep, table empty. Each full sweep
cost 56..65 s, most of it looking at nothing.

## Afternoon: pick as you go (user's design)

`_job_pickall` now walks the ring and at a station that shows something
plannable picks the surest, drops, comes BACK TO THAT STATION (2..4 s, was 3.4
to survey) and looks again; moves on when the station is empty; a miss gets one
retry from the same station (the failed grasp moves the cube, the new look says
where); a round that picks nothing ends the job. No hints in the ring walk:
every bearing gets both looks anyway, and the first live run chained six to
eight hint looks off the neural rung's edge junk ("server room", "clip art")
between two regular stations, all empty. Live, with cubes being fed in during
the run: 6 listed, 6 held (residuals 1.0..6.0 mm), done in 336 s over two
rounds; an empty round is ~65 s. `PICKALL_ROUNDS` still caps at 3 with the
message "is someone adding to the table?". 285 tests, ruff clean.

## Why the small tagged cube was missed for two rounds

Round 3's J1=90 primary frame shows it whole, bottom right. Three rungs, three
reasons: the depth model drew it (bright plateau) but its region ran into the
frame border and `raised_regions` dropped it as cut off; the neural rung saw it
clipped; the tag rung decoded the tag but read it as 21.8 mm through the
homography -- SMALLER than the 26 mm `OBJECT_TAG_M` -- so height None, and
the unlift about a wrong size put it at 124/0 instead of ~163/+23. Two things
behind that: the tag printed on the small cube is not 26 mm (about 23 by the
picture; measure it, and consider a size per tag id), and the primary look's
homography (8 points, worst residual 0.5 mm, near-affine) under-reads near
the bottom of its frame. Gate change anyway: a tag that measured ITSELF raised
(`detect.TAG_RAISED_M`, 10 mm) is kept even when depth has nothing under it.

## Later: one motion per look ("stable movement"), and three panes side by side

The user: the station-by-station move, stop, settle, look is jerky; keep the
arm moving. `Arm.glide()` starts an interpolated move on a thread of its own
and returns at once; `read()` answers meanwhile (every SDK call now goes
through one serial lock, `Arm._io`), `moving()` says whether it is still
going, `hold()` or any `move_to()` stops it where it is. The bridge allows
`glide` and `moving`; `SimArm` has the same, stepping one degree per 10 ms
with time_scale 0, and `SimStream` renders a zero-settle frame on demand so
the simulated frame is never staler than the robot's (60 ms). In the session,
`_ride()` drives to a pass's first station, glides to its last at `SCAN_DPS`
(12 deg/s) and, as each station's yaw goes by, takes the next frame, reads
the yaw before and after it and hands the frame to the detector with a look
at the mean yaw. The full sweep and "clear the table" ride; every other pass
runs backwards so the outer look starts where the primary ended; the one-
click SEARCH still steps station to station (nearest first, hints). A target
seen from the moving camera always gets a standstill re-look before the pick
(`force_refine`). Live: the full sweep took 43.5 s for 16 stations (was
56..65), primary pass 16 s, outer pass 16 s, read yaws within 2 deg of the
nominal stations, four objects listed. Scan accuracy: about a degree of yaw
(3 mm at 160 mm; test tolerance widened to 5 mm), far inside the 20 mm merge.
`SCAN_DPS` is the knob: faster smears the rolling exposure, untested above 12.

CLEAR THE TABLE with the ride, live: three reachable cubes, three held
(landed 4.0 / 4.1 / 1.3 mm from the aim), done in 169.7 s -- round 1 with
the picks 128 s, the empty confirming round 38 s (was ~65). The standstill
re-look moved the ride's estimates by 1.9..7.9 mm. A tagged cube at 215/+92
was listed as unreachable and left, correctly.

The page: camera, "how it sees" and the controls are three columns now
(`.videos { display: contents }`); under 1400 px the two pictures stack again,
under 1000 px everything does. 287 tests, ruff clean.

## J3 floor 10 -> 0 (user: more range)

The 10 was a margin from the servo end like J2's old 15; the SDK discards
anything below 0, so 0 is the end of it. Nearest grasp point 132 -> 122 mm,
grasp grid +6 %, reach 238 mm and ring coverage (46 %) unchanged. The new
122..129 mm band is grasped but not SEEN: the primary look's near edge is
129 mm, so the ring's blind spot is now two rims (test re-pinned). Bridge
recreated with it. The user then drove J3 to 0 by hand (J2 107 at the time).
J4's floor went 10 -> 0 next, same reason: no change to the grasp envelope
at all (J4 only aims the tool). Bridge recreated again.

## The near look (third calibrated look), and the tagged cube

The user: the small tagged cube is not detected, nor a small cube 11 cm
straight ahead. The sweep log says otherwise for the tag: listed every time as
tag 3 at 213/+91, 232 mm out, REFUSED because its grasp point (239 mm) is past
the 238 mm reach -- it needs to come 3 cm nearer (and its tag is probably not
26 mm; the user is asked to measure it). The white cube straight ahead was the
real gap: at ~125 mm it is graspable since the J3 floor went to 0, but the
primary look's picture starts at 129 mm, so every station saw it cut off at
the near edge and nothing listed it.

Fix: a NEAR look. Searched fingertip targets 20..150 mm out, 40..230 up, tool
150..180, inside the limits and clear of the mast, ranked by how near the
picture's near edge lands with the lens 150 mm up or more; only possible
now that J3 goes to 0. Chosen: 110 out / 110 up, tool straight down -> J2 74,
J3 11, J4 0, lens 236 mm up, nadir 110, 163 mm from the mast. Live frame from
there: the white cube whole, the base plate's edge at the bottom. Fitted with
the board (bridge stopped): `calibrate_table.py --look near
--yaws=-20,-10,0,10,20`, 40 corners, worst residual 6.5 mm, board seen
59..170 mm forward (predicted 56..166). Verify drove the fingertip to the
corner at 157/+31 (model 151/+29); the near corners the verify would have
picked first are inside the 122 mm nearest grasp point, so it now skips to
the nearest reachable one. `--look {primary,outer,near}` replaces `--outer`
(kept as an alias). `sweep.hints` now sends a near-edge clip to the nearer
look (`nearest_of`, the mirror of `reach_of`); 24 stations per sweep; the
ride does three passes. 288 tests. Bands from the fitted matrices: primary
110..218, outer 133..266, near 59..179 mm.

## With the near look: the sweep, the back-tilted pitches, and depth over tags

Sweep with three looks (paper and cubes back): 62 s, the white cube listed
for the first time, at 107/-7 by the near look -- and refused: the nearest
grasp point was 122 mm. What stops a nearer grasp is not J3 any more but the
pitch list: straight down needs J3 negative, 140..175 needs J2 negative,
while pitch 190 (the tool leaning 10 deg BACK toward the base) reaches
107 mm at J2 23 / J3 19 / J4 34 with 222 mm of mast clearance. So
`kin.GRASP_PITCHES` gained 185 and 190, tried last: nearest grasp point
90 mm, +16 % grasp points; not 195 or beyond (74 mm, then 23 mm) because
nothing models the base plate, whose edge is ~60 mm out. Pins re-set (one
look 7 %, primary ring 40 %, annulus 90..238).

CLEAR THE TABLE with all of it: 5 held, done in 289 s -- the white cube at
107 mm picked from the near look in one reach pass, 2.0 mm off. The one miss
was the tagged cube: the tag rung placed it at 181/-60, closed on nothing,
and the next look at the same station listed depth's outline at 156/-52 --
25 mm nearer -- which picked. The tag on that cube is not the 26 mm the rung
assumes (still to be measured), and a mis-sized tag mis-lifts. So in auto,
where depth drew a top face within 40 mm of a tag, DEPTH now places and
sizes the object and the tag only names it (`detect.everything`, test added).

## Evening: four wishes and the rim test

* SWEEP_SPAN_DEG = 160: stations J1 15..165 (7 per look, 21 in all); the
  J1 180 station only ever saw the pile. J1 still reaches 180 for the drop.
* With the cube held the trip to the drop pose runs at 60 deg/s
  (grasp.TRANSIT_DPS), and so does the empty-handed return (RETURN_DPS).
* Passes go nearest look first: near, primary, outer, alternating direction.
* A round that picks nothing ends at HOME, not survey.

The rim test (user placed rotated cubes at 214 mm, right and left of
centre). The sweep listed both from the outer look only, each 7..10 deg off
its station's centre. Picked by index with a re-look: the right one's
centred reading was 11 mm from the sweep's (10 deg off centre), the left
one's 3 mm (7 deg); both held. So the outer look's off-centre reading is
biased sideways-outward, and the centred re-look corrects it. The reach
passes cannot see any of this: they compare the joint readback with the
aim, not with the cube. Where the miss the user described can still happen
is the one-click search, which used to skip the re-look within 8 deg of
centred -- at the rim that is ~9 mm. REFINE_IF_OFF_DEG 8 -> 3.

## Late: one round, 140 deg, 16 deg/s, no stops around the drop

User, after stopping a run in its confirming round: one round only; the scan
a little faster; and the lift, the trip to the drop and the return as one
motion. So: PICKALL_ROUNDS 1 (the job ends "the ring is done: N picked up;
going home"); SWEEP_SPAN_DEG 140 (J1 20..160, 7 stations per look, 21);
SCAN_DPS 16; `Arm.move_to(settle=False)` skips the quarter-second pause and
the readback, used by the lift (now at TRANSIT_DPS 60), the trip to the drop
pose and the way back; drop() waits 0.1 s before opening and 0.15 s after,
and no longer reads the gripper for its log line. Bridge recreated for the
arm change.

Live CLEAR THE TABLE with all of it: two rim cubes, both held (3.9 and
2.6 mm from the aim), whole job 96.7 s -- three passes in about 33 s, a
pick-drop-return cycle about 20 s, home at the end.

After the user's robot reboot (bridge and vision restarted, panel reconnected
with /api/arm/connect): five cubes, five held first try, 172 s, home at the
end. Pick-drop-return cycles 17..20 s; the drop 3.5 s, the return 1.1..1.7 s.
One re-look found nothing within 30 mm and used the sweep's estimate; it
still held (6.5 mm from the aim).

## Grip on the left

The user saw one left-side cube taken off-centre with the arm turned. Every
left-side pick today (+77..+88 mm) went through a side re-look at J1 68..74
with the wrist rolled; all held. Next: a pick test on the left with refine off
to separate the re-look's estimate from the wrist roll.

Where things were left: arm at survey, torque on, 11.6 V; six cubes in the
pile at the drop spot; panel on :8091 in manual.

---

# Status, 2026-09-21 evening (the depth rung: Depth Anything V2 Small as a gate)

Branch `drop-pose`, commits after the afternoon's: the "clear the table" job
(`pickall`), the drop-zone exclusion, the depth rung, the depth gate, and the
two diagnostic views. All on the robot, panel restarted, 285 tests green.

## Why depth

The day's misses were the colour rung's: the white picture cube on the white
paper read 15 / 17 / 29 mm (its picture, or its shadow, not the cube), the
tagged cube was placed 19 mm off, and "clear the table" tried three shadows.
An offline test on the saved sweep frames (laptop CPU, `depth-anything/
Depth-Anything-V2-Small-hf`) showed every cube's top face as a crisp plateau
with NO shadow -- white-on-white included -- and nothing on empty-table frames.
What the model cannot do is metric depth: its relative map warps slowly across
a flat table, a global scale+shift fit is ill-conditioned (70..120 mm for
20..40 mm cubes), and background subtraction fails on window size. EDGES work:
Sobel of the map, threshold at median + 8 MAD, regions enclosed by edges that
do not touch the border and are nearer than a plane fitted to a 31 px ring
around them. Blue 40 mm cube: 38..41 mm on four frames.

## What was built

* `vision_service/depth.py`: the engine (`models/da2s_518x686.fp16.engine`,
  53 MB, built with trtexec --fp16 from the laptop's ONNX export at the wrist
  camera's 4:3, 371 s; 26 ms per frame; output matches the CPU model, corr
  0.99999) behind `POST /raised`, same wire format as /detect, label "raised",
  confidence = step / 40 capped at 1. `?map=1` adds the map as a half-size
  8-bit PNG for the panel. No engine -> 503, the client skips the rung.
  Health shows `depth`. The fp16 map is quantised to ~1500 values; the
  quantisation steps were gradients everywhere and lifted the threshold past
  a real rim on two of three frames -- a 3x3 Gaussian blur first fixed it.
* Client: `detect.raised()` ranges the outlines as TOP faces; mode "depth" on
  the panel; the bridge forwards `/vision/raised` (query included).
* **The gate** (`detect.everything`): once depth has answered, a colour or
  neural silhouette it did not confirm is dropped, except a CLIPPED one (never
  graspable, but it feeds `sweep.hints`); a tag with nothing raised under it is
  lying flat. Without a depth answer the other rungs fill in as before.
* "clear the table": `Session._job_pickall` -- full sweep, pick and drop every
  plannable target surest first, sweep again, stop when a full sweep finds
  nothing, give up after 3 rounds. Header button and card button. Anything
  within 80 mm of the drop pose's fingertips is "at the drop-off spot" and
  never a target (the pile is in view from the ring's end stations).
* Views "depth" (the map, viridis, with the raised outlines) and "mixed"
  (every rung ungated, each in its colour: blue colour, magenta tags, yellow
  neural, green depth, with a per-rung count or "off" in the corner). Made
  only while a client is watching them (`Session._wanted`), because each costs
  a vision round trip -- 180..250 ms over the hotspot.

## Live results

| run | listed | attempted | held | ghosts |
|---|---|---|---|---|
| clear the table, auto before the gate | 4 raised + 2 colour | 6 | 4 (all raised) | 3 colour, all shadows |
| clear the table, auto with the gate, cubes added mid-job | raised only | 5 | 4 | 0 |
| CLEAR THE TABLE from the header button, three cubes | raised only | 3 | 3 | 0 |

The last run is the clean case, 15:54: pressed in manual mode, one full sweep
(60 s) listed the three cubes at 169 / 154 / 121 mm, all 27..28 mm, and the
pile as "at the drop-off spot"; three picks landed 5.6 / 0.6 / 3.2 mm from
the aim and all held; the second sweep saw only the pile and the job ended
with "a full sweep found nothing left" after 200.8 s -- the clean exit, not
the round cap. The first cube's re-look found nothing within 30 mm and used
the sweep's estimate, which was good enough to hold.

The one miss with the gate: a cube at 165 / -62 re-looked from a side station,
moved 11 mm by the re-look, tips landed on the aim, closed on nothing. Both of
the tagged cube's earlier misses also followed a re-look, and the tag rung had
it 19 mm left of where depth, neural and colour all put it: the re-look from
the side and the tag unlift are the two things to look at next.

Open: the white "20 mm" cube and the tagged black cube both measure ~28 mm from
the depth outline; measure them. `tools/export_engine.py` does not build the
depth engine yet (it was built by hand with trtexec; the ONNX is in models/).

---

# Status, 2026-09-21 afternoon (fixed drop pose 90 deg right; J1 to 180; offset capped)

Seven picks on the robot, five of them one-click, on the user's phone hotspot
(lab WiFi trouble): robot at 172.20.78.81, laptop on the same network, no
WiFi hop needed. Bridge and vision were down after the reboot and were
started over paramiko (`docker compose --profile bridge up -d bridge`,
`--profile vision up -d vision`); `tools/sync_robot.py` still hard-codes
192.168.73.210, `.claude/launch.json` points at the hotspot address and is
NOT committed. Branch `drop-pose` is pushed with three commits (desktop work,
drop pose + J1, offset cap); the PR is still to be opened by hand, `gh` is not
installed here and there is no token (compare link:
github.com/mys3544/AMEC-roboARM/compare/main...drop-pose).

## What changed

* **A pick ends at a FIXED DROP POSE** (user: "hardcoded pose, that's it"):
  `cfg.DROP_POSE = {1: 180, 2: 44, 3: 38, 4: 19, 5: 90}`, base a full 90 deg
  to the right, tips 189 mm out and ~110 mm above the table, chosen by
  driving the arm there and looking (first at 139 mm, then 50 mm further
  out, then 50 mm higher). `grasp.drop()` moves there and opens fully; no
  descent, no put-down. The one-click and the pick job take no drop_x/drop_y
  any more; the page's "place at x/y" button still uses `grasp.place()`.
* **J1's ceiling 170 -> 180.** The 10 deg was only a servo-end margin, like
  the J2 floor that went 15 -> 5; the servo takes 180 and reads it back. With
  it the arm reaches a wedge on the right the ring never looked at (last
  station J1 165), so `sweep.ring()` adds a station at an end of the yaw
  range more than half a step past the last regular one: 16 stations, the
  envelope test pins bearing -90. Nearest-first search is unaffected.
* **The reach offset is capped at a quarter of the object's width**
  (`grasp.aim_offset()`): see the 20 mm cube below.

## The picks, and what they taught

1. 40 mm cube at 212 mm radius: REFUSED before moving. The grown offset
   (5 + 0.30 per mm beyond 160 = 20.6 mm) aimed at 232.7 mm, 0.7 mm outside
   the 232 mm envelope at grasp height. Retried with the panel's base offset
   at 0: found from the outer look at 182 / +9 (the failed search had
   MERGED four outer-look detections from J1 83..98 into 210 / +30, 28 mm too
   far -- the pooled outer-look merge is suspect, not investigated), passes
   7.3 -> 1.5 mm, held, 28 s.
2. Same cube at 180 mm, offset 0: passes 6.6 -> 4.0 -> 0.3 mm, held, 25 s.
3. Same cube, offset 5 (7 mm at that reach): passes 6.6 / 5.8 / 7.2 mm, never
   converged, landed 3.4 mm off -- and the user, watching, said the fingers
   took the cube "just in the middle", better than run 2. THE PASS RESIDUAL
   IS NOT THE GRIP QUALITY: offset 0 aims at the near face. 5 mm stays. With
   a non-zero base the place aimed beyond the drop point too (185 asked, 198
   aimed), the reason the cube crept outward pick after pick.
4. First drop at the fixed pose: search from J1 180 walked 7 stations
   (20 s), pick from 172 mm, drop, back to survey, 44.5 s. The re-look warned
   "nothing within 30 mm of the sweep's estimate" and used it; harmless here.
5. **20 mm cube, declared 20, at 168 / +40 (173 mm radius): closed on
   nothing** and shoved the cube 10 mm toward the base. The offset was 9 mm
   and the second pass overshot 3.4 mm (accepted, inside the 4 mm tolerance):
   the pads landed on the far edge. Retry at 162 mm (offset 6): held, 24 s.
   Hence the cap: 5 mm for a 20 mm cube, 40 mm cubes unchanged.
6. Two cubes one after the other, no declared width: the small one (colour
   rung read 29 mm -- the shadow again; the cap did not bite) at 161 / -23,
   passes 13.2 -> 2.1, 27 s; the 40 mm one at 145 / +78 (found at J1 65
   after two stations that saw it cut off), passes 10.7 -> 3.5, 31 s. Both
   dropped at the pose.

Pattern in every run: pass 1 lands 6..13 mm short, pass 2 corrects it and
overshoots by 3..4 mm, which the 4 mm tolerance accepts. Fine for 40 mm,
tight for 20 mm; the cap is the cheap answer, a smaller tolerance for small
objects the next one.

## Where everything was left

Arm at survey, torque on, 11.6 V. Table empty; both cubes on the floor/box at
the drop spot. Bridge and vision up, robot tree synced with the branch.
Panel on :8091 with the offset at 5. 270 tests, ruff clean.

---

# Status, 2026-09-21 (the robot's desktop: one X session with or without a monitor)

Designed offline in the morning, installed and checked on the robot at 11:20
with an HP P244 on the port: the driver forces DP-1, reads the monitor's EDID,
X and Mutter show one 1920x1080@60 monitor, gnome-remote-desktop (VNC 5900,
RDP 3389) is up, the Xorg log is clean. Still to do by hand: unplug the
monitor and confirm a 1920x1080 desktop over VNC, plug it back and confirm it
lights up without a restart (list at the end).

## What was wrong

The headless setup from the initial commit forced the DisplayPort head DP-0 to
report as connected (`ConnectedMonitor "DP-0"` added to the vendor
/etc/X11/xorg.conf, a 1920x1080 modeline in xorg.conf.d/20-headless-virtual.conf,
a DP-0/unknown entry in ~/.config/monitors.xml). That gives Mutter a monitor
with nothing plugged in, which VNC and gnome-remote-desktop need. But the
NVIDIA driver's ConnectedMonitor option REPLACES detection with its list, and
DP-0 is the head that is not wired to the connector on this carrier: Mutter's
own monitors.xml records both real monitors ever used here (the MPI7005 1080p
panel and the Lontium 1440x900 adapter) on DP-1. With a monitor plugged in,
DP-1 was ignored and the screen stayed black.

## What changed

- tools/20-headless-virtual.conf is self-contained now (its own Device
  "Tegra0Headless", Monitor and Screen; with no ServerLayout, Xorg takes the
  first Screen section) and forces DP-1, the physical head. The vendor
  /etc/X11/xorg.conf goes back to stock. With no monitor there is no EDID and
  the driver falls back to the modeline, 1920x1080@60; with a monitor, at boot
  or plugged in later, the driver reads its EDID as usual and Mutter applies
  the user's saved configuration for it. Unplugging returns to the virtual
  mode. No restart in either direction.
- tools/monitors.xml is Mutter's SYSTEM-WIDE fallback, installed as
  /etc/xdg/monitors.xml: only the DP-1/unknown (no EDID) entry, pinned to
  1920x1080@60. It applies to the jetson session and to the gdm greeter and
  never touches the per-user file; a real monitor has a real vendor and never
  matches it.
- tools/display_setup.sh: `sudo bash tools/display_setup.sh install [PORT]
  [--restart]`, `check`, `uninstall`. install also drops the stale DP-0 entry
  from the per-user monitors.xml files. tools/sync_robot.py now pushes
  tools/*.sh, *.conf, *.xml and *.service as well.

## Verified on the robot (11:20, HP P244 plugged into the port)

Installed with `sudo bash tools/display_setup.sh install`, gdm3 restarted.
Before: xrandr had `DP-0 connected 1920x1080 0mm x 0mm` (the forced phantom)
and `DP-1 disconnected`, while the kernel showed the HP on card1-DP-1 with a
256-byte EDID: the black-screen bug, exactly. After: the log says
`Using ConnectedMonitor string "DFP-1"` and `HP Inc. HP P244 (DFP-1):
connected`; xrandr `DP-1 connected primary 1920x1080+0+0 530mm x 300mm` (the
size comes from the EDID); Mutter reports ('DP-1', 'HPN', 'HP P244', serial)
with one logical monitor at 1920x1080@60; gnome-remote-desktop is active (VNC
5900, RDP 3389); no (WW)/(EE) in Xorg.0.log once NoDFPNativeResolutionCheck,
which this driver (L4T R36.4) does not know, was dropped from ModeValidation.

Learned on the way: DP-1 is "Internal TMDS" (HDMI-style signalling) and the
driver assumes a 165 MHz pixel clock for it without an EDID, which the
148.5 MHz modeline fits. Remote access is gnome-remote-desktop, not x11vnc:
tools/x11vnc.service was never installed and is deleted. Hostname `yahboom`.
The robot's WiFi (IISLab-AMEC) has no internet, so the laptop has to hop over
and back for every robot session: tools/wifi_hop.ps1.

## Still to verify by hand

1. Unplug the monitor: `bash tools/display_setup.sh check` should show
   `DP-1 connected 1920x1080` in xrandr with no EDID line, the Mutter monitor
   as ('DP-1', 'unknown', 'unknown', 'unknown') with one logical monitor,
   and a 1920x1080 desktop over VNC.
2. Plug it back in: it should light up within seconds, no restart. If it
   stays black, look in /var/log/Xorg.0.log for a new EDID after the plug;
   none means the driver does not re-read EDID under ConnectedMonitor, and a
   gdm restart on plug is the fallback.
3. Reboot with nothing plugged in, then plug in: same expectations.

If forcing DP-1 with nothing attached fails at modeset (DP-0 worked headless
because nothing is behind it) the fallback design is a boot-time switch: force
DP-0 as before when /sys/class/drm shows no connector, force nothing when one
is connected. That gives up hot-plug but keeps both cases working.

# Status, 2026-09-14 (first untagged pick succeeded; neural rung moved to YOLOE-26)

## The gripper fix is confirmed

The first pick of the day was the red 40 mm cube, no tag, through the neural
rung: found at 162 mm fwd / 0 mm left, 39 mm; opened to 57 mm BEFORE moving;
descended; "holding something"; placed at 168 / -78; back at survey in 22.3 s.
The reach pass landed 13.6 mm short of the 20 mm-beyond aim, i.e. 6 mm past
the cube centre and 6 mm up -- inside the grasp. So Thursday's 14.9 mm was a
passenger; the fingers were the fault, and `grasp.pick()`'s opening-first
order is right. The 20 mm reach offset stays.

## The neural rung is now YOLOE-26 (`yoloe-26s-seg-pf.pt`)

ultralytics 8.4.137 in the vision image ships the YOLOE models built on
YOLO26. All were run on the same four lab frames in the container, conf 0.10:

    yoloe-26s-seg-pf   red cube 0.86   tagged wooden cube 0.61   41 ms   <- served now
    yoloe-26m-seg-pf   red 0.85        tagged 0.85               55 ms
    yoloe-11s-seg-pf   red 0.63        tagged 0.82               43 ms   (was served)
    yolo26n-seg (COCO) red 0.79        tagged NOTHING            35 ms
    yolo26s-seg (COCO) red 0.45        tagged NOTHING            37 ms

The plain YOLO26 segmenters have 80 fixed COCO classes and only find what
resembles one (the red cube passes as a "stop sign"); they never see the
wooden cube and will not see a pale one. They stay one env var away
(`ROBOARM_VISION_MODEL=/app/models/yolo26n-seg.pt`, weights are in
`models/`), not the default. The masks from all of them go through the same
`range_block()`; nothing in the client cares which model answered.

Two things the new model showed, both fixed in `detect.objects()`
(`whole_objects()`, 3 new tests, 262 passing, ruff clean):

* every prompt-free YOLOE labels the WHOLE FRAME ("studio shot", "chemistry
  lab") on most shots -- an outline covering >= 80 % of the picture is
  dropped rather than listed as an 85 mm clipped thing;
* it labels PARTS too: the tagged cube came back as a 265 px "traffic sign"
  and a 178 px "direct" for the tag's inner square, 0.61 vs 0.53. Ranged as
  a cube of its own the inner square is a 27 mm block at the same spot, and
  would set a 45 mm opening for a 40 mm cube. An outline with >= 90 % of its
  area inside a larger one is now dropped.

Live on the robot after the change: with the cube half out of frame at the
right edge, the neural rung lists exactly one target ("shape", 38 mm, refused
as clipped) and nothing else.

## The outer rim: outer look fitted, search follows hints (afternoon)

The wooden cube put out at ~200 mm was missed by all 7 primary stations: it
sits beyond the primary look's far edge and only peeks into the far CORNERS
of the +25 and +50 stations (a pitched camera's far corners reach further
than the middle of its far edge). Bearing was never the problem; radius was.

Two things done about it, both on the robot and in the tree:

1. **The outer look is calibrated** (`data/table_homography.json`, `also:
   [outer]`, pose J2=29 J3=51 J4=12 J5=90). From that pose the picture holds
   ONE whole marker, so `calibrate_table.py` grew `--yaws`: the look is fitted
   from the board seen at several base yaws, each frame's table points turned
   back by -yaw into the unyawed look (the same J1 invariance the ring rests
   on) and pooled. `--outer --yaws=-24,-12,0,12,24` -> 12 points over three
   yaws (the ±24 frames held no whole marker), worst residual 5.9 mm, board
   seen 162..229 mm forward. The residual is J1 repeatability (about 1 deg =
   3.5 mm out there) plus lens distortion at the picture edges; expect a few
   mm of error in the outer band, against ±9 mm of grasp tolerance. The
   single-frame fit before it was 4 points / 0.0 mm residual -- exactly
   determined and untrustworthy at the frame edges. NOT --verify'd by hand.
   The search now reports **14 stations, 92.0 % of reach** (was 7 / 68.9 %).
2. **Hints.** A clipped detection used to be thrown away; now `Target.edges`
   records which picture sides the outline ran into (`detect.edges_touched`),
   `sweep.frame_sides()` says which side is far/near/left/right on the table
   for a look, and `sweep.hints()` turns that into the next look: out of the
   FAR edge -> the further-reaching look (outer) centred on that bearing; out
   of a SIDE edge -> this look yawed to centre the bearing; near edge ->
   nothing (too close to grasp). `Session._job_sweep` keeps a queue, puts
   hint looks at its front, visits each (look, J1) once, and logs
   "hint: ... -> outer J1=nn". 8 new tests (271 passing, ruff clean).

**Seen working on the robot** (second retry, board face down): at J1=90 the
primary saw the cube cut off at its far edge, logged the hint, and the very
next station was the outer look at that bearing (J1=97), 3 s in. The cube
was cut off there TOO -- from the outer frame its near edge maps to 198 mm,
so its centre was ~218 mm: beyond the arm (200 mm with the reach offset,
210 raw). The search then ran the remaining 12 stations and failed honestly
after 70 s; `sweep.beyond_reach()` now makes it say "beyond reach" when the
furthest look clips at its far edge.

**How much the outer look actually buys, computed from the fitted matrix:**
its far edge is 236 mm at the frame middle (primary 218), but its lens is
LOWER (179 vs 212 mm) so a 40 mm cube's top is lifted 1.29x about a nadir at
165 mm and leaves the top of the frame once the cube's centre passes ~198
mm. So for a 40 mm cube the outer look measures ~189..198 mm, the primary
129..189, and the planner reaches 200. Consistent, but thin: the rim the
outer look adds is 10 mm wide. A higher outer pose (lens higher, not just
further out) would widen it; the one chosen on 2026-09-10 optimised nadir
distance, not lift. Worth revisiting only if objects really need to sit at
190..210 mm.

**Then both limits were moved (late afternoon), and the outer-rim pick WORKS:**

* `config.SAFE_LIMITS[2]` floor 15 -> 5. It was only a margin from the servo
  end (0), not a collision, and it capped the model at 210 mm when the links
  stretch to 272. Model reach is now 238 mm (the reachable grid's max radius;
  test_sweep pins it), and probed with the arm 50 mm up the servos land at
  206 / 219 / 223 mm for asks of 215 / 225 / 235 -- the same 9-12 mm short
  as always. Coverage percentages all dropped because the denominator grew
  (one look 9.5 %, primary ring 47 %, both rings 92 %); tests re-pinned.
* The outer look re-chosen with tilt allowed to 20 deg (was 10): fingertip
  target 190 fwd / 125 up -> pose J2=61 J3=29 J4=11, lens ~233 mm up, nadir
  145. Fitted `--outer --yaws=-24,-12,0,12,24`: 28 points (the ±24 frames
  now hold two markers), worst residual 9.8 mm, band 133..266 mm at the
  frame middle; a 40 mm cube stays whole to ~225 mm (was 198). The residual
  is J1 repeatability plus lens distortion at the picture edges; the pick
  below says it is good enough.
* **The pick**: cube at 214 mm. Primary J1=90 saw it cut off at the far
  edge -> hint -> outer J1=109 saw it whole (41 mm) -> pick -> "holding
  something" -> placed at 185 / 0. Search 6.2 s (two stations), whole job
  26.7 s. Reach pass landed 16.4 mm short of the 234 mm aim, i.e. 4 mm
  past the cube centre, 0.8 mm up. That is the outer rim, closed.

**Evening, two more picks at the rim, one bug and one finding:**

* Bug: the cube was found whole by the outer look but refused as clipped. The
  detector returned TWO masks for it: cube+shadow (0.72, running out of the
  bottom of the frame) and the clean cube (0.71) inside it; `whole_objects()`
  dropped the clean one as a "part" of the bigger, then the bigger was
  clipped. Fixed: an outline that touches the frame edge is not a whole
  object and cannot swallow one (test added, 78 vision+sweep tests pass).
  Committed as b1a6675 is WITHOUT this fix; the fix is in the tree.
* Both hint kinds seen working in one search: primary J1=90 saw a LEFT-edge
  clip -> centring yaw to J1=82 -> far-edge clip there -> outer J1=74 -> cube
  whole (162 / +81, 42 mm). Search 6.2 s.
* Finding on the reach offset: with the default 20 mm offset (passes off)
  a pick at 200 / +60 landed the tips 1.6 mm from the AIM, i.e. 20 mm past
  the cube, and closed on nothing, shoving the cube 40 mm. The "model lands
  short" shortfall is not constant: 1.6, 6.8, 13.6, 14.2, 16.4 mm on today's
  five picks. With `reach_offset_mm = 0` (which turns the correction passes
  back on) the next pick converged in three passes (15.7 -> 8.6 -> 5.0 mm),
  landed 4.8 mm off, 6.6 mm up, and held; +3 s per pick. The panel is left
  at 0 for the session; `cfg.REACH_OFFSET_M` still defaults to 20. Flip the
  default after a few more picks at 0 -- the passes adapt, the offset guesses.

**2026-09-18, first pick with offset 5 mm + passes (commit bee5f2b):** the
touch probe on the 14th put the real tips ~5 mm short of the model (cube under
the model's 0 and +10, table at -10 and +20), so `REACH_OFFSET_M` went 20 -> 5
and the correction passes now run alongside it. Pick: primary J1=90 saw the
cube cut off at the LEFT edge -> centring yaw to J1=80 -> whole (160 / +71,
37 mm) -> passes 10.8 -> 4.3 -> 3.3 mm -> landed 0.2 mm from the aim, 9.7 mm
up -> held -> placed. 27.4 s. Watched by eye: the fingers took it BY THE MIDDLE (with offset 0 it had
been "barely on its end"). 5 mm + passes is the setting.

**First retry gotcha**: with the ChArUco board still FACE UP the neural rung
reads the board's 38 mm markers as 33 mm objects ("remove", 0.25); the search
stopped at the first one and closed on nothing. Board face down for picks.

## Colour rung, still open

On the red cube the colour rung read 175-179 fwd / 19-22 left, 24-28 mm,
against the neural 162 / 0, 39 mm. Its mask swallows the bounce-lit shadow
(red light off the white paper) and the far-corner edge fit runs through the
bumps. Thursday it read 40 mm on the same cube at a different spot. Tighten
the shadow rejection before trusting it again; use the neural rung meanwhile.

## The pick with the new model: done, 21.0 s

After a charge (the board had read 9.6 V; a reboot for charging also took the
bridge down, and the panel kept showing the stale 9.6 until the bridge was
back -- a refused connection leaves the last battery number on screen). At
11.6 V, `POST /api/auto/oneclick {"detector": "yolo", "drop_x": 165,
"drop_y": 0}`:

    stop sign  181 fwd / -22 left  36 x 36 mm  at J1=90
    refined from J1=103 to 177 / -29 (7.7 mm from the sweep), 38 mm
    opening to 57 mm for a 38 mm object, before moving
    fingers turned to close along +76 deg (J5=94)
    reach pass: fingertip 7.9 mm off (short by +6.8 fwd, -1.6 left, +3.7 up)
    fingertip landed 3.0 mm from the aim point, 11.2 mm up
    holding something -> placed at 185 fwd / 0 -> back at survey

So the mask from YOLOE-26 sets a correct width (38 vs the real 40), a correct
roll, and the grasp closes on the cube. The reach shortfall was 6.8 mm this
time against 13.6 mm in the morning, both inside the 20 mm offset.

Then the PALE cube -- plain wood, no tag, the case the colour rung was
expected to miss. The neural rung read it at 171 fwd / +6 left, 39 mm, 0.69
("parchment"); the colour rung, as it happens, also saw the tan wood at 0.70.
Neural-rung pick: opening 57 mm, roll +74 deg, reach pass 14.2 mm short of
the 20 mm-beyond aim, "holding something", placed at 185 / 0, 19.9 s. Two
untagged cubes of different colour picked through the same mask path.

## Where everything was left

* Arm at the survey pose, torque ON. Bridge and vision containers UP on the
  robot (`vision` recreated with the new model; it has no CLIP module, which
  the prompt-free model does not need). Laptop webapp on :8091.
* Robot tree synced (compose.yaml, detect.py, detector.py, export_engine.py,
  test_vision.py). Local changes NOT committed.
* The factory app's autostart is disabled on the robot
  (`~/.config/autostart/start_app.sh.desktop.disabled-by-roboarm`); after the
  reboot this morning only the two `docker compose ... up -d` were needed.
* `models/` on the robot now also holds yolo26n-seg, yolo26s-seg,
  yoloe-26n/s/m-seg-pf and yoloe-26s-seg (~170 MB).

---

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
