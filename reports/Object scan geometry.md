# The object scan needs the floor, not the object, as its camera reference -- the floor already measures the poses to a millimetre

**Bottom line.** The scan's faults -- the box too wide, a stop's reading 6 cm off, doubled edges, the
see-through ghost -- are camera-pose faults in x, y and yaw; height, tilt and roll are already right,
because every view fits the floor plane its own depth sees. Today x, y and yaw come from wheel odometry
(1-6 cm off per 30 deg leg) corrected by each view's own reading of the object, which leans towards the
camera and trusts single bad readings, then by sequential coloured ICP that cannot undo a centimetre once it
is in the model. The floor round the object is the better reference. On our two saved scans, registering
neighbouring views' orthorectified floor rings measures their disagreement directly -- 16 mm median with
raw odometry, 3.6-9 mm with the fuse's cameras, 1.2-3.7 mm with the splat's refined ones -- and a
floor-only (x, y, yaw) solve brings it to 0.5-0.7 mm median. The object follows only partly: from the
fuse's cameras the floor solve gives the can its sharpest wall (radial spread 4.2 -> 2.6 mm, radius 34.5
against 33 true), but no better box faces, and from raw odometry it smears the can although its views
agree on the floor -- the floor images carry a bias (the colour lens's distortion is not modelled) that
must be removed before the floor can be the pose source rather than the pose meter. The pipeline: hold the exposure; print the floor disagreement as the scan's pose metric;
add a ring of AprilTags for unambiguous absolute poses on any floor; solve all views at once with the loop
closed; TSDF; the splat's refinement for the last millimetre; a closer second ring for the top; sizes from
fitted faces. Feed-forward 3D models, SfM, neural SDFs and depth networks are second-order here. Topic 8
("Base localization while turning") fixes the base to ~0.5 cm on the fence at each stop; this report fixes
the camera on the floor; the two meet as prior and measurement.

Labels: [REAL] our capture or our computation on it, [BENCH] a published benchmark, [CLAIM] an author's
statement, [2nd] a secondary source or a search summary. Evidence and links: `research_notes/Object scan
geometry/` (`poses_and_registration.md`, `surfaces.md`, `coverage_shiny_colour.md`). The web-search budget
ran out partway; later checks used arXiv and GitHub pages directly.

## 1. The errors are pose errors, and the object-as-anchor is their source

How a view's camera is built today (`m3pro/tools/m3_scan.py fuse`, `m3pro/roboarm/m3scan.py`):

1. the arm model's camera, levelled on the floor plane its depth sees (`m3vision.level`: height, tilt and
   roll from the floor; x, y, yaw from the model),
2. placed in the world by the wheel odometry at that stop,
3. shifted in x and y -- never turned -- so that the object's reading in that view lands on the readings'
   median (`anchor`; SIGHT_PUSH 0.25 x radius when the top is not seen whole),
4. corrected by coloured ICP against the union of the views before it, the cloud cut to the object's box
   plus 4 cm, corrections over 3 cm or 8 deg rejected.

What the saved scans show [REAL]:

| | box 20261012_172202 (24 views) | can 20261012_165048 (36 views) |
|---|---|---|
| object's re-read, stop to stop | 8-59 mm (stop 4 -> 5: 59) | 7-33 mm |
| coloured ICP moved cameras | 0.7 cm median, 6.1 cm max | (same run) |
| the splat then moved them | 0.3-7.5 mm | 2.5-17.3 mm |
| size as fused | top 16.7 x 11.5 cm, sides 17.1 x 12.4 (true 16.4 x 11.0), 0.7 cm too tall | 16.7 cm tall (16.8), radius 31.0 mm (33) |

A one-sided reading of a curved thing leans 6-8 mm towards the camera (single-view arc fits 31.9 mm against
25.7 fused); a lean of b in every anchor pushes every view outward by b and grows the thing ~2b across.
Four attempts to estimate the lean from the data failed (memory, 2026-10-12). The problem is structural:
while the object places the cameras, its shape biases its own measurement, and one bad reading moves a
whole view. Coloured ICP adds a second problem: its colour term compares raw colours, and the DCW2's auto
exposure swings x0.89-1.20 between views.

## 2. The floor measures the poses to about a millimetre -- our check on the saved scans

Method (scratch scripts outside the repo, read-only on the scans): per view, a 1 mm/px top-down image of a
ring of floor 3-17 cm beyond the object (each floor cell projected into the colour image through that
view's camera; kept where the view's depth agrees with the floor within 15 mm, so the object and its
shadow drop out). Views of the same or neighbouring stops are registered pairwise (rigid ECC on a 3-level
pyramid, SIFT + RANSAC as a second start, correlation 0.97-0.99 median); a pair's disagreement is the rms
displacement over its overlap -- zero if both cameras were right. Then one (x, y, yaw) correction per view
from all pairs (least squares, pairs off by > 1.5 mm dropped), and the floor images rebuilt and registered
afresh as an independent check.

| camera set | box: floor disagreement median / p75 / p90 (mm) | can |
|---|---|---|
| raw odometry + levelled arm model | 16.4 / 20.9 / 31.1 | 16.4 / 24.3 / 31.6 |
| the fuse's (anchor + coloured ICP) | 3.6 / 9.2 / 30.7 | 9.0 / 15.2 / 21.8 |
| the splat's (refined) | 1.2 / 4.3 / 22.3 | 3.7 / 7.4 / 18.2 |
| floor-solved from raw | 0.7 / 2.3 / 29.6 (21 of 24 views tied) | 0.5 / 0.8 / 7.0 (35 of 36) |
| floor-solved from the fuse's | 0.7 / 1.2 / 9.8 (23 of 24) | 0.5 / 1.3 / 22.9 (34 of 36) |

What this shows [REAL]:

- **A pose metric the scan has lacked.** The floor ranks the camera sets as the splat's held-out scores
  do, with no ruler and no object model, in minutes of laptop CPU. Every fuse should print it.
- **The floor alone ties the views to under a millimetre** where its texture is good -- from raw odometry
  (which it moved 22-23 mm median, the drift) as well as from the fuse's cameras. That is better than the
  splat's own refinement (1.2-3.7 mm), which cannot cross centimetres.
- **The tails are where the floor is weak.** Box stops 0-1 did not tie from raw; p90 stays 7-30 mm where
  overlaps are plain grey floor or the repeating zebra print. A tag ring removes both.
- **The object follows only partly.** TSDF (1.5 mm, no ICP), sizes from fitted extents:

  | camera set | can radius / radial spread (mm), true 33 | box top face; sides (mm), true 164 x 110 |
  |---|---|---|
  | the fuse's | 30.6 / 4.2 | 154.6 x 104.6; 163.8 x 121.3 |
  | the splat's | 34.7 / 3.8 | 158.6 x 107.2; 169.6 x 122.1 |
  | floor-solved from the fuse's | **34.5 / 2.6** | 153.4 x 105.7; 169.1 x 126.2 |
  | floor-solved from raw | 41.9 / 11.2 (smeared) | 154.2 x 104.1; 168.9 x 109.1 (14 views) |

  (Box: views reading it > 15 mm off left out.) Started from the fuse's cameras, the floor gives the can its
  sharpest wall; started from raw odometry it agrees on the floor just as well (0.5 mm) yet smears the can --
  but the smear comes from one sector: leaving out the views whose own reading of the can is > 15 mm off
  (they cluster in stops 0-2, where the floor tied worst; a selection that uses the object, so read with
  care), the floor-from-raw cameras give a 0.8 mm radial spread over 16 views, the splat's 1.4, the fuse's
  4.2. Every well-registered set reads the can 1.5-2.8 mm wider than its true radius: a depth or ruler bias
  (shiny metal, the edge halo), not a pose.
  So the floor images carry a bias or a weak mode the floor cannot see by itself -- the colour lens's
  distortion (only K is used), the depth-to-colour alignment, the floor plane's tilt -- and the solve's
  ~3 deg turns about the object may be part of it. Box stops 1-4 also read the box 4-9 cm off under every
  camera set, the floor's included: those readings, which the anchor trusts today, are themselves suspect.
  The colour-lens calibration on the ChArUco board and a scan with tags decide it.

## 3. Tags or the printed mat: absolute poses for every view

The mat needs no hardware but repeats (the zebra stripes alias by one period) and exists only in our arena;
the Healthcare Challenge's floor will be plain. A ring of 6-8 AprilTags (or a ChArUco mat) 15-25 cm round
the object gives every view 2-4 uniquely identified tags. Published accuracy: AprilTag 36h11 ~1.4 mm average
position error and ~0.01 deg in TopoTag's robot-referenced benchmark [BENCH, 2nd]; BlobBoards (Aug 2026)
3.6-5.0 mm median against motion capture, AprilTag 70-89 % worse on the same boards [CLAIM]; a single small
ArUco 5.7 mm with very noisy orientation [BENCH, 2nd] -- boards and many corners, not single tags.
Ground-texture localisation (Micro-GPS, a camera 26 cm over the floor) reports "a few millimetres" [BENCH].
At 0.3-0.45 m the colour camera resolves ~0.6-0.9 mm per pixel, so sub-pixel corners on several tags should
put a camera to ~1 mm and ~0.1 deg (estimate, to measure). The lab has a ChArUco board; Yahboom's M3 stack
ships `apriltag_localization`. Tags on known spots also put the object into the arena's frame, and their
yaw is independent of the mat's texture -- the test of the floor solve's ~3 deg.

## 4. One global solve, then the splat's dense refinement

Floor pairs, tags, coloured ICP on the object, odometry between stops: all belong in one least-squares
problem with the loop closed, not in a sequence. Open3D's multiway registration (Choi et al. 2015: odometry
and loop-closure edges, line-process pruning, Levenberg-Marquardt) is the 6-DoF ready-made [BENCH]; with
the cameras levelled on the floor, a 3-DoF graph is better conditioned and a few dozen lines of NumPy (as in
our check). BundleFusion and BAD SLAM are the dense RGB-D bundle-adjustment references, built for video
and "highly sensitive" to RGB-depth sync and rolling shutter [CLAIM]. The splat's per-view refinement is
already our dense photometric + depth BA; it belongs after the floor solve, and NEXT.md's "the scan's cloud
from the splat's cameras" becomes a check -- the splat should then move cameras ~1 mm -- not a fix.

## 5. Feed-forward models and SfM are initialisers here, not the answer

VGGT (CVPR 2025) reaches camera AUC@30 deg 88.2 on CO3Dv2 in ~0.2 s for 10 frames [BENCH]; MapAnything
(3DV 2026) takes our intrinsics, depth and poses and reaches two-view AUC@5 deg 73.5 with images +
intrinsics + depth, 93.2 with poses too [BENCH]; Depth Anything 3 claims 44 % better pose accuracy than VGGT,
Pi3X takes pose/intrinsics/depth conditioning, VGGT-Omega (CVPR 2026) scales with data [CLAIM]. These are
degree-level numbers -- at 0.38 m, 0.5 deg is ~3 mm -- and the weights are mostly CC-BY-NC (MapAnything also
has an Apache-2.0 checkpoint). Keep MapAnything (intrinsics + depth + odometry in) as the fallback start
when the odometry jumps by decimetres.

COLMAP 4.0 (Mar 2026) integrated GLOMAP with pose and position priors, 4.1 a GPU bundle adjuster, 4.2
learned matchers [BENCH]; InstantSfM v2 claims metric depth priors in a GPU BA [2nd]. A COLMAP run with
odometry priors would give a joint 6-DoF BA of floor and object features -- plan B, at the cost of a
dependency and the classic failure cases (repeating print, symmetric objects; COLMAP's RRA@15 on 10-view
CO3D objects is 31.6 [2nd]).

## 6. Surfaces: the TSDF is enough once the poses are right; read sizes from fitted faces

The DCW2 reads a 30 mm cube's height to ~0.5 mm at the scan's range, and the TSDF (1.5 mm voxels, 6 mm
band) averages 24-36 views. Re-fused with four camera sets, the box's top stays at 44.5-45.5 mm over the
floor [REAL] -- its ~5 mm excess height is not an x/y/yaw fault, so caliper the box and read the DCW2 on its
glossy top against a matte card before blaming the fusion.

Surface-aligned splats now reach 0.50-0.54 mm Chamfer on DTU (GSPrior, PGSR, QGS) against 0.80 for 2DGS,
0.74 GOF and 0.61 Neuralangelo, in 20-50 min against > 12 h [BENCH, GSPrior's table] -- image-only, at
1600x1200, with perfect cameras. With measured depth their gain for us is in the depth's holes (the can's
lid) and at sharp edges; gsplat 1.5.3, which we run, has 2DGS with pose gradients since 1.5.2 [BENCH], so
"2DGS + our depth loss + a TSDF of rendered depths" is the cheap trial. Neural SDFs (NeuS2, Neuralangelo)
are slow; BundleSDF wants 10 Hz video [CLAIM].

The ruler matters as much as the method: the fuse's 98th percentile of heights reads a noisy top high
(46.9 mm against the TSDF top face's 45.5), and hulls of all points read sides wide. Fit the box's top,
floor and side planes, the can's circle on a 3-13 cm band and its rim plane, and report them beside the
floor disagreement and the held-out PSNR / depth error.

## 7. Coverage, shine and colour

- **Coverage is geometry.** From 0.38 m the arm sees a 15 cm thing at 38-56 deg elevation, from 0.20 m at
  67-89 deg: a higher look needs a closer stop. A second ring of 4-6 stops at ~0.28-0.30 m, its views
  chosen greedily from the first ring's TSDF (surface voxels seen by fewer than 3 views), costs ~40-60 s.
  NBV research (FisherRF, ActiveGS, ObjSplat) targets open-ended exploration; its gain over a fixed
  two-ring plan on one object is unmeasured.
- **The ghost is mostly pose.** The box splat's see-through turntable frames (13-15 of 24) also show the
  yellow floor line twice, a few cm apart, in the sector of the stop-4 -> 5 jump -- two registrations of one
  floor. Fix the poses; then render with SH degree 0 (the console's `.splat` is DC-only anyway) and drop
  Gaussians in space the depth saw as empty.
- **Shine and dark print.** Median 3-5 depth frames per view (PLAN.md M4 A planned it), more lid views from
  the second ring, saturated highlights masked; learned refiners that take RGB + raw depth exist (Camera
  Depth Models, ICLR 2026: AbsRel 0.071 -> 0.031 on D435 data, 0.15 s on an RTX 4090, no Orbbec model;
  Prompt Depth Anything; Marigold-DC), and FoundationStereo works on IR stereo pairs if the DCW2's driver
  gives both (not established). For a demo object, matte spray.
- **Colour: hold it, then model the rest.** `arena_map.py camlock` already holds the DCW2's exposure through
  the driver's services (white balance cannot be read on the DCW2). Holding it also helps every photometric
  registration (coloured ICP, ECC, the splat's refinement). Then per-view gains (`scan_photometry.py` fits
  them) before the TSDF's colours; PPISP only for the splat; Open3D's colour map optimisation for a sharp
  mesh texture.

## 8. The pipeline for our scan, ranked by gain over effort

| # | step | method | runs on | expected gain | effort | check against the ruler |
|---|---|---|---|---|---|---|
| 1 | a ruler and a baseline | fitted faces (box planes; can circle + rim); caliper the box; three re-scans of one object for the spread | laptop | knowing which change is real | 0.5 day | spread of three scans; caliper vs the DCW2 on a matte card |
| 2 | the floor as a metric | the floor-ring disagreement (median, p90) printed by every fuse | laptop | a pose number for every scan and every change | 0.5 day | the camera sets rank raw > fuse > splat, as measured |
| 3 | hold the exposure | `set_color_auto_exposure false` + `set_color_exposure` at the scan's start (as `camlock`) | robot | colours x0.9-1.2 -> ~1.0; steadier photometric registration | 1-2 h | `scan_photometry.py`: exposure range ~1.00 |
| 4 | calibrate the colour lens | distortion + D2C check on the ChArUco board; undistort before the floor images | laptop | removes the suspected bias behind the floor solve's ~3 deg | 0.5 day | reprojection < 0.3 px; the floor solve's yaws shrink |
| 5 | a tag ring + one global solve | 6-8 AprilTags 15-25 cm round the object; tags, floor pairs, odometry and object ICP edges in one (x, y, yaw) graph; replaces `anchor` + sequential ICP | laptop, later Orin | floor disagreement 3.6-9 -> < 1 mm median (measured on the floor); the object's sizes to ~2-3 mm (expected, unproven) | 2-3 days + print | box faces within 2-3 mm; can radius within 1-2 mm; floor p90 < 3 mm |
| 6 | splat after the floor | the splat starts from the solved cameras | 5090 | sharper edges, no doubled floor | hours | splat moves cameras <= 1-2 mm; held-out PSNR up |
| 7 | depth averaging | median of 3-5 depth frames per view while the arm stands | robot | less noise, fewer holes on dark print | hours (+~0.3 s a view) | top-plane RMS; holes on the lid |
| 8 | a closer second ring | 4-6 stops at ~0.28-0.30 m, views from the TSDF's least-seen voxels | robot | the top and the lid seen; no high-view ghost | 1-2 days (+40-60 s a scan) | held-out high views; unseen voxels |
| 9 | splat surface | gsplat 2DGS + our depth loss + TSDF of rendered depths | 5090 | edges and the lid where the DCW2 reads nothing | 1-2 days | same ruler, TSDF vs 2DGS |
| 10 | splat clean-up, colour | free-space culling; SH 0-1 for console splats; per-view gains; PPISP if still needed | 5090 / laptop | ghost and floaters; even colours | hours to 2 days | turntable at 45 and 70 deg; floor cell spread |

Not now: feed-forward models for poses (degree-level), COLMAP SfM (plan B), BundleSDF and neural SDFs
(video or hours), depth networks (the DCW2 is ~10x better where it reads). Within ~5 minutes a scan:
steps 2-5 and 7 are seconds to a few minutes of CPU on the laptop (the floor registration could move to
the Orin), the TSDF tens of seconds; steps 6, 9 and 10 are 1-5 minutes on the 5090.

## 9. Risks

- **The floor solve may carry a bias.** ~3 deg median rotations about the object, no better box faces,
  and a smeared can when started from raw odometry, on the saved scans. Until the lens distortion and D2C
  are checked and a tag scan agrees, treat the floor as a metric and a refinement from the fuse's cameras,
  not yet as the pose source.
- **The print repeats and wrinkles.** Zebra stripes can alias a pair by one period; creases break
  planarity by millimetres. ECC from a close start, loop closure and pruning catch most; tags remove it.
- **No mat elsewhere.** Plain floors have no texture: the tag ring is then the method.
- **The ring is the only reference.** A tall or wide object hides more of it and steep views see less;
  tags 15-25 cm out and the object's own ICP edges in the same graph cover that.
- **Navigation still uses the object.** The drive round it re-reads the object; topic 8's standing fence
  fix (~0.5 cm) is the base's answer while driving, and the tags could re-read the base at each stop.
- **The box's height** (+5 mm in every camera set) may be the DCW2's depth on a glossy black top or the
  box itself; no pose method moves it.
- **Two scans, one person's scripts.** Every number in section 2 comes from two saved scans and quick
  scratch code; repeat on new scans with the exposure held before changing the robot's code.

## Conclusion

The object scan needs a better reference, not a better optimiser. The object cannot place the cameras
that measure it -- its reading leans and sometimes jumps by centimetres -- while the floor round it can:
on the saved scans it measures the cameras' disagreement (16 mm raw, 3.6-9 mm fused, 1.2-3.7 mm after the
splat) and a floor-only solve brings the views to 0.5-0.7 mm median on the floor. The object follows
only partly -- the can's wall sharpens from the fuse's start (spread 4.2 -> 2.6 mm), the box does not, and a
start from raw odometry smears -- so the lens distortion must be calibrated and a tag scan must confirm
before the floor replaces the anchor. Do it in this order: a ruler, the floor metric and the exposure held; the colour lens
calibrated; a tag ring and one global solve; the splat after, for the last millimetre; a closer ring for the
top; 2DGS surfaces once the poses are right. Feed-forward models, SfM and neural SDFs stay in reserve.
