# Poses and registration: where the scan's cameras come from (state on 2026-10-27)

Scope: how to give each of the scan's 24-36 RGB-D views a camera pose good to a millimetre or two --
what limits the current poses (wheel odometry + the object's own reading + sequential coloured ICP), the
floor as a reference (our check on the two saved scans), fiducial tags and ground texture, global
optimisation (pose graphs, RGB-D bundle adjustment, the splat's own refinement), feed-forward 3D models
(VGGT, MapAnything, Pi3, Depth Anything 3, VGGT-Omega), structure-from-motion with priors (COLMAP 4.x /
GLOMAP, InstantSfM), object-centric SLAM (BundleSDF) and the lidar pinning the base (topic 8 of
RESEARCH.md section 10, researched separately: only the link is given here). Surfaces are in
`surfaces.md`, coverage, shiny things and colour in `coverage_shiny_colour.md`. Labels: [REAL] our capture,
[BENCH] a published benchmark, [CLAIM] an author's claim, [2nd] a secondary source or search summary. The
web-search budget ran out partway; later checks used arXiv abstracts and GitHub pages only.

## 1. The current poses are cm-level because the object is the anchor

### Takeaway
Each view's camera height, tilt and roll come from the floor plane its own depth sees -- good to about a
millimetre and a tenth of a degree. What is left, x, y and yaw on the floor, comes from wheel odometry that
drifts 1-6 cm per 30 deg leg, corrected by the object's own reading in that view (the "anchor") and then
by sequential coloured ICP against the growing model. Three structural limits follow: the anchor leans
6-8 mm towards the camera on a curved thing and trusts a single bad reading (the box's stop 5 jumped
59 mm); the anchor moves views but never turns them, so yaw is left to ICP; and frame-to-model ICP with no
loop closure, a 3 cm / 8 deg acceptance gate, only 4 cm of floor kept round the thing and a colour term
that assumes constant brightness (the exposure swings +-20 %) cannot undo a cm error once it is in the
model.

### Evidence
- [REAL] `m3pro/roboarm/m3vision.py` `level()`: the camera's height, tilt and roll from the floor plane
  (RANSAC + least squares, 6 mm tolerance), "its place along the floor and its heading from the arm
  model, which a plane cannot tell".
- [REAL] `m3pro/roboarm/m3scan.py` `anchor()`: each view's camera is shifted in x, y only, so that its
  reading of the thing lands on the readings' median; SIGHT_PUSH 0.25 x radius when the top is not seen
  whole. `fuse()`: each view's cloud cut to the thing's box (radius + 4 cm), coloured ICP against the
  union of the views before it (coarse to fine 8/4/2 mm), corrections past 3 cm or 8 deg not believed,
  then the TSDF.
- [REAL] Per-stop re-reads of the thing's centre moved 10-59 mm (box) and 7-33 mm (can) from stop to stop
  (`scan.json` "centres"); coloured ICP moved cameras 0.7 cm median, 6.1 cm max, two views found no
  correspondences (memory, 2026-10-12); the splat's refinement moved them 3-6 mm median, 17 mm max.
- [REAL] Tried and failed to remove the lean (memory 2026-10-12): top-band centre, occupancy minimisation,
  wide floor-ring coloured ICP, all-vs-all point-to-plane ICP (stuck at 26 mm); drift+lean least squares
  unstable (b swings -45..+25 mm).
- [BENCH] Coloured ICP: point-to-plane plus a colour term on the tangent plane; "geometric constraints
  alone cannot stop two planar surfaces from sliding" -- the colour term is what locks a flat floor, and
  it compares raw colours between scans. -- [Open3D coloured ICP](https://www.open3d.org/docs/latest/tutorial/pipelines/colored_pointcloud_registration.html)

### Inferences
- Floor-ring coloured ICP probably failed for three reasons that a 2D image registration avoids: a
  colour term biased by the exposure swings; a tiny basin on a print with large uniform patches
  (gradients only at edges); and a 3D cloud of the floor at 2-8 mm voxels, coarser than the print's edges.
- Removing the object from the pose problem removes the lean by construction: the thing is then measured,
  not used to place the cameras.

## 2. The floor already measures the poses to about a millimetre: our check on the saved scans

### Takeaway
The printed floor round the object is a pose meter the scan has lacked. Registering the views' floor
rings pairwise ranks the three camera sets we have exactly as the splat's held-out scores do -- raw
odometry ~16 mm, the fuse's cameras 3.6-9 mm, the splat's 1.2-3.7 mm median disagreement -- and a
floor-only (x, y, yaw) solve over all neighbouring pairs makes the views agree on the floor to 0.5-0.7 mm
median, better than the splat's dense refinement. The object follows only partly: started from the fuse's
cameras, the floor solve gave the can its sharpest wall (radial spread 4.2 -> 2.6 mm, radius 34.5 mm
against 33 true, the fuse's 30.6) but no better box faces; started from raw odometry it agrees on the floor
just as well yet smears the can (radius 41.9 mm) -- so the floor images carry a bias or a weak mode that the
floor alone cannot see (the colour lens's distortion is not modelled; the solve turns cameras ~3 deg
about the object). A sector of box views (stops 1-4) also reads the box 4-9 cm off under every camera set.
A calibrated lens and a scan with a tag ring decide it.

### Method [REAL, scratch scripts outside the repo, read-only on `m3pro/out/scan/`]
- Per view, a 1 mm/px top-down image of the floor ring 3-17 cm beyond the object's radius (box: 119-259 mm
  from its centre; can: 62-202 mm), each floor cell projected into the colour image through that view's
  camera (K only), kept where the view's depth agrees with the floor within 15 mm (the object and its
  shadow drop out).
- Pairs: views of the same or neighbouring stops (up to 2 apart), floor overlap >= 0.8 dm^2; rigid ECC
  (`cv2.findTransformECCWithMask`, a 4/2/1 px pyramid) from the identity and from SIFT + RANSAC, the better
  correlation kept (median 0.97-0.99), rotations over 12 deg rejected. Disagreement = rms displacement of
  three points spread over the pair's overlap.
- Solve: one (x, y, yaw) per view, rotation about the object's centre, point residuals, gauge on the
  means, Gauss-Newton; pairs disagreeing with the solution by > 1.5 mm dropped twice; then the floor images
  rebuilt with the new cameras and registered afresh (an independent check of the solve).

### Results [REAL]

| camera set | box 20261012_172202, 24 views: floor disagreement median / p75 / p90 / max (mm) | can 20261012_165048, 36 views |
|---|---|---|
| raw (odometry + levelled arm model) | 16.4 / 20.9 / 31.1 / 86.3 (86 pairs) | 16.4 / 24.3 / 31.6 / 47.8 (203 pairs) |
| the fuse's (anchor + coloured ICP) | 3.6 / 9.2 / 30.7 / 33.3 | 9.0 / 15.2 / 21.8 / 32.1 |
| the splat's (refined) | 1.2 / 4.3 / 22.3 / 25.7 | 3.7 / 7.4 / 18.2 / 32.1 |
| floor-solved from raw | 0.7 / 2.3 / 29.6 / 101.6 (21 of 24 views connected; solve residual 0.43 mm) | 0.5 / 0.8 / 7.0 / 33.1 (35 of 36; residual 0.18 mm) |
| floor-solved from the fuse's | 0.7 / 1.2 / 9.8 / 22.2 (23 of 24; residual 0.55 mm) | 0.5 / 1.3 / 22.9 / 32.4 (34 of 36; residual 0.34 mm) |

- The floor solve moved the raw cameras 21.6 mm median (box, max 66.9) and 23.0 mm (can, max 51.7) -- the
  odometry's drift -- and turned them 3.7 / 2.5 deg median. From the fuse's cameras it moved them 8.9 mm
  median (max 31.1) and turned them 3.5 deg median (-6.1..+3.7) on the box.
- The tails (p90 7-30 mm) are pairs and views the floor does not tie well: the box's stops 0-1 (3 views
  not connected from raw) and pairs whose overlap is plain grey floor or the repeating zebra print.
- The object under each camera set (box; the thing's centre as each view reads it, `find_thing`):
  the fuse's cameras put 18 of 24 readings within 15 mm of their median (by construction: the anchor),
  but views of stops 1-3 and 8 read it 46-93 mm off; the floor-solved cameras from raw put 14 within
  15 mm and stops 1-4 38-69 mm off. TSDF (1.5 mm, no ICP) without those views, box faces from fitted
  extents (true 164 x 110, top 40 by the earlier note): fuse's 154.6 x 104.6 top face, 163.8 x 121.3 sides;
  splat's 158.6 x 107.2 / 169.6 x 122.1; floor from raw 154.2 x 104.1 / 168.9 x 109.1 (14 views); floor
  from the fuse's 153.4 x 105.7 / 169.1 x 126.2. The top at 44.5-45.5 mm in every set.
- The can (TSDF of all views, a circle fitted to the 3-13 cm band; true radius 33 mm): the fuse's cameras
  30.6 mm, radial spread 4.2 mm; the splat's 34.7 / 3.8; floor-solved from the fuse's **34.5 / 2.6** -- the
  sharpest wall of the four; floor-solved from raw 41.9 / 11.2 -- smeared, although its views agree on the
  floor to 0.5 mm median. The floor moved the can's cameras 16.7 mm median from the fuse's (max 35.6) and
  turned them 3.1 deg median (max 6.8).
- Dropping the views whose own reading of the can lies > 15 mm from the readings' median (a selection that
  uses the object, and `find_thing` here searched 20 cm round the centre against the scan's 15 cm -- read
  with care): floor-solved from raw, 16 views, radius 35.8 mm, radial spread **0.8 mm**; floor from the
  fuse's, 18 views, 34.8 / 1.3; the splat's, 17 views, 35.4 / 1.4; the fuse's, 24 views, 30.3 / 4.2. The
  dropped views of the floor-from-raw set cluster in stops 0-2 (64-77 mm off), the sector the floor tied
  worst (view 01_0 not connected). Every well-registered set reads the radius 1.5-2.8 mm over the true
  33 mm: a depth or ruler bias (a shiny metal wall, the depth's edge halo), not a pose.
- The anchor's lean, seen with cameras that never used the object (floor from raw): each view's reading of
  the can sits 14 mm median towards its own camera (mean 14.6; 20 cm search) -- the scan's own estimate
  was 6-8 mm. Either way the object is a biased reference for its own cameras.

### Inferences
- As a metric the floor is ready: it ranks camera sets without a ruler, costs minutes of CPU, and should be
  printed by every fuse (median and p90 of the views' floor disagreement).
- As a pose source it is promising but unproven at the object: the floor fixes where each camera is
  relative to the floor; the object's 3D points come from the depth image. Whatever differs between the
  two -- the colour lens's distortion (only K is used), the depth-to-colour registration, the floor plane
  fit's tilt -- turns floor-perfect cameras into object errors. Calibrate the colour distortion on the
  ChArUco board and check the D2C alignment before trusting the floor solve's yaws.
- The sector reading 4-9 cm off under every camera set (stops 1-4 of the box) says those readings, not
  only the cameras, are suspect (a partial view, a neighbouring thing within the 20 cm reach): exactly the
  readings the anchor trusts today.

## 3. Tags and ground texture: absolute poses from the floor, with or without the printed mat

### Takeaway
A planar target seen in every view gives each camera an absolute pose with no drift and no loop to
close. Single fiducial tags report ~1-1.5 mm position error in benchmarks and boards do better;
ground-texture localisation (a camera reading the floor's own texture) reports "a few millimetres". Our
printed mat is a ready-made, high-contrast target round the arena's middle; a ring of 6-8 AprilTags (or a
ChArUco mat) round the object is the robust version -- unique IDs instead of the mat's repeating zebra
stripes, and it works on any floor (the Healthcare Challenge's will not be printed).

### Evidence
- [BENCH] TopoTag's benchmark (robot ground truth): AprilTag 36h11 average position error 1.402 mm (max
  3.824), rotation 0.010 deg (max 0.018); TopoTag 1.011 mm; distances not in the snippet.
  -- [arXiv 1908.01450](https://arxiv.org/pdf/1908.01450) via a search summary [2nd]
- [CLAIM] BlobBoards (Aug 2026): median translation error 3.6-5.0 mm against motion capture; AprilTag's
  median translation error 70-89 % higher on the same boards; detection 80 % vs AprilTag 74 % vs ArUco
  58 %; under 50 % occlusion 69 % vs 0 %. -- [arXiv 2608.28830](https://arxiv.org/abs/2608.28830)
- [BENCH] A single ArUco marker against robot forward kinematics: 5.73 +- 1.44 mm translation, very noisy
  orientation (29.75 +- 48.92 deg) -- single small markers are a poor reference; boards and many corners
  are the remedy. -- [arXiv 1804.01708](https://arxiv.org/pdf/1804.01708) via a search summary [2nd]
- [BENCH] Micro-GPS (ICRA 2019): global localisation from ground texture "accurate to a few millimeters",
  a downward camera 260 mm above the ground; a localisation counts as failed at more than 30 px (4.8 mm)
  or 1.5 deg. -- [arXiv 1710.10687](https://ar5iv.arxiv.org/html/1710.10687)
- [2nd] Ground-texture localisation outdoors loses reliability over time and weather (Schmid,
  dissertation 2023); a monocular ground-texture SLAM exists (arXiv 2303.05946). -- search summaries of
  [Schmid 2023](https://publikationen.ub.uni-frankfurt.de/files/82263/2023-12-14_JFSchmid_Dissertation_A.pdf),
  [arXiv 2303.05946](https://arxiv.org/pdf/2303.05946)
- [REAL] The lab has a ChArUco board (the M3's twin model measured on it to ~5 mm rms; NEXT.md); Yahboom's
  M3 stack ships `apriltag_localization` (m3pro/PLAN.md, "Their stack").

### Inferences
- Tags' error budget at our range: at 0.3-0.45 m the DCW2's colour camera has ~0.6-0.9 mm per pixel
  (fx 481 px), so sub-pixel corners on 4-6 tags per view put a camera to ~1 mm and ~0.1 deg; better than
  the arm model (~5 mm rms) and the anchor (6-8 mm lean) by a wide margin. Tags at 15-25 cm from the
  thing's centre stay in view at every stop (the views see the floor ~10-30 cm round it).
- The tags also give the scan's scale and a fixed world frame for the console (the thing's position in the
  arena if the tag ring is placed on known spots).

## 4. Solve once, globally: a pose graph over all pairs, then the splat's dense refinement

### Takeaway
Whatever the pairwise measurements (floor, tags, coloured ICP, odometry between stops), they belong in one
least-squares problem over all views with the loop closed, not in a sequence. Open3D's multiway
registration (Choi et al. 2015) does it in 6-DoF with line-process outlier pruning; for our levelled
cameras a 3-DoF (x, y, yaw) graph is better conditioned and is a few dozen lines of NumPy. The splat's
per-view pose refinement is already a dense photometric + depth bundle adjustment: it is the right last
step (mm) but cannot recover cm errors, so it must start from floor-registered poses. BundleFusion and
BAD SLAM are the dense RGB-D BA references, built for video SLAM.

### Evidence
- [BENCH] Open3D multiway registration: pose graph of odometry edges (`uncertain=False`) and loop-closure
  edges (`uncertain=True`), information matrices from point clouds, Levenberg-Marquardt global
  optimisation with `max_correspondence_distance`, `edge_prune_threshold` (0.25 in the example) and
  `reference_node`; a first pass prunes false loop closures, a second solves without them.
  -- [Open3D multiway registration](https://www.open3d.org/docs/latest/tutorial/pipelines/multiway_registration.html)
- [CLAIM] BundleFusion: per-frame global pose optimisation over the complete RGB-D history from sparse
  features plus dense geometric and photometric matching, in real time; "quality on par to offline
  methods". -- [arXiv 1604.01093](https://arxiv.org/abs/1604.01093)
- [CLAIM] BAD SLAM (CVPR 2019): fast direct RGB-D bundle adjustment; direct RGB-D SLAM found "highly
  sensitive" to rolling shutter, RGB-depth synchronisation and calibration errors; code BSD-3, last pushed
  2024-02-14. -- [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.html),
  [badslam listing](https://awesome.ecosyste.ms/projects/github.com%2FETH3D%2Fbadslam) [2nd]
- [REAL] The splat's refinement on our scans: cameras moved 3-6 mm median (17 mm max); held-out PSNR box
  24.4 -> 25.5 dB, can 20.3 -> 21.5 dB (PLAN.md M4 C).
- [BENCH] Open3D colour map optimisation (Zhou & Koltun 2014) refines camera poses (and optionally local
  image warps) against a fixed mesh for texture sharpness -- a texture step, not a geometry step.
  -- [Open3D colour map optimisation](https://www.open3d.org/docs/latest/tutorial/pipelines/color_map_optimization.html)

### Inferences
- With absolute floor/tag poses the graph is nearly trivial (every view tied to the floor); its value is
  combining the floor with the odometry between stops (a weak prior) and with the object's own coloured
  ICP edges (where the floor is occluded).
- After the floor solve, NEXT.md's open item "the scan's cloud from the splat's cameras" becomes a check,
  not a fix: the splat's corrections should drop to ~1 mm.

## 5. Feed-forward 3D models give degree-level poses: an initialiser, not the answer

### Takeaway
VGGT and its successors estimate all cameras in a second or two, and MapAnything and Pi3X accept our
intrinsics, depth and odometry poses as inputs -- but their published accuracies are measured in degrees
(AUC at 5-30 deg thresholds) and centimetres; at our 0.38 m, 0.5 deg is ~3 mm at the object. They are
worth keeping as a sanity check when odometry fails badly, and their weights are mostly non-commercial.

### Evidence
- [BENCH] VGGT (CVPR 2025): camera AUC@30 88.2 on CO3Dv2 (91.8 with BA), 85.3 on RealEstate10K
  (93.5 with BA), ~0.2 s feed-forward / ~1.8 s with BA for 10 frames on one H100; DTU dense
  reconstruction without known cameras: accuracy 0.389, completeness 0.374, overall 0.382 (units not
  stated on the page; DTU's protocol is mm); backbone 0.31 s / 5.6 GB for 20 frames, 1.04 s / 11.4 GB for
  50 at 336x518. -- [arXiv 2503.11651](https://arxiv.org/html/2503.11651)
- [BENCH] MapAnything (3DV 2026; v3 23 Jan 2026), two-view pose on ETH3D / ScanNet++ v2 / TartanAir v2,
  AUC@5 deg: images only 51.9; + intrinsics 58.5; + intrinsics + depth 73.5; + intrinsics + poses 91.7;
  all four 93.2 (Pow3R 33.7-75.6). Inputs: images, intrinsics or rays, depth (z), OpenCV cam2world poses,
  metric flag; images resized to 518 px; checkpoints CC-BY-NC 4.0 or an Apache-2.0 variant; its code also
  wraps VGGT, VGGT-Omega, Pi3, Pi3-X, MASt3R, MUSt3R, Pow3R, DA3.
  -- [arXiv 2509.13414](https://arxiv.org/html/2509.13414v1); [GitHub](https://github.com/facebookresearch/map-anything)
- [CLAIM] Depth Anything 3 (13 Nov 2025): beats VGGT by 44.3 % in camera pose accuracy and 25.1 % in
  geometric accuracy on average; with or without known poses. -- [arXiv 2511.10647](https://arxiv.org/abs/2511.10647)
- [CLAIM] Pi3X (28 Dec 2025): conditioning on poses, intrinsics and depth, approximate metric scale,
  "slightly better" than Pi3; code BSD-3, weights CC BY-NC 4.0. -- [GitHub yyfz/Pi3](https://github.com/yyfz/Pi3)
- [CLAIM] VGGT-Omega (CVPR 2026 oral; 14 May 2026, rev. 25 Oct 2026): quality scales with model and data;
  77 % better camera estimation than the previous best on Sintel. -- [arXiv 2605.15195](https://arxiv.org/abs/2605.15195)
- [REAL] Our own: DA3 single-view depth was clearly worse than DA V2 Large on our close looks (NEXT.md,
  2026-10-16), i.e. these models' depth is no substitute for the DCW2's.

### Inferences
- MapAnything with our intrinsics + depth + odometry poses is the best-placed of them (it is trained to
  use exactly these inputs), and could replace the odometry as the starting point if a run's odometry
  jumps by decimetres. It cannot reach the 1-2 mm that the floor gives.

## 6. Structure-from-motion with priors: a full 6-DoF BA, if the floor approach fails

### Takeaway
COLMAP 4.x now has GLOMAP's global mapper, pose and position priors, a GPU bundle adjuster and learned
matchers; InstantSfM adds metric depth priors to a GPU SfM. A COLMAP run on our 36 pictures with the
odometry as pose priors would give a joint 6-DoF BA of floor and object features at sub-pixel
reprojection -- but it is a heavier dependency, depth enters only as a scale check, and the mat's
repeating stripes and symmetric objects are the classic failure cases. Keep it as plan B.

### Evidence
- [BENCH] COLMAP changelog: 3.13.0 (07/11/2025) pose-prior mapper more robust to outlier priors; 4.0.0
  (03/14/2026) GLOMAP integrated, gravity in pose priors, optional position priors in absolute pose
  refinement, ALIKED + LightGlue; 4.1.0 (06/26/2026) Caspar GPU BA ("often 1-2 orders of magnitude faster
  than the Ceres CUDA backend" on medium to large problems); 4.2.0 (08/31/2026) LoMa matchers via ONNX,
  disconnected view graphs as several models. No depth priors or RGB-D items.
  -- [COLMAP changelog](https://colmap.github.io/changelog.html)
- [CLAIM] InstantSfM (Oct 2025, v2): GPU, sparse-aware BA, up to ~40x faster than COLMAP; v2 embeds metric
  depth priors in global positioning and BA through a depth-constrained Jacobian.
  -- [arXiv 2510.13310](https://arxiv.org/abs/2510.13310v1) via a search summary [2nd]
- [2nd] On CO3Dv2 with 10 views, COLMAP RRA@15 31.6 / RTA@15 27.3 and GLOMAP 45.9 / 40.3 (Light3R-SfM's
  table); MASt3R-SfM better than both -- sparse, wide-baseline object views are hard for classical SfM.
  -- search summary of [Light3R-SfM](https://arxiv.org/pdf/2501.14914)

## 7. Object-centric SLAM and the lidar pinning the base

### Takeaway
BundleSDF (object-centric RGB-D tracking + neural SDF) expects video at ~10 Hz with small motion between
frames, not 30 deg stops. Pinning the base with the lidar and the arena map (topic 8) would cut the
odometry's drift to the map's centimetre level, but the camera would still carry the arm model's ~5 mm rms
and the sag; it helps the drive round the thing and is a good prior for the floor solve, not a mm pose.

### Evidence
- [CLAIM] BundleSDF: 10 Hz, online pose graph of at most K = 10 memory frames, correspondences rejected
  past 1 cm. -- [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.html),
  [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wen_BundleSDF_Neural_6-DoF_CVPR_2023_supplemental.pdf)
- [REAL] Topic 8 (RESEARCH.md section 10): the base board's clock 55-75 ms ahead of the Jetson, lidars at
  ~7 Hz reading ~1 cm long, EKF odom at ~6-7 Hz, RTAB-Map's grid +7 cm; at 0.3-0.6 rad/s a scan sweeps
  2.5-5 deg. The twin model's camera is ~5 mm rms on the ChArUco board (memory, 2026-10-26).

## Gaps
- No published accuracy for registering a robot's oblique RGB-D views to a printed floor mat was found;
  the numbers in section 2 are our own, on two scans.
- No ChArUco- or AprilTag-ring accuracy at 0.3-0.45 m with a 640x480 camera was found with stated
  distances; the ~1 mm estimate in section 3 is an inference to measure.
- MapAnything's multi-view pose numbers (2-100 views) are only in a figure; not read.
