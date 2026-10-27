# Surfaces: TSDF, splat surfaces, neural SDFs, and measuring against the ruler (state on 2026-10-27)

Scope: which surface method gives the scan's best shape once the cameras are right -- the TSDF we use
(Open3D 0.20, 1.5 mm voxels), the surface-aligned Gaussian splats (2DGS, PGSR, GOF, RaDe-GS, QGS and their
2025-2026 successors), and neural SDFs (NeuS2, Neuralangelo, BundleSDF) -- and how to read the box's and
the can's sizes so that a 2 mm improvement is visible. The poses are in `poses_and_registration.md`;
coverage, shiny surfaces and colour in `coverage_shiny_colour.md`. Labels: [REAL] our capture, [BENCH] a
published benchmark, [CLAIM] an author's claim, [2nd] secondary source or search summary. The web-search
budget ran out partway; later checks used arXiv abstracts and GitHub pages only.

## 1. With the DCW2's depth, the TSDF is not what limits the shape -- the poses and the ruler are

### Takeaway
The DCW2 measures the surface to about a millimetre at the scan's 0.3-0.45 m, and the TSDF averages 24-36
views of it; what moves the box's and the can's sizes by 5-15 mm is where the views are put (the poses)
and how the size is read off the cloud (the "ruler"). Our own check: re-fusing the box with four
camera sets (the fuse's, the splat's, two floor-solved ones) moves its side lengths by 5-17 mm but leaves
its top's height at 44.5-45.5 mm -- the height is not an x/y/yaw fault (the camera's height and tilt come
from the floor in every view).

### Evidence
- [REAL] The DCW2 reads a 30 mm cube's height to ~0.5 mm (task brief; NEXT.md depth duel: the DCW2's own
  floor is the duel's reference).
- [REAL] The current fuse (`m3pro/roboarm/m3scan.py` `fuse`): Open3D `UniformTSDFVolume`, voxel 1.5 mm,
  `sdf_trunc` 6 mm, depth cut to a box round the thing (its radius + 4 cm); `ScalableTSDFVolume` returned
  nothing in Open3D 0.20 (memory, 2026-10-09).
- [REAL] Re-fused (TSDF as above, no ICP) and measured with fitted faces (method in section 4), the box
  (true 164 x 110 mm footprint): with the fuse's cameras top face 159.6 x 105.4 mm, mid-height sides 164.8
  x 121.6, top at 45.5 mm over the floor; with the splat-refined cameras 159.4 x 106.3, 170.3 x 123.0,
  44.5 mm; with views that read the box > 15 mm off left out, and with floor-solved cameras, the sides
  range 163.8-169.1 x 109.1-126.2 mm and the top 44.5-45.5 mm (`poses_and_registration.md`, section 2).
  The fuse's own reading (98th percentile of heights, hull of all points) was 46.9 mm tall, top 16.7 x
  11.5 cm, sides 17.1 x 12.4 cm (memory, 2026-10-12). Scratch scripts outside the repo.
- [REAL] The anchor's lean: a one-side reading of the can lands 6-8 mm towards the camera (circle fits
  of single-view arcs 31.9 mm against the fused 25.7 mm, memory 2026-10-12); SIGHT_PUSH 0.25 x radius was
  fitted on one can. A lean of b in every view's anchor moves every view outward by b and grows the fused
  thing by ~2b across -- the size of the box's side excess.

### Inferences
- The box's extra 5 mm of height in every camera set is a depth-or-ruler matter: a glossy black top read
  high by the structured light, a lid that bulges, or the true height not 40 mm. Measure the box with a
  caliper and read the DCW2's depth on its top against a matte card laid on it before blaming the fusion.
- TSDF resolution is not the limit: 1.5 mm voxels with a 6 mm band already sit under the poses' errors.
  Lower voxel sizes would only show the pose errors more sharply.

## 2. Surfaces from splats: PGSR-class methods reach ~0.5 mm on DTU; for us their gain is in the depth's holes

### Takeaway
On DTU (image-only, 49-64 views, ~0.5 m away) the surface-aligned splats now reach 0.50-0.54 mm Chamfer
(GSPrior, PGSR, QGS) against 0.74-0.80 for GOF and 2DGS and 0.61 for Neuralangelo, in 20-50 minutes per
scene against Neuralangelo's >12 h. All of them finish with a TSDF of rendered depths. Our scans already
have measured depth, so their gain is where the DCW2 has none (the can's lid, glossy print) and at sharp
edges; gsplat 1.5.3, which we run, has a 2DGS mode with pose gradients, so a "2DGS + our depth loss +
rendered-depth TSDF" variant of `train_splat.py` is the low-effort way to try it.

### Evidence
- [BENCH] DTU mean Chamfer (mm) and training time, as tabulated by GSPrior (CVPR 2026, Table 2): NeuS 0.84
  (>>12 h), Neuralangelo 0.61 (>>12 h), 2DGS 0.80 (19.2 min), GOF 0.74 (1 h), SuGaR 1.33 (1 h), GSDF 0.80
  (32 min), QGS 0.54 (48 min), PGSR 0.53 (40 min), GSPrior 0.50 (42 min); GSPrior's prior is a TSDF fused
  from the Gaussians' own rendered depths, "without external knowledge".
  -- [arXiv 2603.19682](https://arxiv.org/html/2603.19682v2)
- [CLAIM] RaDe-GS: Chamfer "comparable to NeuraLangelo on the DTU dataset" with "similar training and
  rendering time as traditional Gaussian Splatting" on T&T. -- [arXiv 2406.01467](https://arxiv.org/abs/2406.01467v1)
- [2nd] 2D-SuGaR (May 2026): DTU Chamfer 0.67, its mesh refinement slow. -- search summary,
  [ResearchGate](https://www.researchgate.net/publication/404396389_2D-SuGaR_Surface-Aware_Gaussian_Splatting_for_Geometrically_Accurate_Mesh_Reconstruction)
- [CLAIM] MILo (Jun 2025, rev. Oct 2025): the mesh extracted differentiably during training, "an order of
  magnitude fewer mesh vertices than previous methods". -- [arXiv 2506.24096](https://arxiv.org/abs/2506.24096)
- [CLAIM] DN-Splatter (WACV 2025): depth and normal priors regularise splats of indoor scenes and allow
  direct mesh extraction. -- [arXiv 2403.17822](https://arxiv.org/abs/2403.17822)
- [CLAIM] PGSR (TVCG per its DOI): planar Gaussians, "fast training and rendering", better than 3DGS- and
  NeRF-based methods. -- [arXiv 2406.06521](https://arxiv.org/abs/2406.06521)
- [BENCH] gsplat release notes: 2DGS support since v1.4.0; "2DGS now supports camera gradients (camera
  pose)" in v1.5.2 (13 May 2025); v1.5.3 (4 Jul 2025) is the latest release.
  -- [gsplat releases](https://github.com/nerfstudio-project/gsplat/releases)
- [REAL] Our splat (gsplat 1.5.3, 3DGS, depth loss 0.5, pose refinement): held-out depth error 3.2-3.8 mm
  (box), 6 mm (can); held-out PSNR 21.5-25.5 dB; 53-61 s for 7000 steps on the 5090
  (`splat_train.log`, PLAN.md M4 C).

### Inferences
- DTU's 0.5 mm is at 1600x1200 and with perfect cameras; at 640x480 and with our cameras, expect the
  surface to inherit the pose error. The order is poses first, then a splat surface.
- The cheapest experiment: in `train_splat.py` switch `rasterization` to gsplat's 2DGS rasterizer (normals
  and median depth come with it), keep the depth loss and the pose refinement, add a depth-normal
  consistency term (2DGS's own), then fuse the rendered depths of the training views in the same TSDF and
  measure with the same ruler. PGSR's multi-view terms are a second step only if 2DGS helps.

## 3. Neural SDFs: smooth and watertight, but slow or built for video

### Takeaway
Neuralangelo (0.61 mm on DTU) takes >12 h; NeuS2 is ~100x faster than NeuS but older than the splat
surfaces that now beat it; BundleSDF -- the object-centric RGB-D option -- tracks at 10 Hz from video with
small motion between frames and caps its online pose graph at 10 memory frames, which our 30 deg stops
do not resemble. None is a first step for a 2-minute scan.

### Evidence
- [BENCH] Neuralangelo 0.61, NeuS 0.84 (both >>12 h) -- GSPrior's Table 2 (above).
- [CLAIM] NeuS2 (ICCV 2023): "two orders of magnitude" faster than NeuS (~8 h). -- [arXiv 2212.05231](https://arxiv.org/abs/2212.05231)
- [CLAIM] BundleSDF (CVPR 2023): near real-time (10 Hz) 6-DoF tracking of an unknown object from RGB-D
  video with a neural object field learned with a pose graph; the supplement caps online pose-graph frames
  at K = 10 and rejects correspondences farther than 1 cm.
  -- [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.html),
  [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wen_BundleSDF_Neural_6-DoF_CVPR_2023_supplemental.pdf)

### Inferences
- If a watertight mesh is needed (Isaac Sim colliders), Poisson or the TSDF's marching cubes on good poses
  are enough for a box or a can; a neural SDF earns its hours only on thin or concave shapes.

## 4. The ruler: read sizes from fitted faces, not from percentiles of the cloud

### Takeaway
A mm-level claim needs a mm-level ruler. The fuse's 98th percentile of heights reads a noisy top high;
hull or min-area rectangles of all points read sides wide (flying pixels, the edge halo); the TSDF erodes
the top face's rim. Fit primitives instead -- box: the top plane against the floor plane, the four side
planes (RANSAC) for length and width; can: a circle fitted to a mid-height band and the rim plane --
and report them beside two pose checks that need no ruler: the views' misregistration on the floor and
the held-out views' PSNR and depth error.

### Evidence
- [REAL] On the same box fuse, the 98th percentile of heights gives 46.9 mm, the TSDF top face's height
  mode 45.5 mm; the top face's 1-99 % extent is 159.6 x 105.4 mm, the mid-height (12 mm to the top
  minus 12 mm) side extent 164.8 x 121.6 mm (true 164 x 110): the long side right, the short side 11 mm
  wide -- a pose fault on the box's long faces, seen only by separating faces from sides.
- [REAL] The can's fused radius: 31.0 mm by the fuse's reading (33 true; memory 2026-10-12).

### Inferences
- The test protocol for every pipeline change: the same two saved scans (box 20261012_172202, can
  20261012_165048) plus one new scan with the floor or tags; per scan: box top height, L, W from faces;
  can radius from a 3-13 cm band, height from the rim; held-out PSNR / depth error; floor misregistration
  median and p90. A change is real when it moves a size by more than the spread of three re-scans
  (unknown yet -- measure it once).
