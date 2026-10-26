# Isaac Sim 5.0 / 5.1 / 6.x side by side on Windows 11 (Dionysos, hekate) and which version to standardise on

Research date: 2026-10-26. Machines: Dionysos (Win 11 Education build 26200, RTX A4500 20 GB, driver 581.42, i7-8700K, 64 GB, Isaac Sim 5.0.0 standalone at C:\isaac-sim-standalone-5.0.0-windows-x86_64, shared by ~19 accounts) and hekate (Win 11, RTX 5090 32 GB, driver 617.14, Isaac Sim 6.0.0 standalone + Isaac Lab 3.0.0-beta2.patch1).

**The premise "5.1 or 6.0" is out of date.** As of October 2026 the newest GA release is Isaac Sim 6.2.0 (14 Oct 2026), and Isaac Lab 3.0.0 GA (20 Oct 2026) targets Isaac Sim 6.2. Isaac Sim 6.1.0 GA came out 10 Sep 2026, and a 7.0.0 Alpha 1 pre-release appeared 18 Sep 2026. See the timeline under Q5.

## Q1. Official system requirements of 5.1 / 6.x on Windows, and drivers (does 581.42 satisfy 6.0? does 5.0 run on newer drivers?)

### Takeaway
Dionysos's driver 581.42 is exactly the Windows driver NVIDIA tested Isaac Sim **6.0.0** on, so 6.0.x needs no driver change there. Isaac Sim **6.1 / 6.2** were tested on Windows **595.97**, and **Isaac Lab 3.0.0 states 595.97 as the Windows minimum**. Standardising on 6.2 + Isaac Lab 3.0.0 therefore means a machine-wide driver upgrade on the shared Dionysos. NVIDIA publishes no statement on running Isaac Sim 5.0 under an R595 or newer driver.

### Cited Findings
- **Isaac Sim 6.0.0 requirements (Windows): OS Windows 11 only.** "Windows 10: Not supported" (Microsoft ended Win 10 support on 14 Oct 2025). Minimum GPU GeForce RTX 4080 / 16 GB VRAM, Good RTX 5080 / 16 GB, Ideal RTX PRO 6000 Blackwell / 48 GB. RAM 32 / 64 / 64 GB. CPU: i7 7th gen (4 cores) minimum, i7 9th gen (8 cores) good. Storage 50 GB SSD minimum. **Tested drivers: Linux 580.95.05, Windows 581.42.** The page lists one tested driver per OS for all tiers and no separate minimum. GPUs without RT cores (A100, H100) are unsupported. Below 16 GB VRAM, scenes rendering more than 16 MP per frame may not fit. — [Isaac Sim 6.0.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)
- **Isaac Sim 6.1.0 requirements:** the same hardware tiers, Windows 11 only, **tested drivers Linux 595.58.03 and Windows 595.97.** — [Isaac Sim 6.1.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
- **"latest" requirements page (6.2 era):** the same tiers, with Linux 595.58.03 and Windows 595.97 listed as tested. It also says Ubuntu 22.04 support is deprecated, and that outbound HTTPS to `omniverse-content-production.s3-us-west-2.amazonaws.com` must be allowed for the default online asset root. — [Isaac Sim requirements (latest)](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- **Isaac Sim 5.1.0 requirements:** Windows 10/11, the same GPU tiers, **tested drivers Linux 580.65.06 and Windows 580.88.** The page banner says "Isaac Sim 5.1.0 is no longer supported". — [Isaac Sim 5.1.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- **Isaac Sim 5.0.0:** the README lists "Windows 10/11 or Linux (Ubuntu 22.04)", minimum RTX 4080, Python 3.11, and no driver number (it points to NVIDIA Driver Requirements). The versioned 5.0.0 requirements page returns 404. — [IsaacSim v5.0.0 README](https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/README.md)
- **Isaac Lab 3.0.0 minimum drivers: Linux x86_64 595.58.03, Linux aarch64 580.159, Windows x86_64 595.97.** The default CUDA 13 stack (PyTorch 2.11.0 cu130) needs Turing or newer (CC 7.5+). The page says Warp's own minimum driver is not enough for the default GPU training environment. — [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)
- **Omniverse (Kit) Technical Requirements, validated Windows Workstation drivers: R570 573.42 (Ada/Ampere/Turing), R580 581.42, R595 595.97.** "Newer drivers may function properly, but have not been fully tested for stability and compatibility." The RTX Ampere architecture (CC 8.6, the A4500's class) is listed as supported, without DLSS Frame Generation, SER or Opacity Micro-Map. Kit minimum hardware is RTX 3070 / 16 GB RAM. — [Omniverse Technical Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- **Windows 11 update KB5074109:** "Vulkan use requires NVIDIA drivers with a specific fix: R580 >= 582.41 or 595.97 for R595." The same page notes that the Linux Azure 595.58.03 preview driver "is not compatible with Kit versions older than 109.0.5 and 110.1.0", which shows that driver/Kit incompatibilities do occur. — [Omniverse Technical Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- **The 595.xx branch breaks Isaac Sim 5.1 on Blackwell GPUs.** NVIDIA staff (VickNV) wrote on 13 May 2026 and 11 Jun 2026 that the crash is "caused by a known incompatibility between the 595.xx driver branch and Blackwell GPUs." The fix given was to downgrade to 580.88 (Windows). The thread never confirms the fix worked. — [NVIDIA forum: 5.1.0 crash on Windows Server 2025 / RTX PRO 6000 Blackwell](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-shortly-after-startup-on-windows-server-2025-with-rtx-pro-6000-blackwell/370054)
- A user thread title reports that an Isaac Sim 5.1 GUI crash on an RTX 5070 Ti (Blackwell) was "fixed by driver downgrade to 591.74" (from 595.79). This is a user report, not staff. — [NVIDIA forum thread 365335](https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335)
- Kit has a driver-version check. The Omniverse page gives `--/rtx/verifyDriverVersion/enabled=false` as a workaround for misreported versions (535.256+ on Vulkan). — [Omniverse Technical Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)

### Inferences
- **6.0.x on Dionysos: no driver change.** 581.42 is the 6.0.0 tested driver and the Omniverse-validated R580 driver.
- **6.1 / 6.2 on Dionysos: officially needs 595.97.** That is a machine-wide change on a shared PC: it affects every account, the existing 5.0 install, and the Omniverse Launcher apps that other users (e.g. mImrich) run.
- **5.0 on R595 is untested, but the known failure may not apply.** The documented R595 failure is specific to Blackwell, and the A4500 is Ampere, so that particular bug probably does not hit Dionysos. NVIDIA says nothing about Kit 107 (5.0) on R595 for Ampere. Treat it as "probably works, must be tested", with a rollback (keep the 581.42 installer).
- **hekate (617.14) is newer than every validated branch.** It meets Isaac Lab 3.0.0's 595.97 minimum. Its RTX 5090 is Blackwell, so any 5.x install there would risk the known Blackwell/new-driver crashes, which is another reason not to standardise on 5.x.
- **KB5074109 caveat.** If Dionysos has this Windows update installed and Isaac Sim uses Vulkan there, 581.42 sits below the 582.41 fix. 5.0 runs fine there today, so either the update or the Vulkan path is not involved. An in-branch R580 update (≥582.41) would be the smallest driver change if that ever bites. It would not reach Isaac Lab 3.0.0's 595.97 minimum.
- **The A4500 sits below the listed minimum tier, but nothing excludes it.** It is below the RTX 4080 "minimum" in raw speed, but it has RT cores, 20 GB VRAM (above 16 GB) and is an Omniverse-supported Ampere RTX part. The docs warn that "some tutorials and benchmarks may not run" below minimum, especially multi-sensor workflows. The i7-8700K (6 cores) falls between the minimum and good tiers, and 64 GB RAM meets good/ideal.

### Gaps
- No official tested Windows driver for Isaac Sim 5.0.0 was found: the versioned requirements page is 404 and the README has no number.
- No NVIDIA statement on Isaac Sim 5.0 (Kit 107) under R595 or newer drivers on Ampere GPUs.
- Whether Isaac Sim 6.2 refuses to start, or only warns, on 581.42 was not found. Isaac Lab calls 595.97 a "minimum"; Isaac Sim pages only say "tested".
- Whether Windows 11 *Education* differs from Pro in support was not mentioned anywhere (only "Windows 11").
- Which graphics API (Vulkan or DX12) Isaac Sim uses by default on Windows was not confirmed. The 5.0 log on Dionysos would show it.

## Q2. Can two standalone installs coexist? Caches, data dirs, env variables, isolation

### Takeaway
Yes, in practice. Standalone zips are self-contained folders, and Kit keeps per-user config/data in **version-numbered** subfolders under `%LOCALAPPDATA%\ov\data\Kit\Isaac-Sim\<ver>`. Logs and the shader cache are per user, under `%USERPROFILE%`. For full isolation, launch each version with `--portable-root <dir>`. The documented pitfalls: never rename or move an extracted install afterwards, and never set Isaac/Kit variables machine-wide (`ISAACSIM_ASSET_ROOT` overrides everything).

### Cited Findings
- **Default Windows paths:**
  - App: `C:\isaacsim`
  - Shader cache: `%userprofile%\AppData\Local\ov\cache\Kit`
  - Logs: `%userprofile%\.nvidia-omniverse\logs\Kit\Isaac-Sim`
  - Configs and data: `%userprofile%\AppData\Local\ov\data\Kit\Isaac-Sim`
  - `user.config.json`: `...\ov\data\Kit\Isaac-Sim\5.0\user.config.json` (the example shows a version subfolder)

  — [Isaac Sim 6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html)
- **Isolation flags:**
  - `--portable` enables portable mode (the root defaults to the Kit path).
  - `--portable-root PATH` "enables portable mode and places the data, cache, and logs folders at the given path".
  - `--clear-cache` and `--clear-data` clear `$cache` / `$data` before startup.
  - `--reset-user` skips loading `user.config`.
  - `clear_caches.bat` and `warmup.bat` (shader cache) ship in the Windows package.

  — [Isaac Sim 6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html)
- **Asset root precedence:** `ISAACSIM_ASSET_ROOT` (env) "overwrites the setting regardless of any value provided by a .kit file or command-line argument". It is followed by `--/persistent/isaac/asset_root/default`, then the `.kit` file, then `extension.toml`. Asset roots are versioned (e.g. `.../NVIDIA/Assets/Isaac/6.0`). — [Isaac Sim 6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html)
- A Windows 11 user cleared caches by hand under `C:\Users\<USER>\AppData\Local\ov\data\kit\Isaac-Sim Full\5.0\` (and `4.5`), i.e. one subfolder per version. — [NVIDIA forum 343372 (via search result)](https://forums.developer.nvidia.com/t/isaac-sim-5-0-and-4-5-on-windows-11-hangs-at-0-icons-lights-distant-light-png/343372)
- **Renaming breaks installs (Windows).** A user's 4.5.0 workstation install at `C:\isaacsim` lost material creation (albedo/normal/UV components, path errors) after the folder was renamed, and renaming it back fixed it. NVIDIA staff (VickNV, 24 Jul 2025): "specify different installation paths for each version during the initial installation process". Staff believed the problem was Windows-specific. The thread was closed without a confirmed root cause (20 Oct 2025). — [NVIDIA forum 339998](https://forums.developer.nvidia.com/t/installing-multiple-versions-of-isaacsim-without-the-depreciated-launcher/339998)
- Isaac Sim 6.0.0 GA added "support for running multiple Isaac Sim instances in parallel on a single machine", but it is listed under Docker (Linux-only container). — [Isaac Sim 6.0.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html)
- A Windows 11 user reported running two Isaac Sim versions (2023.1.1 and 4.5.0) at once. This is a user anecdote, not a guarantee. — [NVIDIA forum 323749 (via search result)](https://forums.developer.nvidia.com/t/can-i-use-both-isaac-sim-4-5-0-and-2023-1-1-on-the-same-work-station/323749)
- The 6.0.1 Setup Tips page does not mention `ISAAC_PATH`, `CARB_APP_PATH`, `EXP_PATH` or `OMNI_KIT_ACCEPT_EULA`. — [Isaac Sim 6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html)

### Inferences
- **Other users are untouched by default.** All cache, data and log paths live in the launching user's profile, so installing and running 6.x as Hercules does not touch other users' `%LOCALAPPDATA%\ov` or their own Isaac/Omniverse setups. The only shared things are the GPU (VRAM), the disk, the driver, and machine-level env vars and registry.
- **5.0 and 6.x already separate their config.** The `Isaac-Sim\<ver>` subfolders keep 5.0 and 6.x config apart even without portable mode. The shader cache folder `ov\cache\Kit` is shared per user. It is normally keyed by content/driver, but if a version ever misbehaves, `--portable-root C:\Users\Hercules\isaac-data\6.x` gives it a private cache, data and log tree and removes any doubt.
- **`ISAAC_PATH`, `CARB_APP_PATH`, `EXP_PATH` are expected to be process-local.** They are normally set by the package's own `setup_python_env.bat` / `python.bat` for that process only, so two folders do not collide unless someone set them machine- or user-wide. Verify with `type C:\isaac-sim-standalone-5.0.0-windows-x86_64\setup_python_env.bat` and `[Environment]::GetEnvironmentVariables('Machine')` / `('User')`. Also check that no `ISAACSIM_ASSET_ROOT` is set globally, or 6.x would load a 5.0 asset tree.
- **Extract to the final path.** Unzip 6.x straight into its final folder (e.g. `C:\isaac-sim-standalone-6.x.y-windows-x86_64`, mirroring the 5.0 naming), run any post-install script there, and never rename or move it afterwards.
- **The Omniverse Launcher is independent of standalone zips.** The forum title calls it deprecated. Leave other users' Launcher installs alone.

### Gaps
- No official NVIDIA page explicitly documents "two standalone versions side by side on Windows".
- The exact contents of `clear_caches.bat` (whether it wipes all versions' caches or one) were not found. Read it before running it, because it may clear the 5.0 caches too.
- Whether the shader cache (`ov\cache\Kit`) is safely shared between Kit 107 (5.0) and Kit 110 (6.x) is not documented.

## Q3. pip route (isaacsim wheels in a venv) vs standalone zip

### Takeaway
Both work on Windows. The **pip/uv route is now NVIDIA's documented way to pair Isaac Sim with Isaac Lab 3.0.** Isaac Lab 3.0's installation page uses uv, pulls `isaacsim[all,extscache]==6.2.0.0`, and does not describe linking a standalone binary. It also fits our uv workflow and isolates each version in its own venv. Python is 3.12 for 6.x and 3.11 for 5.x. The standalone zip remains the simplest for plain `python.bat` scripts like our current 5.0 jobs.

### Cited Findings
- **Isaac Sim 6.0 pip:**
  - Python: "Isaac Sim requires **Python 3.12**."
  - Commands: `pip install isaacsim[all,extscache]==6.0.0.1 --extra-index-url https://pypi.nvidia.com`. Bundles are `all`, `extscache`, `compatibility-check`, `ros2`.
  - PyTorch: `torch==2.11.0` from cu128 (CUDA 12) or cu130 (CUDA 13).
  - Windows: "it may be necessary to enable long path support".
  - EULA: accepted at the first import, or skipped by setting `OMNI_KIT_ACCEPT_EULA=YES`.
  - Headless: `SimulationApp({"headless": True})`, or `isaacsim isaacsim.exp.full.streaming --no-window`.
  - Some extension dependencies may need a manual `pip install`.

  — [Isaac Sim 6.0 pip install](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)
- **Isaac Sim 5.1 pip:** Python 3.11, `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com`, Windows long-path note, EULA via `OMNI_KIT_ACCEPT_EULA`. 5.1.0 is marked "no longer supported". — [Isaac Sim 5.1 pip install](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- **Isaac Lab 3.0.0 install:**
  - Recommended route: automatic `uv` setup (`uv run isaaclab ...`, `--extra isaacsim`), Python 3.12.
  - The `isaacsim` extra installs `isaacsim[all,extscache]` 6.2.0.0, with Warp 1.17 and PyTorch 2.11.0 / torchvision 0.26.0 on cu130.
  - Wheels come with `--index https://pypi.nvidia.com --torch-backend cu130`.
  - Supported OS: Windows 11 x86_64 (enable long paths before cloning, Command Prompt syntax).
  - "Isaac Sim 5.1 and older are unsupported."
  - No standalone Isaac Sim binary install is described.

  — [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)
- The 6.0.1 Setup Tips page lists Windows launchers in the standalone package: `python.bat`, `isaac-sim.bat`, `isaac-sim.streaming.bat` (headless with the WebRTC streaming service), `clear_caches.bat`, `warmup.bat`. — [Isaac Sim 6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html)

### Inferences
- **pip/uv pros:**
  - Fully isolated per venv; deleting the venv is the rollback.
  - Our Dionysos training venv already uses Python 3.12 + torch 2.11.0+cu128, which matches the 6.0 pip page.
  - Isaac Lab 3.0 expects this route.
  - Nothing is written to `C:\` outside the user profile, apart from the machine-wide long-path setting.
- **pip/uv cons:**
  - A large download per venv (the extscache packages) and a large uv cache in the user profile.
  - First-run cache builds.
  - The EULA prompt blocks an unattended SSH run unless `OMNI_KIT_ACCEPT_EULA=YES` is set in the process environment.
  - Long-path support is a machine-wide registry setting (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`). It is harmless to other users, but on this shared PC it counts as a system change, so check whether it is already enabled first.
- **Standalone pros and cons.** Pros: one folder, the `python.bat` workflow we already use on Dionysos, and the same layout as hekate's 6.0.0. Cons: bundled Python, so mixing in LeRobot/torch versions is awkward, and Isaac Lab 3.0 no longer documents linking to it.

### Gaps
- The exact pip version string for 6.0.1 (e.g. `6.0.1.0`) and 6.1 was not confirmed; only `6.0.0.1` and `6.2.0.0` were seen.
- No documented comparison of pip vs standalone feature parity on Windows was found.
- Download and installed sizes of the 6.x wheels and zips were not found.

## Q4. Isaac Lab versions vs Isaac Sim, features per version, Windows limitations

### Takeaway
- **Version pairing:** Isaac Lab 2.2.x works with Isaac Sim 4.5/5.0, and 2.3.x with 4.5/5.0/5.1. 3.0.0-beta2 (+patch1) pairs with 6.0 / 6.0.1, 3.0.0-EA with 6.1, and **3.0.0 GA (20 Oct 2026) with 6.2**.
- **Mimic is Linux-only.** The current docs say: "Isaac Lab Mimic is only supported on Linux". On our two Windows machines Mimic/SkillGen is off the table in any version.
- **What Windows still gets:** parallel envs and the multi-env `Camera` (formerly `TiledCamera`) work in Isaac Lab on Windows. The Newton backend and multi-backend support exist only in 3.x.

### Cited Findings
- **README compatibility table:**

  | Isaac Lab | Isaac Sim |
  |---|---|
  | release/3.0.0 and main | 6.2 |
  | v3.0.0-EA tag | 6.1 |
  | v3.0.0-beta2 tag | 6.0 |
  | v2.3.X | 4.5 / 5.0 / 5.1 |
  | v2.2.X | 4.5 / 5.0 |
  | v2.1.X, v2.0.X | 4.5 |

  The README shows a windows-64 badge and a python-3.12 badge. — [Isaac Lab README](https://github.com/isaac-sim/IsaacLab)
- **Isaac Lab release dates** (from the releases page; years inferred for some entries):
  - v3.0.0 GA, 20 Oct 2026: Isaac Sim 6.2, Python 3.12, "recommended baseline for new projects and migrations from Isaac Lab 2.x".
  - v3.0.0-EA, 16 Sep 2026: Sim 6.1.
  - v3.0.0-beta2.patch1, 2 Jul 2026: Sim 6.0.1.
  - v3.0.0-beta2, 17 Jun 2026: Sim 6.0. Windows wheels and docs, tiled camera folded into `Camera`.
  - v3.0.0-beta, 17 Mar 2026: Sim 6.0, "Ubuntu only", Windows "will be available soon".
  - v2.3.2, 2 Feb 2026: the final 2.x release.
  - v2.3.1, 4 Dec 2025.
  - v2.3.0, 28 Oct 2025: Sim 5.1, SkillGen with cuRobo.

  — [Isaac Lab GitHub releases](https://github.com/isaac-sim/IsaacLab/releases)
- **Mimic:**
  - "Isaac Lab Mimic is only supported on Linux."
  - Mimic supports visuomotor (camera) environments, e.g. `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`, and parallel generation via `--num_envs`. The examples use 300 envs for visuomotor, sized for an RTX PRO 6000 Blackwell.
  - Output format is HDF5. LeRobot format is not mentioned.

  — [Isaac Lab teleop & imitation (Mimic)](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html)
- **Isaac Lab 3.0 breaking changes vs 2.x** (beta release notes):
  - Quaternion order changed WXYZ → XYZW.
  - `.data.*` returns `wp.array` instead of `torch.Tensor` (wrap with `wp.to_torch()`).
  - Write methods split into `_index` and `_mask` variants.
  - `root_physx_view` → `root_view`.
  - `--headless` replaced by `--viz` / `--visualizer`.
  - `TiledCamera` folded into `Camera` / `CameraCfg`.
  - URDF and MJCF importers rewritten for Isaac Sim 6.0.

  Physics and platform notes from the same notes:
  - PhysX is the default backend; Newton is selected via presets (e.g. `physics=newton_mjwarp`). Surface grippers are PhysX-only.
  - Kit-less runs are possible with Newton.
  - Windows uv/pip installs do not include Pinocchio, so Pink IK tasks cannot run on Windows (beta2).

  — [Isaac Lab release notes](https://docs.nvidia.com/isaaclab/references/release-notes)
- Isaac Lab 3.0.0 also offers a kit-less Docker image (`nvcr.io/nvidia/isaac-lab:3.0.0-kitless`, Ubuntu 24.04, no Kit visualizer). Docker paths are Linux-oriented, with no Windows-specific instructions. — [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)

### Inferences
- **hekate is slightly out of step.** It runs Isaac Sim 6.0.0 with Isaac Lab 3.0.0-beta2.patch1, but patch1 targets 6.0.1. It is also on a beta API that GA has since superseded. New Isaac Lab code written now should target 3.0.0 GA on Isaac Sim 6.2, not beta2.
- **Our pipeline does not need Mimic.** It uses a privileged scripted expert, so Mimic is not required. What Isaac Lab adds for us is parallel envs plus the multi-env `Camera` (tiled rendering) for faster expert rollouts with images. LeRobot export would be our own converter, since Mimic and Isaac Lab write HDF5.
- **Mimic would need Linux.** If Mimic/SkillGen data augmentation is ever wanted, it needs a Linux machine. SkillGen also relies on cuRobo, which is not supported on the DGX Spark/aarch64 build and, per the Mimic page, sits on Linux.
- **Gripper physics.** Surface grippers are PhysX-only, and our M3 grip sim uses articulated fingers, so stay on PhysX (the default) until Newton is proven on our gripper.

### Gaps
- Windows support details for Isaac Lab 2.2/2.3 (e.g. which tasks or extensions are missing) were not fetched.
- Whether Isaac Lab 3.0.0 GA (as opposed to beta2) still lacks Pinocchio on Windows was not confirmed.
- No published tiled-camera throughput numbers for Ampere-class GPUs (A4500) were found.
- The SkillGen page was not fetched, so its OS restrictions are not quoted directly.

## Q5. What changed 5.0 → 5.1 → 6.0 → 6.1 → 6.2, and GA status as of Oct 2026

### Takeaway
6.x is GA and on its third minor release (6.2.0, 14 Oct 2026). 5.1 is officially unsupported.

The 6.0 step is the big break:
- Python 3.11 → 3.12.
- Kit 107 → 110.
- The `omni.isaac.*` compatibility shims were removed, and `isaacsim.core.api/prims/utils` deprecated in favour of `isaacsim.core.experimental.*`.
- Camera/RTX sensor APIs were deprecated or replaced.
- The URDF importer was reworked, and experimental Newton physics was added.

6.1 and 6.2 are incremental. They added Newton 1.6.1 and TiledCameraSensor improvements, and removed Cortex.

### Cited Findings
- **Isaac Sim GitHub releases** (page shows day/month; years inferred from the sequence and corroborated by the forum dates below):

  | Version | Date | Status |
  |---|---|---|
  | v5.0.0 | 8 Aug 2025 | GA |
  | v5.1.0 | 21 Oct 2025 | GA |
  | v6.0.0-dev | 19 Dec 2025 | Early Developer Release, pre-release |
  | v6.0.0-dev2 | 16 Mar 2026 | Early Developer Release 2 |
  | v6.0.0 | 4 Jun 2026 | GA |
  | v6.0.1 | 22 Jun 2026 | GA |
  | v6.1.0 | 10 Sep 2026 | GA |
  | v7.0.0a1 | 18 Sep 2026 | Alpha 1, pre-release |
  | v6.2.0 | 14 Oct 2026 | GA, marked "Latest" |

  — [Isaac Sim GitHub releases](https://github.com/isaac-sim/IsaacSim/releases)
- **Corroboration of the 2026 dates:** NVIDIA staff wrote on 26 May 2026 that the "Isaac Sim 6.0 GA requirements page will" link driver downloads, which is consistent with 6.0 GA in June 2026. — [NVIDIA forum 370054](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-shortly-after-startup-on-windows-server-2025-with-rtx-pro-6000-blackwell/370054)
- **6.0.0 GA release notes:**
  - Kit updated to 110.1.1 (from 110.0.0); `omni.physx` 110.0.7 → 110.1.11.
  - "Added experimental support for the Newton physics engine".
  - URDF importer updated: `fix_base` became tri-state, articulation roots no longer carry `PhysxArticulationAPI` (self-collision via `NewtonArticulationRootAPI`), the URDF exporter was rewritten, and `URDFImportFromROS2Node` was deprecated.
  - **Removed `omni.isaac.*` compatibility shims.**
  - Deprecated `isaacsim.core.api`, `isaacsim.core.prims` and `isaacsim.core.utils` in favour of `isaacsim.core.experimental.*`.
  - Deprecated `isaacsim.sensors.camera` and `isaacsim.sensors.rtx` in favour of `isaacsim.sensors.experimental.rtx`. Camera prims as RTX sensors via JSON configs were removed.
  - Deprecated `omni.isaac.ml_archive` (install PyTorch yourself).
  - Python binding modules moved into `bindings/` subfolders.
  - Deformable, cloth and particle APIs replaced with error stubs.
  - Multitick rendering (sensors rendered at physics-time-driven rates).
  - Windows: ROS 2 supported via Pixi.

  — [Isaac Sim 6.0.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html)
- **Per the search-engine summary of the release-notes pages** (not verified in the fetched 6.0.0 page text, which did not mention them):
  - The 6.0 Early Developer Release moved to Kit 109.0.2 and Python 3.12, with the default render mode changed to RT 2.0.
  - 6.0.0 GA introduced "Asset Structure 3.0" (geometry as binary USDC, physics and materials as USDA).
  - 6.0.1 moved to Kit 110.1.2.

  Treat these as unconfirmed. — [Isaac Sim release notes 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html), [6.0.1](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html)
- **6.2.0 release notes:**
  - Kit 110.3.0 → 110.4.0; PhysX packages 110.3.2 → 110.4.4.
  - Newton 1.6.1, with MuJoCo / MuJoCo Warp 3.12.0 and Warp 1.17.0. PhysX remains the default in Docker Compose (Newton via `ISAACSIM_PHYSICS_ENGINE=newton`).
  - URDF/MJCF importers and the URDF exporter now load from a separate `isaacsim-asset` wheel. The exporter raises `ValueError` on invalid joint limits.
  - `TiledCameraSensor` accepts `tick_rate`, schemas and attributes, and authors cameras via `RtxCamera`.
  - Cortex behaviors, examples and framework removed.
  - `get_sensor_reading()` now returns an owned snapshot (use `get_sensor_reading_view()` for live data).
  - Windows: xacro prerequisite fix, and bundled ROS 2 Humble/Jazzy builds dated 1 Oct 2026.
  - Known issue: camera gizmos can block RTX sensor rendering.

  — [Isaac Sim 6.2.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html)
- Python: 5.1 requires 3.11 ([5.1 pip](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)), 5.0 carries a Python 3.11 badge ([v5.0.0 README](https://github.com/isaac-sim/IsaacSim/blob/v5.0.0/README.md)), and 6.0 requires 3.12 ([6.0 pip](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)).
- Support status: the 5.1.0 docs carry the banner "Isaac Sim 5.1.0 is no longer supported", so bug fixes and new features come only in newer releases. — [5.1 pip page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- Windows 10 was dropped from 6.0 onward (6.0.0 requirements). — [Isaac Sim 6.0.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)

### Inferences
- **Any move off 5.x forces the same port.** Our 5.0 scripts (grip sim, synthetic generator) need it whichever 6.x we pick: `omni.isaac.*` imports become `isaacsim.*` (the shims are gone in 6.0), any camera/RTX-sensor code must be checked, and Python 3.12 applies. The cost is the same for 6.0, 6.1 or 6.2, so 5.1 buys nothing as a waypoint.
- **Expect more breaks inside 6.x.** NVIDIA drops support for old minors fast (5.1 was unsupported within about a year), so a newer minor gives a longer runway. The 6.x "experimental" core and sensor APIs are still moving, e.g. the 6.2 `get_sensor_reading` semantics change.
- **URDF assets.** Re-import our robot URDFs (M3 arm) under 6.x rather than reusing 5.0 USDs blindly. The articulation-root API and `fix_base` semantics changed, and Asset Structure 3.0 (if confirmed) changes the file layout.
- **7.0.0a1 is an alpha.** Do not standardise on it.

### Gaps
- 5.0 → 5.1 changes were not researched in detail. 5.1 is unsupported and not a candidate.
- The full 6.1.0 release notes were not retrieved.
- RT 2.0 and Asset Structure 3.0 are not confirmed from a fetched page.
- No benchmark data on tiled-camera performance changes across 5.x → 6.x was found.

## Q6. Headless / remote operation on Windows over SSH (no logged-in user), per version

### Takeaway
NVIDIA documents headless operation (`SimulationApp({"headless": True})`, `isaac-sim.streaming.bat`, `--no-window`) but **nothing about Windows OpenSSH or session-0 operation, for any version**. Our own evidence: plain-SSH headless `python.bat` runs of 5.0 work on Dionysos. On hekate, the *streaming* app has to be started from a scheduled task. Expect 6.x headless scripts to behave like 5.0, but verify with a smoke test.

### Cited Findings
- Headless scripting: create `SimulationApp` with `"headless": True`; `update()` renders a frame. — [Isaac Sim 6.2.0 Python environment](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/python_scripting/manual_standalone_python.html)
- pip headless livestream: `isaacsim isaacsim.exp.full.streaming --no-window` (WebRTC). EULA via `OMNI_KIT_ACCEPT_EULA`. — [Isaac Sim 6.0 pip install](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)
- `isaac-sim.streaming.bat` runs the headless full app with the WebRTC streaming service, accessed remotely from the Streaming Client. — [Isaac Sim 6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html)
- Isaac Lab 3.0 replaces `--headless` with `--viz` / `--visualizer` options. — [Isaac Lab release notes](https://docs.nvidia.com/isaaclab/references/release-notes)
- Kit/Windows known issue: with hardware-accelerated GPU scheduling on, "Kit Apps may freeze when using Omniverse WebRTC Streaming" (R550–R575 drivers). — [Omniverse Technical Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- **Our evidence, Dionysos (2026-10-19):** "headless Isaac Sim over plain SSH works (no scheduled task needed, unlike hekate)". The run was `C:\isaac-sim-standalone-5.0.0-windows-x86_64\python.bat run_standalone.py ...`: up in about 13 s, about 1.2 s per frame, with harmless Warp CUDA errors at start. — [project memory: dionysos-server.md](C:/Users/Mys/.claude/projects/C--Users-Mys-Desktop-RoboARM-2026/memory/dionysos-server.md)
- **Our evidence, hekate:** "Never launch Isaac Sim directly over SSH: it would die with the SSH session and has no desktop session for the GPU". The streaming app is started via `schtasks /run /tn IsaacSimStreaming`. — [project CLAUDE.md](C:/Users/Mys/Desktop/RoboARM 2026/CLAUDE.md)

### Inferences
- **6.x over SSH is very likely to work as 5.0 does.** It uses the same Kit headless mechanism, and the RTX renderer renders offscreen. Two conditions apply: set `OMNI_KIT_ACCEPT_EULA=YES` in the process (pip installs prompt otherwise), and keep runs short or detached, since an SSH disconnect kills the process tree.
- **Use scheduled tasks for anything long.** For long jobs or the streaming app, use the hekate-style scheduled task. On Dionysos, creating one counts as a persistent configuration change on a shared machine, so ask first.
- **Isaac Lab 3.0 scripts.** Update calls from `--headless` to the new `--viz` / `--visualizer` semantics (e.g. no visualizer) when porting.

### Gaps
- No NVIDIA documentation or forum thread on Windows OpenSSH, session 0 or service-account GPU rendering for any Isaac Sim version was found.
- Whether 6.x's RT 2.0 default (if confirmed) behaves differently offscreen in session 0 is unknown. This needs a smoke test.

## Q7. Recommendation for Dionysos (and hekate): what to install, how, rollback

### Takeaway
**Standardise both machines on the Isaac Sim 6.x line, with Isaac Sim 6.2.0 + Isaac Lab 3.0.0 GA (Python 3.12, installed via uv/pip venvs) as the target.**
- **Why 6.2 + Lab 3.0.0.** It is the only Isaac Lab GA release for 6.x, NVIDIA calls it the baseline for new projects, and 5.x is unsupported.
- **The blocker.** Dionysos must move from driver 581.42 to ≥595.97 (R595), a shared-machine change that needs the lab's agreement and a 5.0 regression test.
- **Until then.** Install Isaac Sim **6.0.1** standalone on Dionysos. It runs on 581.42 (the 6.0.0 tested driver) and matches hekate's 6.0 line. Use it for plain Isaac Sim scripts (grip sim, synthetic generator).
- Leave 5.0 untouched throughout.
- Do not pick 5.0/5.1: unsupported, Python 3.11, no Newton, and the `omni.isaac` → `isaacsim` port is unavoidable later anyway. Do not pick 7.0 alpha.

### Cited Findings
- 6.0.0 tested on Windows 581.42 ([6.0.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)); 6.1/6.2 tested on 595.97 ([6.1.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html), [latest](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)); Isaac Lab 3.0.0 minimum Windows driver 595.97 and "Isaac Sim 5.1 and older are unsupported" ([Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)).
- Isaac Lab 3.0.0 (20 Oct 2026) is the "recommended baseline for new projects and migrations from Isaac Lab 2.x"; beta2.patch1 targets 6.0.1. — [Isaac Lab GitHub releases](https://github.com/isaac-sim/IsaacLab/releases)
- 595.97 is an Omniverse-validated R595 Windows Workstation driver. The documented R595 problem is on Blackwell GPUs, not Ampere. — [Omniverse Technical Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html), [NVIDIA forum 370054](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-shortly-after-startup-on-windows-server-2025-with-rtx-pro-6000-blackwell/370054)
- Isolation tools: `--portable-root`, per-version data dirs, do not rename installs, and `ISAACSIM_ASSET_ROOT` takes precedence. — [6.0.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_faq.html), [forum 339998](https://forums.developer.nvidia.com/t/installing-multiple-versions-of-isaacsim-without-the-depreciated-launcher/339998)
- Mimic is Linux-only, so it plays no part in the Windows plan. — [Isaac Lab Mimic page](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html)

### Inferences (proposed step-by-step plan; practice-based where no doc exists)

**Phase 0: inventory (read-only, no changes)**
1. Check free VRAM (`nvidia-smi`) and who is using the GPU.
2. Record the driver (581.42).
3. Dump machine and user env vars and look for `ISAAC_PATH`, `CARB_APP_PATH`, `EXP_PATH`, `ISAACSIM_ASSET_ROOT`, `PYTHONPATH`.
4. Check `LongPathsEnabled` in HKLM.
5. Read `C:\isaac-sim-standalone-5.0.0-windows-x86_64\setup_python_env.bat` and `clear_caches.bat` to confirm what they touch.
6. Run the current 5.0 smoke test (`run_standalone.py` with 2–3 frames) and keep its log as the baseline.

**Phase 1: Isaac Sim 6.0.1 standalone on Dionysos (no driver change)**
1. Download the official 6.0.1 Windows standalone zip and extract it directly to `C:\isaac-sim-standalone-6.0.1-windows-x86_64`. Never rename it afterwards.
2. Run its post-install script if the package has one.
3. Launch only through its own `python.bat`, with `OMNI_KIT_ACCEPT_EULA=YES` set in the process.
4. Optionally add `--portable-root C:\Users\Hercules\isaac-data\6.0.1` for full cache/data/log isolation.
5. Port the scripts to `isaacsim.*` imports and re-import the URDF. Smoke-test headless over SSH.
6. Re-run the 5.0 baseline and compare.
7. Rollback: delete the 6.0.1 folder and its `%LOCALAPPDATA%\ov\data\Kit\Isaac-Sim\6.0` subfolder (or the portable root), and its log folder. 5.0 is never touched.

**Phase 2: driver upgrade on Dionysos (gated)**
1. Agree with the lab and the machine's other users on a window.
2. Download and keep the 581.42 RTX Enterprise installer for rollback.
3. Install the R595 595.97 Workstation driver with no clean-install wipe of other settings.
4. Re-run the 5.0 baseline and the 6.0.1 smoke test.
5. Rollback: reinstall 581.42, or use Device Manager "Roll back driver".

**Phase 3: Isaac Sim 6.2.0 + Isaac Lab 3.0.0 (needs Phase 2 on Dionysos)**
1. In a user-owned folder, clone Isaac Lab 3.0.0 and run `uv sync --extra isaacsim` (Python 3.12, `isaacsim[all,extscache]==6.2.0.0`, torch 2.11.0 cu130). This needs long paths enabled, which is machine-wide, so ask if it is off.
2. It lives entirely in a venv; deleting it is the rollback.
3. On hekate (617.14 already meets the minimum), do the same in a new venv next to the existing 6.0.0 standalone. Per the project CLAUDE.md, a software change on hekate needs the owner's go-ahead.
4. Retire Isaac Lab 3.0.0-beta2.patch1 once the expert pipeline runs on 3.0.0.

**If the driver upgrade is refused**
1. Standardise on **6.0.1 on both machines**: upgrade hekate 6.0.0 → 6.0.1, which also matches beta2.patch1.
2. Use Isaac Lab 3.0.0-beta2.patch1 only for experiments, knowing its API differs from 3.0.0 GA.
3. Keep Isaac Lab out of the critical path. The scripted expert, LeRobot and ACT pipeline does not require it.

**Other risks**
- 6.2 and Lab 3.0.0 are 12 and 6 days old, so expect early bugs. Pin exact versions in the uv lock.
- The A4500 (Ampere, below the RTX 4080 minimum tier) will limit `num_envs` with cameras compared with hekate's 5090. Plan for hekate to do the heavy parallel-camera rollouts and Dionysos the lighter jobs.

### Gaps
- Running Isaac Sim 6.2 or Isaac Lab 3.0.0 on driver 581.42 is untested and not documented. Isaac Lab says 595.97 is the minimum, and whether that is enforced at runtime is unknown.
- No source confirms Isaac Sim 5.0 keeps working after an R595 driver upgrade on Ampere. Phase 2 must test it.
- The current RTX Enterprise (A4500) driver releases in the R580 and R595 branches, and their exact version numbers, were not checked on nvidia.com.
- The exact names and sizes of the 6.0.1 / 6.2.0 Windows standalone zips and whether they include `post_install.bat` were not verified.
