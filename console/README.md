# The console -- both robots on one page

One page on this laptop for the X3 Plus and the M3 Pro: each robot alone or both side by side,
a STOP on each and STOP ALL always on screen (Space or Esc too). Plan v2.3, M1, first version.

```bash
.venv/Scripts/python console/serve.py --fake    # pretend robots, no hardware
.venv/Scripts/python console/serve.py           # the real ones (robots.json)
```

then open http://localhost:8099/ (`.claude/launch.json` has both as `console-fake` and `console`).

- **The robots** are listed in `robots.json`: the X3's panel running on this laptop
  (`tools/webapp.py --robot http://192.168.73.210:8761 --port 8091`, launch config
  `webapp-robot`) and the M3's drive page (`m3pro/tools/arena_map.py start`, port 8095).
  `robots.fake.json` points at `tools/webapp.py --sim` and `m3pro/arena/fake_m3.py`, which
  `--fake` starts: the M3's real drive logic on the saved arena map, with a ray-cast lidar.
- **X3 card:** the wrist camera, arm / job / camera state, Pick one, Clear the table, Home;
  everything else on its own page (the "own page" link).
- **M3 card:** the live map with the fence fit, the robot, its scan and the loop; hold the pad
  or W/A/S/D (sideways on A/D), Q/E to turn; the auto loop. The robot's own limits apply: at
  most 0.15 m/s, stopped 0.4 s after the page stops repeating a command; the loop also stops
  35 cm short of anything in its path, driving by hand has no distance limit.
- **STOP ALL** lets go of any held key, stops every robot in `robots.json` -- shown or not --
  and logs how long each took to confirm (pretend robots: M3 20 ms, X3 about 0.8 s).
- **3D scan:** the M3's coloured 3D scans of the arena, as `m3pro/tools/arena_map.py
  rtab-export` leaves them in `m3pro/out/arena/rtab/` (on this laptop, so no robot needed); pick
  one from the list, or open or drop any `.ply`. Drag turns, right-drag or Shift-drag moves,
  the wheel (or a pinch) zooms, a double-click or "reset view" goes back to the view from
  above. Camera colour or height colour (0 to 45 cm above the floor, blue to red), and the point
  size (a 7 MB export, 463k points, is on screen in 0.1-0.2 s). "Floor print" shows the
  export's floor print instead (`NAME_floor.png`: the floor from above, 1 cm a pixel), turned a
  quarter when it stands upright so that it lies as the 3D view starts, the same way round.
- **Pop out** opens one robot, or the 3D scan, alone in its own window (`/?solo=m3`,
  `/?solo=cloud`), e.g. for a second monitor; the layout is remembered in the browser.

How it is built: `serve.py` (standard library) serves the page and passes JSON through as
`/robot/<id>/<path>` -- the X3's panel sends no CORS headers; camera streams and maps come
straight from the robots. The scans come from `serve.py` too: `/clouds.json` lists the PLY
files in the export folder, each with its floor print, and `/clouds/<name>` is one of those
(that folder, `.ply` and `.png` only).
`cloud.js` draws them with WebGL2 and no library (binary little-endian or ASCII PLY, x y z
and red green blue; other properties skipped), loaded only when the 3D view opens. No
framework, no build step. It listens on localhost only (IPv4 and IPv6: on Windows `localhost`
is tried as `::1` first, and without it every connection waited 2 s). Not yet: dockview panels
(drag, split, float), the arena panel with both robots on the mat, SSE instead of polling,
voice.

`python -m pytest console` runs `test_serve.py`.
