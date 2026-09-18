#!/usr/bin/env python3
"""Read-only preflight: is every subsystem actually talking to us?

Sends no motion commands. Run this first, every session, before anything else.
Exits non-zero if any check fails, so it can gate a launch script.

    docker compose run --rm core python tools/check_hardware.py
"""

import sys
import time

import cv2

from roboarm import config as cfg
from roboarm.arm import read_pose

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


def check_arm() -> None:
    """Open the board and read it. Never commands a move."""
    try:
        from Rosmaster_Lib import Rosmaster
    except ImportError as exc:
        record("arm sdk", FAIL, f"vendored SDK not importable ({exc}); run tools/vendor_sdk.sh")
        return

    try:
        bot = Rosmaster(com=cfg.SERIAL_PORT)
    except Exception as exc:  # noqa: BLE001 - pyserial raises several unrelated types here
        record("serial port", FAIL, f"cannot open {cfg.SERIAL_PORT}: {exc}")
        return

    bot.create_receive_threading()
    time.sleep(1.0)  # the board streams asynchronously; give it a beat

    version = bot.get_version()
    battery = bot.get_battery_voltage()
    angles = bot.get_uart_servo_angle_array()

    # The board reports -1 / 0.0V for everything when the port opened but no data
    # arrives. On this robot that almost always means Yahboom's factory web app
    # (rosmaster_main.py) still holds /dev/ttyUSB0 -- not broken hardware.
    if version == -1 and battery == 0.0 and all(a == -1 for a in angles):
        record(
            "serial port",
            FAIL,
            "port opened but the board sent nothing. Something else is holding it -- "
            "on the host run:  pkill -f 'rosmaster_mai[n]'",
        )
        del bot
        return

    record("serial port", PASS, cfg.SERIAL_PORT)
    record("firmware", PASS, f"v{version}")

    if battery >= cfg.MIN_BATTERY_V:
        record("battery", PASS, f"{battery:.1f} V")
    else:
        record("battery", WARN, f"{battery:.1f} V -- below {cfg.MIN_BATTERY_V} V, charge before demoing")

    # Shared with arm.py: batch read, then a spaced direct query for whatever the
    # batch call drops. Neither reader is reliable alone -- J4 was invisible to the
    # batch call for a whole session while answering direct queries perfectly.
    pose = read_pose(bot)
    silent = [j for j, v in pose.items() if v is None]
    live = {j: v for j, v in pose.items() if v is not None}
    out_of_range = {
        j: v for j, v in live.items() if not (cfg.HARD_LIMITS[j][0] <= v <= cfg.HARD_LIMITS[j][1])
    }

    if silent:
        record(
            "arm readback",
            FAIL,
            f"servo {', '.join(str(s) for s in silent)} answered neither the batch read "
            f"nor a direct query. Check the connector.",
        )
    else:
        record(
            "arm readback",
            PASS,
            " ".join(f"J{j}={live[j]}" + ("!" if j in out_of_range else "") for j in cfg.JOINT_IDS),
        )

    if out_of_range:
        record(
            "joint range",
            WARN,
            f"{', '.join(f'J{j}={v}' for j, v in out_of_range.items())} outside limits -- "
            f"the servo is fine; engage torque to pull it back in "
            f"(tools/arm_release.py --engage)",
        )

    roll, pitch, yaw = bot.get_imu_attitude_data()
    if (roll, pitch, yaw) == (0.0, 0.0, 0.0):
        record("imu", WARN, "reads exactly zero -- suspicious, check it is level")
    else:
        record("imu", PASS, f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}")

    del bot


def check_camera(index: int, name: str, size: tuple[int, int]) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        record(name, FAIL, f"/dev/video{index} will not open (held by another process?)")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    frames, start = 0, time.time()
    frame = None
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            frames, frame = frames + 1, f
    cap.release()

    if frames == 0:
        record(name, FAIL, f"/dev/video{index} opened but returned no frames")
    else:
        fps = frames / (time.time() - start)
        record(name, PASS, f"{frame.shape[1]}x{frame.shape[0]} @ ~{fps:.0f} fps")


def check_aruco() -> None:
    """Generate a marker and detect it back -- proves contrib is really present."""
    try:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.generateImageMarker(d, 7, 200)
        img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        _, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(img)
        if ids is not None and 7 in ids.ravel():
            record("aruco", PASS, "generate/detect round-trip")
        else:
            record("aruco", FAIL, "generated marker did not detect back")
    except AttributeError as exc:
        record("aruco", FAIL, f"opencv built without contrib: {exc}")


def main() -> int:
    check_arm()
    check_camera(cfg.WRIST_CAM, "wrist camera", cfg.WRIST_CAM_SIZE)
    check_aruco()

    width = max(len(n) for n, _, _ in results)
    print()
    for name, status, detail in results:
        mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")

    failed = sum(1 for _, s, _ in results if s == FAIL)
    print()
    print(f"{len(results) - failed}/{len(results)} checks passed" if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
