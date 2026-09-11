#!/usr/bin/env python3
"""Log every angle array we hand the SDK, and flag the ones it will silently drop.

    docker compose run --rm core python tools/trace_commands.py

set_uart_servo_angle_array() validates its input and, if anything is out of range,
prints "angle_s input error!" and returns having done NOTHING. The move still looks
like it worked because the next command usually lands. This reproduces the sequence
that produced those errors and shows exactly which value the SDK objected to.
"""

import sys

from roboarm import config as cfg
from roboarm import kinematics as kin
from roboarm.arm import Arm, ArmError

# What the SDK itself accepts -- transcribed from the vendored source.
SDK_RANGE = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270), 6: (0, 180)}


def main() -> int:
    rejected: list[tuple[list[int], list[str]]] = []
    sent = 0

    try:
        with Arm() as arm:
            original = arm.bot.set_uart_servo_angle_array

            def traced(angle_s, run_time=500):
                nonlocal sent
                sent += 1
                bad = [
                    f"J{j}={angle_s[j - 1]} outside {SDK_RANGE[j]}"
                    for j in cfg.JOINT_IDS
                    if not (SDK_RANGE[j][0] <= angle_s[j - 1] <= SDK_RANGE[j][1])
                ]
                if bad:
                    rejected.append((list(angle_s), bad))
                return original(angle_s, run_time)

            arm.bot.set_uart_servo_angle_array = traced

            # The exact sequence that produced the errors: a wide swing, then a
            # top-down solve.
            arm.move_to({1: 140, 2: 35, 3: 60, 4: 120, 5: 90, 6: cfg.GRIPPER_OPEN},
                        speed_dps=20, verify=False)
            pose, _pitch = kin.solve(0.170, 0.0, -cfg.TABLE_BELOW_PLATE + 0.030)
            pose[6] = cfg.GRIPPER_OPEN
            print("IK produced:", {j: pose[j] for j in cfg.JOINT_IDS})
            arm.move_to(pose, speed_dps=15, verify=False)

            print(f"\n{sent} commands sent, {len(rejected)} would be REJECTED by the SDK")
            for angles, why in rejected[:10]:
                print(f"  {angles}  ->  {', '.join(why)}")
    except ArmError as exc:
        print(f"\nARM ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
