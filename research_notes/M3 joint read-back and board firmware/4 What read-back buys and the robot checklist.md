# 4. What joint read-back would buy, and a read-only checklist for the robot

Scope: what the M3's servo read-back would and would not measure (resolution, accuracy, latency),
how we would use it (the policy's state, sag and backlash, stall and grasp detection, teaching by
hand, the robot card), with numbers from the X3, which has read-back, and from the M3's camera
probes; then the read-only checklist for a later look at the robot. Labels as in note 1.

## 1. Resolution is 0.08 deg; accuracy is unmeasured; the X3's read-back is good to ~1-2 deg

### Takeaway
A pulse is 0.082 deg on Yahboom's map, so resolution is not the limit. No independent measurement
of potentiometer read-back noise or linearity was found for Yahboom's (or Hiwonder's, or Feetech's
SCS) servos. Our X3 is the best evidence: its batch read in whole degrees repeats to ~1 deg, the raw
count read places the camera to ~1 deg / 2.5-3 mm through FK, and it plainly shows gravity droop and
the side a joint came from.

Evidence:
- est.: 2200 pulses / 180 deg = 12.2 pulses a degree, 0.082 deg a pulse (map in note 1, section 2).
- Vendors: Hiwonder LX-16A "Servo Accuracy: 0.3 degrees" with no method; Yahboom's 15 kg servo
  "Servo Accuracy: 1" with no unit ([search summary](https://ca.robotshop.com/products/hiwonder-hiwonder-lx-16a-full-metal-gear-serial-bus-servo-with-real-time-feedback-function-rc-robot-control-angle-240);
  [hardware_precision_limits.md](../Presný%20tréning%20VLA%20pre%20lacné%20rameno/hardware_precision_limits.md), line 37) [2nd].
  Searched: potentiometer bus-servo feedback noise for LX-16A and SCS15 -- vendor specs only.
- X3: "`get_uart_servo_angle_array()` gives position readback (+-1-2 deg)"; worst-case spread 1 deg
  per joint over 10 home/start round trips; `read_fine()` reads each raw count to 0.08 deg, 50 ms
  per joint; with it FK placed the camera to 8.5 px (~3 mm), and a gravity-sag term (J2/J3 2-3.5
  deg at full lever) brought it to 6.0 px (`x3plus/PLAN.md` X2.1; project memory) [OURS].

## 2. Read-back sees droop, backlash side, lag and stalls -- not a wrong pulse-to-degree scale

### Takeaway
A servo closes its loop on its own potentiometer, so read-back reports what the servo believes:
load droop, which side of the backlash it stopped on, how far it trails the command, and that it
stopped. It cannot see an error in the pulse-to-degree map, link flex or the camera mount. One such
scale error may already be visible: the ~10 % "gain" the M3's camera probe measured is what Yahboom's
map (12.2 pulses/deg) gives on a servo of ~13 pulses/deg (96-4000 over 300 deg, the servo's sheet) --
a hypothesis, which read-back would not show and the camera would.

Evidence:
- X3, read back: J2 droops -6 deg at J2=120 and "saturates near 121 deg actual regardless of command
  (130/140/150 all give 121)"; "J3 read 22 from below, 24 from above, same command"
  (project memory rosmaster-sdk-readback-gotcha; `x3plus/docs/STATUS.md`) [OURS].
- M3, by camera (`m3pro/tools/m3_servo_probe.py`, NEXT.md 2026-10-23): backlash 0.36-1.09 deg, repeat
  sd 0.01-0.17 deg, static gain 0.91-0.94 (J1), 0.92 (J2), 0.91 (J4), 1.06-1.15 (J3, pose dependent),
  lag 100-195 ms, ringing up to ~1 s on J2/J3 after fast moves [OURS].
- est. (the gain hypothesis): commanded 1 deg = 12.22 pulses. At 3904 pulses / 300 deg (13.01 per
  deg) that turns 0.94 deg; at 4096 / 300 (13.65) 0.90 deg. The measured 0.91-0.94 on the unloaded
  joints sits in that band. A raw-count read of a joint swept against the camera would settle it.

## 3. How we would use it

### Takeaway
In order of value: (1) stall, collision and grasp detection, including a gripper that stops closing
when its fingers stop moving instead of burning; (2) sag and backlash measured at every pose without
the ChArUco board; (3) teaching by hand with the torque off (kinesthetic demonstrations and the taught
aim); (4) a measured robot card for the simulator; (5) the policy's state. The last is worth less
than it looks: a measured state still follows the last command 0.1-0.2 s later, so it does not cure
the copycat problem by itself; it helps where the arm did not follow.

Evidence:
- Policy: our ACT's `observation.state` equals the previous action to the bit; ALOHA's ACT fed the
  follower's measured joints; the largest published fix is dropping the state input
  ([Why our ACT does not look](../../reports/Why%20our%20ACT%20does%20not%20look.md)) [OURS, 2nd there].
- Stall rule (est.): with lag 0.1-0.2 s and backlash up to ~1 deg, flag a joint when |command
  interpolated to now - 0.15 s - measured| > 3 deg for > 0.3 s; on the gripper, close in steps and
  stop when two successive reads differ by < 1 pulse-degree while still short of the target -- the
  stop angle gives the held object's width. Yahboom warns the gripper "may cause the servo to stall
  and burn" ([Precautions for DIY Robotic Arm](https://www.yahboom.net/study/ROSMASTER-M3PRO/17534)) [DOC];
  the M3 now judges a grasp by looking again (`m3arm.py`) [OURS].
- Teaching by hand: Yahboom's X3 "memory learning" (torque off, read, record)
  ([8362](https://www.yahboom.net/study/ROSMASTER-X3-PLUS/8362)) [DOC]; our M3 aim is taught by
  keyboard because the arm cannot go limp (project memory m3pro-taught-aim) [OURS]. Note 2 section 1:
  the factory protocol can make it limp.
- Load, temperature, voltage: if the servos implement the SCS table (addresses 60, 62, 63), load
  gives a second stall signal and temperature a duty limit (note 1, section 2) -- unverified.
- Latency (est.): a firmware publisher's sample is at most one sweep (~12 ms) old plus <1 ms of
  transfer; a bus tap synced to 10 Hz commands gives one set per cycle ~30-45 ms after the command;
  the X3's host polls take 1-30 ms per request and need 50 ms spacing (note 2, section 5).

## 4. Read-only checklist for the robot (for a later session; not run)

### Takeaway
Three tiers. Tier A changes nothing and keeps the robot running: topics, the two Yahboom scripts and
their library read with `cat`/`grep` (not run, not imported), the serial device, the agent, the
scan fields, photos of the board and a servo label. Tier B stops the agent (ask first; the board may
need a power cycle afterwards). Tier C attaches an adapter that only listens on the servo bus, plus
one optional harmless query on the board's UART6 (ask first; hardware).

Tier A -- over SSH, nothing stopped, nothing written:
1. `ros2 node info /YB_Node` and `ros2 topic list -t | grep -i -E "joint|servo|arm"` -- confirm no
   servo state is published; `ros2 interface show arm_msgs/msg/ArmJoints` and `.../ArmJoint`.
2. `ls -la ~/*.py ~/*.sh`; `cat ~/start_agent.sh`; `sed -n '1,400p' ~/config_robot.py` and
   `~/calibrate_arm.py`; `grep -n -E "^(import|from)|def |FUNC_|0x[0-9A-Fa-f]{2}|serial|baud" ~/config_robot.py ~/calibrate_arm.py`
   -- the frame format (`FF FC`?), the baud rate, and whether a servo read (`FUNC_UART_SERVO` /
   `FUNC_ARM_CTRL` request, `get_uart_servo_*`) exists; whether the file only reads when every
   `set_` line is commented.
3. Find the library those scripts import and read it the same way (`grep -rn "def get_\|def set_uart" <path>`);
   do not `import` it (that opens the port).
4. `find ~ -maxdepth 4 \( -iname "*.hex" -o -iname "*FW*" -o -iname "Board_Samples" \) 2>/dev/null | head`
   -- a firmware image or the sample sources already on the robot.
5. `ls -l /dev/myserial /dev/serial/by-id/`; `udevadm info -q property -n /dev/myserial | grep -E "ID_VENDOR|ID_MODEL|ID_SERIAL"`
   (expect a CP2104); `dmesg | grep -i cp210`.
6. `ps -ef | grep -i [m]icro_ros` (agent arguments, baud 2000000); `docker ps` if it runs in Docker;
   `ros2 pkg xml micro_ros_agent 2>/dev/null | grep -i version`.
7. `ros2 topic echo --once /scan0 --field time_increment`, `... --field scan_time`, `... --field header`;
   `ros2 topic hz /scan0`; `ros2 topic delay /scan0` and `/imu/data_raw`; `cat /proc/uptime` --
   the stamp offset at this uptime (drift check).
8. `grep -rn -E "GetCurrentArmJointAngles|def _get_joints|Curjoints|joint_states" ~/M3Pro_ws/src ~/yahboomcar_ws/src --include=*.py | head -30`
   -- confirm Yahboom's "current angles" are a record of commands.
9. Photos: the board's MCU marking (STM32H743V?T6 -- settles 1 MB or 2 MB), the S3/S4/S5, UART6,
   SWD, CAN headers, BOOT0/RESET; the label on one arm servo (model); the connector pinout at S3.

Tier B -- agent stopped (ask the user; expect a possible power cycle, 2026-10-15):
10. With every `set_` line commented (checked in step 2): `python3 ~/config_robot.py` -- car type,
    firmware version, domain ID, scales, PIDs, `arm_mid`. Write them down before any flashing.
11. Optional, only with the user: a backup read of the flash with STM32CubeProgrammer through the
    bootloader (BOOT0 + RESET, Type-C to the laptop): it either reads the image or reports
    read protection; it writes nothing. Device ID tells the flash size.

Tier C -- a listener on the servo bus, one query on UART6 (ask; hardware):
12. Measure the bus pins first (GND, 6.8 V, signal; signal idle level 3.3 or 5 V). Connect only GND
    and the adapter's RX to the signal. Log at 115200 8N1 while the bridge sends a few moves: servo
    IDs, single writes (`0x03`) or sync writes (`0x83`), whether the board ever reads (`0x02`), and
    the silent gaps.
13. Optional, same tier: the version request on the board's UART6 (note 2 section 2), with a 3.3 V
    adapter, TX/RX/GND only.
