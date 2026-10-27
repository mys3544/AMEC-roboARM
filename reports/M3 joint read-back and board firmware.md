# The M3's servos can already report their angles: read them from the bus between our commands, and rebuild the board's firmware only with Yahboom's source

**Bottom line.** The M3 Pro has one control board, an STM32H743 (Cortex-M7) running micro-ROS
Humble, and its six arm servos hang directly on the MCU's USART3. Yahboom's own STM32 code talks to
those servos in a Feetech-SCS-style protocol that includes a position read (`FF FF id 04 02 38 02`),
and the factory firmware evidently uses it: with the micro-ROS agent stopped, Yahboom's
`calibrate_arm.py` switches the servos limp, and the board then stores each joint's mid value as a
pulse count (2025, 2077, 1957, ... on Yahboom's unit) that can only come from reading the servo.
What the firmware does not do is publish positions over ROS. Its source is not published -- only a hex file
(`microROS_STM32-FW_V1.1.3.hex`) and the course's sample projects -- so "add a joint publisher" means
rebuilding the whole base firmware from samples, which would also re-implement the motor PIDs,
odometry, IMU and lidars we have calibrated against. The safe order is therefore: read the robot
(a read-only checklist, below); use Yahboom's existing protocol with the agent stopped for
torque-off teaching and calibration reads, if its scripts show a read call; listen on the servo bus
to see when the board talks; then let our bridge read the servos itself in the ~80 ms of silence
after each 10 Hz command. Firmware is worth doing only together with the lidar fix (per-point time,
whole revolutions, periodic clock sync -- the parts no host-side re-stamper can do), and only if
Yahboom shares its source. Flashing is recoverable (ROM bootloader, vendor hex), but the arm's
calibration and the ROS settings live in the board's flash and must be written down first.

Labels: [DOC] vendor document, [CODE] source read, [CLAIM] unmeasured statement, [2nd] secondary,
[OURS] our robots' records, "est." our arithmetic. Details and every link are in the notes:
`research_notes/M3 joint read-back and board firmware/` (1 the board and firmware, 2 reading
without new firmware, 3 lidar stamps, 4 what read-back buys + checklist). Nothing was run on the
robot.

## 1. The board is a single STM32H743 micro-ROS board, and its arm servos answer position reads

Yahboom's control-board course names the chip (STM32H743VGT6), the micro-ROS build (branch
`humble`, a custom serial transport on USART1 at 2 Mbaud through an on-board CP2104, FreeRTOS) and
the pins: bus servo on PD8/PD9 (USART3), the lidars on UART4 and UART5, a debug log on UART7, a
"control interface" on UART6, FDCAN, SWD, BOOT0 and RESET buttons [DOC]. There are three bus-servo
ports; "the M3PRO's robotic arm bus servo is 6.8V", so the arm uses S3 [DOC]. No separate servo
controller exists, and no direction pin is listed for the servo line, so the board joins TX and RX
onto the one wire in hardware (the sibling board's schematic uses a 74HC126 tri-state buffer with
1 kOhm series resistors and a 4.7 kOhm pull-up to 5 V) [DOC, sibling board].

The servo protocol is visible in Yahboom's STM32 course for its other robots: `UartServo_Ctrl()`
writes `FF FF id 07 03 2A posH posL timeH timeL chk`, `UartServo_Get_Angle()` sends
`FF FF id 04 02 38 02 chk`, and the parser reads a big-endian 16-bit position out of an 8-byte reply
[CODE, vendor sample]. These are Feetech SCS packets: WRITE at address 42 (goal position) and READ
at address 56 (present position); Feetech's SCS table continues with speed (58), load (60), voltage
(62), temperature (63) and current (69) [CODE, Feetech library]. Whether Yahboom's servos fill those
other registers is unverified. The M3's stored arm mid values, `[2000, 2000, 2000, 2000, 1486, 3100]`,
are exactly the X3's Rosmaster_Lib map (90 deg = 2000 for J1-J4, 1486 for J5, 3100 = gripper shut)
[DOC, CODE], so the M3's arm uses the same Yahboom servo family as the X3's, which reads back fine.
The M3's servo model itself is not named anywhere we looked.

Two notes in our own files need correcting. `m3arm.py` says "there is no torque switch"; the factory
firmware has one, reachable only with the agent stopped (`calibrate_arm.py` "unloads" the servos)
[DOC]. Yahboom's FAQ says the arm is "torque-free upon power-up"; ours came up stiff, most likely
because the gamepad node sends the ready pose when it starts [DOC, OURS].

## 2. The factory firmware is a closed hex: "add a publisher" means rebuilding the base firmware

Yahboom's download drive holds the factory firmware as `microROS_STM32-FW_V1.1.3.hex` (971 kB,
28 Aug 2025), next to the CP2104 driver, STM32CubeIDE and STM32CubeProgrammer [DOC]. The GitHub
repository holds 288 course PDFs and no firmware or robot source; so do the repositories of the
sibling robots [CODE: trees].
The course's micro-ROS and STM32 samples are source (`Board_Samples/...`) and cover every piece --
odometry, IMU, lidar, the arm subscriber, flash storage, PID -- but not the factory firmware's
non-ROS protocol that `config_robot.py` and `calibrate_arm.py` use, nor its integration [DOC].
A 1.21 GB "Code" zip on the same drive may contain more; it was not downloaded (a download needs the
user's approval).

If the source were in hand, a joint publisher is a small change with a clear bus budget (est.): one
READ is 16 bytes on a 115200-baud wire, ~1.4 ms plus the servo's reply delay, so six joints take
~9-12 ms; a six-joint command is 3.3 ms (sync write) to 5.7 ms (six single writes). Sweeping all six
at 20 Hz while commanding at 10 Hz uses ~25 % of the bus; 50 Hz would use ~57 % and only works if
commands always pre-empt reads. Over micro-ROS a compact six-joint message is ~0.3 ms at 2 Mbaud,
and the firmware has room for a sixth publisher if it was built with the course's limits (10
publishers, 5 used) [DOC, est.]. Husarion's ROSbot firmware is a public example of the same pattern
(wheel `JointState` stamped with `rmw_uros_epoch_nanos()`) [CODE].

Without the source, the alternative is a replacement firmware assembled from the samples. It would
behave differently in ways that matter to us -- motor PID, odometry integration, IMU yaw hold, the
lidar packing, the stored parameters -- and every base calibration (lidar biases, the odometry
error study, the EKF) would need re-checking. That is weeks of work for a result the bus tap of
section 3 can approach without touching the board.

Flashing itself is the easy part. The ROM bootloader is entered by holding BOOT0 and pressing RESET;
STM32CubeProgrammer writes the hex over the Type-C port (unplugged from the Jetson and plugged into
a computer), or an ST-Link writes over SWD [DOC]. The ROM bootloader cannot be erased, so a bad image
is always recoverable with Yahboom's hex. The traps are elsewhere: the board keeps the arm's mid
values, the ROS domain ID and namespace, the motor and yaw PIDs and the velocity scales in its own
flash, so a full erase loses the arm's calibration until `set_arm_mid_value` restores it [DOC]; and
if the chip is read-protected (RDP level 1), the current image cannot be backed up and lowering the
protection erases it [2nd]. A backup read is harmless to try: it works or it says no. The option
bytes (RDP, boot address) must never be touched.

## 3. Three ways in without reflashing, safest first

**Yahboom's own protocol, agent stopped.** With the agent closed, `calibrate_arm.py` prints the board
version, makes the servos limp and stores each one's present position, and `config_robot.py` reads
back car type, firmware version, domain ID, scales, PIDs and `arm_mid` [DOC]. These match the X3's
Rosmaster_Lib calls (`get_version`, `set_uart_servo_torque`, `set_uart_servo_offset` with an
`(id, state)` reply), whose library also has `get_uart_servo_value(id)` and
`get_uart_servo_angle_array()` [CODE]. If the M3 scripts (or the library they import) contain that
read, an "arm-only session" -- agent stopped, base silent -- gives torque-off teaching and full
joint reads with no hardware and no flashing. It cannot serve the policy at run time: micro-ROS and
this protocol share the one USB port. Each agent stop risks the board hang we saw on 2026-10-15
(power cycle needed) [OURS].

**The board's second serial port.** UART6 is documented as a "control interface ... send protocol
to control the robot", and CAN likewise [DOC]. No chapter says what the factory firmware runs there.
If it answers Yahboom's protocol while micro-ROS runs on USART1, a 3.3 V USB-TTL adapter from the
Jetson gives joint reads through the board with nothing on the servo bus. One version request
(a read; moves nothing) settles it.

**A second reader on the servo bus.** First as a listener only (adapter RX and GND on the signal
line): it shows the servo IDs, whether the board sends single or sync writes, whether it ever reads,
and how long the bus is silent. Then as a second master that sends READ requests only in silence.
Because the board writes only when we publish an arm command (to be confirmed by listening), our
bridge owns the schedule: publish the 10 Hz command, wait ~30 ms (the board's ROS loop spins every
10 ms, then 3-6 ms on the wire), read six joints in ~12 ms, and stay quiet until the next command --
10 Hz joint states in step with the policy, from a free connector on the gripper servo or a splitter
at S3. The adapter must never drive the line high (open-drain or tri-state output, or a diode on
TX) and must never touch the 6.8 V pin. A collision could garble a board WRITE into another valid
WRITE (an 8-bit checksum passes ~1 in 256 corrupted frames), so the reader also compares its own
echo with what it sent and, on a mismatch, stops and re-sends the last pose (est.). Yahboom's gamepad
node must not publish arm commands while we read.

The X3 shows the approach works and what to design around: its board polls servos on request,
reads drop out after motion, two reads need ~50 ms between them, and a host read once held the
command stream's line for 50-300 ms [OURS]. On the M3 the reader has its own adapter, so it cannot
block the micro-ROS link; it can only collide on the servo wire, which the schedule prevents.

## 4. Only firmware gives the lidar per-point time; the host can fix the clock and the mean lag

The same firmware stamps both lidars. Yahboom's sample keeps a rolling array of angle bins, copies
it out on a 140 ms timer and stamps it at sending, leaving `time_increment` and `scan_time` at zero
[CODE, vendor sample]; our scans behave that way (7.1 Hz, 500 points, stamps 55-78 ms ahead of the
Jetson, 1.1 s after 23 h) [OURS]. Three faults follow: the clock offset and its drift, a stamp ~83 ms
after the points' mean time, and points spanning one revolution with a moving seam.

A host re-stamper (designed in the Base localization report) removes the offset to a few ms and
shifts by a measured mean lag -- the two largest terms -- for all four board topics, with no risk.
The rest needs the firmware: stamp each lidar packet when it arrives (a microsecond timer converted
to the micro-ROS epoch), start each scan at the lidar's ring-start packet (`CT_RingStart = 1` in
YDLIDAR's protocol), publish whole revolutions with `time_increment` = 1/4000 s and the measured
`scan_time`, and call `rmw_uros_sync_session()` every 10-30 s instead of once [CODE: YDLIDAR SDK,
micro-ROS `time_sync.h`]. YDLIDAR's own host driver dates points exactly this way, from arrival time
minus bytes waiting and point periods [CODE]. Husarion's firmware, like Yahboom's, syncs only once
at connect [CODE]. An untested host-side idea may recover part of the per-point time without
firmware: about 16 % of a revolution is not refreshed between two published scans, so a run of bins
identical to the previous scan in both range and intensity marks where the write head was.

## 5. What read-back buys, in numbers

Resolution is not the issue: a pulse is 0.082 deg. Accuracy of the potentiometer is unmeasured by
anyone we found (vendors quote "0.3 degrees" or "1" without a unit) [2nd]. Our X3 is the evidence:
whole-degree batch reads repeat to ~1 deg; raw-count reads place the camera through FK to ~3 mm,
6.0 px with a sag term; read-back shows J2 drooping 6 deg at 120 deg and J3 reading 22 or 24 deg
for one command depending on the side it came from [OURS].

Read-back measures what each servo believes. It sees load droop, the backlash side, the lag
(the M3's camera probe found 100-195 ms) and a stop; it cannot see a wrong pulse-to-degree scale,
link flex or the camera mount. One scale error may already be in our data: Yahboom maps 12.2 pulses
to a degree, the servo sheet implies ~13 (96-4000 over 300 deg), which would turn a joint 0.90-0.94
of its command -- the camera probe measured 0.91-0.94 on the unloaded joints [OURS, est.]. Read-back
converted with the same map would call that 1.00; the camera stays the judge of scale.

Uses, most valuable first:
- **Stall, collision and grasp.** Flag a joint when the command (delayed 0.15 s) and the reading
  differ by > 3 deg for > 0.3 s; close the gripper until its reading stops changing, which both
  prevents the stall that can burn servo 6 and measures the held object's width -- a free grasp
  verdict instead of a second look (est.).
- **Sag and backlash at every pose**, without the ChArUco board: a droop table per pose and load,
  approach-side backlash, settling times for the robot card.
- **Teaching by hand.** Torque off and record at 10 Hz: kinesthetic demonstrations and a hand-taught
  aim table instead of keyboard teaching [DOC: Yahboom's X3 "memory learning"].
- **The policy's state.** ALOHA fed measured joints; ours feeds the last command, which lets ACT
  copy itself. A measured state still trails the command by 0.1-0.2 s, so it is not the cure (the
  ACT report's fix, dropping the state input, stands); it adds information when the arm does not
  follow.
- **Load and temperature**, if the servos fill SCS registers 60-63: a second stall signal and a duty
  limit (unverified).

## 6. The paths, ranked

| rank | path | what it gives | risk | effort | how to undo |
|---|---|---|---|---|---|
| 1 | Read-only survey (tier A below) | facts: scripts' protocol and read call, serial chip, agent, scan fields, chip marking, servo label | none | 1 h | nothing to undo |
| 2 | Parameter read-out, `config_robot.py` with all `set_` lines commented (agent stopped) | firmware version, `arm_mid`, PIDs, scales, domain ID -- the record needed before any flash | low: the board may need a power cycle | 15 min | restart the agent, power-cycle |
| 3 | Bench test: a spare Yahboom servo + a half-duplex adapter on the laptop | SCS protocol confirmed, which registers exist (56-70), reply delay, read time, noise at a fixed angle | none to the robot | half a day | n/a |
| 4 | Arm-only session with Yahboom's protocol (if a read call exists) | torque-off teaching; full joint reads for sag, backlash and gain tables | low (agent stop) | 1 day | restart the agent |
| 5 | Listen on the servo bus (tier C) | when the board talks, IDs, write type, silent gaps | very low: RX and GND only | 2 h + a cable | unplug |
| 6 | Query the board's UART6 | if answered: run-time reads through the board, nothing on the servo bus | low: 3.3 V adapter, one read request | 1 h to test | unplug |
| 7 | Second master on the servo bus, reads scheduled by the bridge | 10 Hz joint states in step with commands; stall and grasp detection at run time | medium: line contention, a collision could corrupt a command (mitigated by schedule, open-drain TX, echo check) | 2-3 days | unplug the tap; bridge flag off |
| 8 | Host re-stamper for the board topics (lidar, odometry, IMU) | clock offset and mean scan lag removed, EKF stops waiting | none | 1-2 days | stop the node |
| 9 | New firmware: joint publisher + per-revolution lidar stamps + periodic sync | 20 Hz joints stamped at the read; true per-point lidar time; no drift | high without Yahboom's source (base behaviour changes); recoverable by flashing | days with the source, weeks without | flash `V1.1.3.hex`, restore parameters |

Ask Yahboom support (support@yahboom.com, on their site's footer) for the factory firmware source
now; the answer decides whether row 9 is days or weeks.

## 7. Read-only robot checklist (for a later session; not run here)

Tier A -- over SSH, robot running, nothing stopped or written:
1. `ros2 node info /YB_Node`; `ros2 topic list -t | grep -i -E "joint|servo|arm"`;
   `ros2 interface show arm_msgs/msg/ArmJoints` and `arm_msgs/msg/ArmJoint`.
2. `ls -la ~/*.py ~/*.sh`; `cat ~/start_agent.sh`; `sed -n '1,400p' ~/config_robot.py` and
   `~/calibrate_arm.py`; `grep -n -E "^(import|from)|def |FUNC_|0x[0-9A-Fa-f]{2}|serial|baud" ~/config_robot.py ~/calibrate_arm.py`
   -- frame format, baud, a servo read call, and whether the script only reads with every `set_`
   commented.
3. Read (not import) the library those scripts import: `grep -rn "def get_\|def set_uart" <its path>`.
4. `find ~ -maxdepth 4 \( -iname "*.hex" -o -iname "*FW*" -o -iname "Board_Samples" \) 2>/dev/null | head`.
5. `ls -l /dev/myserial /dev/serial/by-id/`; `udevadm info -q property -n /dev/myserial | grep -E "ID_VENDOR|ID_MODEL|ID_SERIAL"`;
   `dmesg | grep -i cp210`.
6. `ps -ef | grep -i [m]icro_ros`; `docker ps`; `ros2 pkg xml micro_ros_agent 2>/dev/null | grep -i version`.
7. `ros2 topic echo --once /scan0 --field time_increment` (and `scan_time`, `header`);
   `ros2 topic hz /scan0`; `ros2 topic delay /scan0`, `/imu/data_raw`; `cat /proc/uptime`.
8. `grep -rn -E "GetCurrentArmJointAngles|def _get_joints|Curjoints|joint_states" ~/M3Pro_ws/src ~/yahboomcar_ws/src --include=*.py | head -30`.
9. Photos: the MCU marking (STM32H743V?T6: VG = 1 MB or VI = 2 MB -- the course says both),
   the S3/S4/S5, UART6, SWD and CAN headers, BOOT0/RESET, one arm servo's label, S3's pinout.

Tier B -- ask first; the agent is stopped:
10. `python3 ~/config_robot.py` with every `set_` line commented (after step 2 shows it only reads):
    record version, `arm_mid`, PIDs, scales, domain ID.
11. Optional, with the user: a flash backup read through the bootloader (BOOT0 + RESET, Type-C to the
    laptop, STM32CubeProgrammer): reads the image or reports protection; writes nothing.

Tier C -- ask first; hardware:
12. Meter the S3 pins (GND, 6.8 V, signal; signal idle level). Adapter RX + GND only on the signal,
    115200 8N1; log while the bridge sends a few moves.
13. Optional: one version request on UART6 with a 3.3 V adapter (TX, RX, GND only).

## 8. Risks

- **Flashing:** recoverable through the ROM bootloader, but a full erase loses the arm's mid values
  and the ROS settings (record them, tier B); RDP may block a backup; a self-built base firmware
  changes odometry and PID behaviour and invalidates base calibrations; never write option bytes.
  The flashing port is the agent's port: the robot is offline while flashing.
- **The bus tap:** the bus carries 6.8 V on one pin and a 5 V (sibling board) or 3.3 V signal --
  meter it; a push-pull adapter fights the board's driver; a collision can corrupt a command into
  another valid one. Listen before transmitting; transmit only in the bridge's slot; check the echo.
- **Torque off drops the arm.** Yahboom raises the arm vertical before unloading; support it by hand.
- **Agent stops** have once left the board silent until a power cycle (2026-10-15).
- **Read-back is not truth.** It reports the servo's own potentiometer in Yahboom's pulse map; a
  scale error (the ~0.9 gain) or link flex stays invisible; the camera remains the reference.
- **Unknowns that change the plan:** whether the M3 scripts expose a servo read; what UART6 runs;
  whether the board ever reads or writes the bus on its own; which registers the servos fill;
  whether Yahboom will share the firmware source.

## Conclusion

The M3's arm is not deaf; its firmware just never asks over ROS. The servos speak a
Feetech-SCS-style protocol with a position read, and the factory firmware switches torque off and,
by all signs, reads positions for its own calibration. Do not reflash first. Read the robot (an hour, tier A), record the
board's parameters, and test whether Yahboom's own protocol hands positions to the host: if it does,
torque-off teaching and calibration reads work today. For reads while the policy runs, listen on the
servo bus, then let the bridge read in the silence after each command, with an open-drain tap and
an echo check. Rebuild the board's firmware only when it can carry the lidar fix as well, and only
from Yahboom's source; keep their hex and the recorded parameters as the way back.
