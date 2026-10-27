# 2. Reading the M3's joints without new firmware

Scope: every way found to get the M3 Pro's servo angles back without replacing the board's factory
firmware: Yahboom's own non-ROS protocol on the board's USB port (used by `calibrate_arm.py` and
`config_robot.py`), the board's second serial "control interface", a second reader on the servo
bus itself, Yahboom's teach ("memory learning") features on its other arms, and the X3's
Rosmaster_Lib route. The board, the servos and the firmware facts are in note 1. Labels as in
note 1.

## 1. The factory firmware already has a non-ROS protocol that turns the servos off and calibrates them

### Takeaway
With the micro-ROS agent stopped, Yahboom's `~/calibrate_arm.py` talks to the board directly over
`/dev/myserial`: it prints the board's firmware version, "unloads" the servos so the arm can be
moved by hand, and stores each servo's present position as its new mid value (a per-servo
"state = 1" on success). So the factory firmware can switch the arm's torque off and can read each
servo's position -- just not over micro-ROS. This contradicts the note in `m3arm.py` that "there is
no torque switch". Whether that protocol also hands the positions to the host (as the X3's board
does) is not documented; the two scripts on the robot will say (note 4 checklist).

Evidence:
- "use ctrl+c in the proxy terminal to close the proxy program and run the following command again
  to unload the servo `python3 ~/calibrate_arm.py`. After running, the underlying control board
  version V1.X will be printed ... After running, the six servos can be moved at will ... enter ...
  y, confirm that the centering is complete ... The calibration will print the state values of the
  six servos respectively. The calibration is successful only when state = 1. ... reconnect the
  proxy, `sh start_agent.sh`"
  ([Robotic Arm Calibration](https://www.yahboom.net/study/ROSMASTER-M3PRO/15767)) [DOC].
- `config_robot.py` (agent closed first) writes and then reads back `car_type: 7`, `version: 1.1.2`,
  `domain_id`, `ros_namespace`, `ros_scale_line/angular`, the motor and IMU PIDs and
  `arm_mid: 2025,2077,1957,2017,1419,3005` (Yahboom's screenshot)
  ([Modify the firmware parameters](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771)) [DOC].
- These are the functions and replies of the X3's Rosmaster_Lib: `get_version()` (`FUNC_VERSION
  0x51`), `set_uart_servo_torque()` (`FUNC_UART_SERVO_TORQUE 0x22`), `set_uart_servo_offset(id)`
  (`FUNC_ARM_OFFSET 0x24`, waits for the board's `(id, state)` reply), and the read requests
  `get_uart_servo_value(id)` (`FUNC_REQUEST_DATA 0x50` + `FUNC_UART_SERVO 0x20`, reply `(id, int16
  position)`) and `get_uart_servo_angle_array()` (`0x50` + `FUNC_ARM_CTRL 0x23`, reply six int16),
  frames `FF FC len func ... checksum` at 115200 baud (Rosmaster_Lib V3.3.9) [CODE]. That the M3
  scripts speak this same frame format is an inference from the matching names, replies and
  defaults (`set_arm_mid_value` = six pulses on the same map); the scripts were not read.
- A mid-value calibration needs the servo's present position, so the factory firmware contains a
  position read (as Yahboom's `UartServo_Get_Angle()` sample does, note 1 section 2) [inference].
  The stored values support it: the board reads back `arm_mid` in pulses (2025, 2077, 1957, 2017,
  1419, 3005 on Yahboom's unit), i.e. where each servo stood when the arm was held straight, not a
  flag. The other way to calibrate -- telling each servo to re-zero itself (Feetech's `INST_CAL
  0x0B`, used by its `CalibrationOfs` examples) -- would leave nothing to store on the board, and
  Feetech ships that example only for its STS and HLS lines, not SCS
  ([FTServo_Arduino examples](https://github.com/ftservo/FTServo_Arduino)) [CODE, inference].
- Yahboom's FAQ: "The robotic arm is in a torque-free state upon power-up" until the first
  `/arm6_joints` command ([FAQ](https://www.yahboom.net/study/ROSMASTER-M3PRO/15772)) [DOC]; on our
  robot the arm came up stiff in the ready pose after a power cycle (`m3pro/roboarm/m3arm.py`
  docstring, 2026-10-05) [OURS] -- consistent if the gamepad node, which sends the ready pose at its
  start, came up first.

What it allows (est.): an "arm-only session" -- agent stopped, the base and lidars silent -- in
which a host script switches the torque off for teaching by hand and, if the read request is
answered, reads all six joints like the X3 does (one request, six values). Good for calibration
tables (sag, backlash, gain) and kinesthetic demonstrations; useless for the policy at run time,
because micro-ROS and this protocol share one port and only one can run.

## 2. The board's "control interface" (serial port 6) is an unexplored second door

### Takeaway
The board has a second host serial port, "serial port 6 ... can be connected to TTL module and send
protocol to control the robot", plus a CAN port. If the factory firmware serves its control
protocol on UART6 while micro-ROS runs on USART1, a 3.3 V USB-TTL adapter from the Jetson to that
header would give joint reads through the board with nothing on the servo bus and no firmware
change. Nothing published says which protocol (if any) the factory firmware runs there; one
harmless request (the version query) answers it.

Evidence:
- "Control interface: Serial port 6, can be connected to TTL module and send protocol to control the
  robot"; "CAN interface: can be connected to CAN bus devices and send protocols to control the
  robot"; pins PC6/PC7 and PD0/PD1
  ([Introduction to the Control Board](https://www.yahboom.net/study/ROSMASTER-M3PRO/20493)) [DOC].
- The course's CAN sample only prints received frames in loopback mode
  ([CAN bus communication](https://www.yahboom.net/study/ROSMASTER-M3PRO/20517)) [DOC]; no chapter
  documents a protocol on UART6 (all 34 control-board chapters listed through the course API,
  2026-10-27).
- The test (not done): with the robot running normally, send only the version request (the
  Rosmaster frame for `FUNC_REQUEST_DATA` + `FUNC_VERSION`, if `config_robot.py` confirms that
  format) on UART6 and listen. A request for data moves nothing. Levels: the H743 is a 3.3 V part,
  so use a 3.3 V adapter and connect only TX, RX and GND.

## 3. A second reader on the servo bus works if it only speaks when the board is silent

### Takeaway
The arm's bus is one data line shared by the board and six servos. A USB-to-TTL adapter on the
Jetson can join it: first as a pure listener (RX and GND only: no risk, and it shows exactly what
the board sends and when), then as a second master that sends READ requests only in the gaps after
our own commands. Since the board writes to the bus only when we publish to `/arm6_joints` or
`/arm_joint` (to be confirmed by listening), the bridge can own the schedule: publish a command,
wait ~30 ms, read six joints in ~12 ms, repeat -- 10 Hz joint states in step with the 10 Hz policy,
without touching the firmware. The electrical rule: the adapter must never drive the line high
(open-drain or tri-state output, or a diode on TX), and must never see the 6.8 V servo supply.

Evidence:
- The board's servo port has no direction pin in the M3 pin table (PD8/PD9 = USART3 TX/RX only)
  ([20493](https://www.yahboom.net/study/ROSMASTER-M3PRO/20493)) [DOC]. Yahboom's sibling board's
  schematic (STM32F103 course) joins TX and RX onto the single "Servo" line with a 74HC126 tri-state
  buffer, an SN74AHC1G04 inverter making `TXEN` from the data line, 1 kOhm series resistors and a
  4.7 kOhm pull-up to 5 V; the connector J10 carries GND, 6.8 V and the signal
  ([M1 Pro course 20707, schematic image](https://www.yahboom.net/study/ROSMASTER-M1PRO/20707)) [DOC].
  The M3 board's schematic was not found; measure the idle level and the pinout before connecting.
- The firmware receives on USART3 by interrupt, one byte at a time, into a parser that looks for
  `FF FF` frames of 8 bytes ([20707](https://www.yahboom.net/study/ROSMASTER-M1PRO/20707)) [CODE,
  vendor sample]: replies to someone else's reads would reach it too. In the sample the parser only
  prints; in the factory firmware it presumably matters only during its own reads (calibration).
- Collision hazard (est.): the SCS checksum is 8 bits, so a corrupted frame passes it with a chance
  of about 1 in 256; a board WRITE garbled by our READ could move a joint to a wrong place. Hence
  "never transmit while the board might": after each publish wait out the board's latency (its ROS
  loop spins every 10 ms, `vTaskDelayUntil(&lastWakeTime, 10)`, [20522](https://www.yahboom.net/study/ROSMASTER-M3PRO/20522))
  plus the command's own 3-6 ms on the wire; and compare our echo with what we sent -- a mismatch
  means a collision, so stop and re-send the last pose.
- Who else publishes arm commands: Yahboom's gamepad node `yahboom_joy_M3Pro` (desktop autostart)
  sends the ready pose at its start [OURS, `m3arm.py`]; every other source must be off during
  reading, or the bridge must be the only publisher.
- Bus timing (note 1 section 4, est.): one read ~1.5-2 ms, six ~9-12 ms; a 6-joint command
  3.3-5.7 ms. A 100 ms cycle has ~80 ms of silence.
- Where to tap (est.): bus servos have two connectors and are daisy-chained, so the last servo
  (the gripper) has a free one, or a 3-pin splitter at the board's S3 port. Feetech's and
  Waveshare's bus-servo adapters are made for this protocol family
  ([Waveshare Bus Servo Adapter (A)](https://www.waveshare.com/wiki/Bus_Servo_Adapter_(A)): "ST/SC
  series serial bus servos") [DOC]; their output stage was not documented on the page, so check it
  is not push-pull on the line before using one as a second master.

## 4. Yahboom's teach modes exist on its other arms, not on the M3 Pro

### Takeaway
Yahboom's X3 Plus and DOFBOT arms have a "memory learning" mode: torque off, move the arm by hand,
record each pose from the servo read-back, play it back. The M3 Pro course has no such chapter and
no read-back topic; its demos publish `Curjoints`, which is their own record of what they sent. The
OpenClaw "GetCurrentArmJointAngles" tool on the M3 is very likely the same kind of record (there
is no source topic for a measured angle).

Evidence:
- X3 Plus: "the robotic arm will turn off the torque. At this time, we can manually move each
  servo"; it uses `driver.car.set_uart_servo_torque(0)` and `get_uart_servo_angle_array()`, up to 20
  poses ([Voice control robotic arm memory learning](https://www.yahboom.net/study/ROSMASTER-X3-PLUS/8362)) [DOC].
- DOFBOT Pro: `Arm.Arm_serial_servo_read(id)` returns the current angle (0-180, J5 0-270)
  ([Read Servo Angle](https://www.yahboom.net/study/DOFBOT-Pro/12622)); `Arm_Action_Study()` records
  the current pose in "Memory Learning Mode"
  ([17621](https://www.yahboom.net/study/DOFBOT-Pro/17621)) [DOC].
- M3 Pro demos: `self.pub_cur_joints = self.create_publisher(CurJoints,"Curjoints",1)` and "Publish
  the message of the six servo values of the robot arm at this moment, so that the clamping node can
  calculate the current position" -- values the node itself commanded
  ([Tracking and gripping color block](https://www.yahboom.net/study/ROSMASTER-M3PRO/15846)) [CODE,
  vendor sample]; OpenClaw's tool list has "GetCurrentArmJointAngles: Retrieve current robot arm
  joint angles" ([OpenClaw robotic arm control](https://www.yahboom.net/study/ROSMASTER-M3PRO/19099)) [DOC];
  "Since the servo has no torque feedback, the grasping closure angle of the gripper needs to be
  specified through parameters" ([19106](https://www.yahboom.net/study/ROSMASTER-M3PRO/19106)) [DOC].
  We searched the 37 arm, MoveIt, OpenClaw-arm and multi-robot-arm chapters of the M3 Pro course for
  `joint_states`, `torque`, "read servo", "current angle", "unload" (2026-10-27): only the
  calibration chapter (section 1) unloads servos.

## 5. The X3's route is the same idea one level up: the board polls, the host asks

### Takeaway
The X3's STM32F103 board reads its servos on request from Rosmaster_Lib at 115200 baud, one joint
or all six per request; that is why the X3 has read-back. On the M3 the equivalent would be the
factory protocol of section 1, available only with the agent stopped. The X3 also shows the
pitfalls to design around: reads drop out, a negative angle and "no reply" look alike, and host
polls stall a command stream that shares the line.

Evidence:
- `get_uart_servo_value()` sends `0x50 0x20 id`, then polls 30 x 1 ms for `(read_id, value)`,
  returning `(-1, -1)` on timeout; `get_uart_servo_angle_array()` asks `0x50 0x23` and converts six
  int16s; `get_uart_servo_angle()` clamps out-of-range angles to -1 (Rosmaster_Lib V3.3.9) [CODE].
- Batch reads drop a joint "most often right after motion"; two reads back to back corrupt each
  other, 50 ms apart they coexist (10/10 complete poses); -1 means dead or out of range
  (`x3plus/roboarm/arm.py`, project memory rosmaster-sdk-readback-gotcha) [OURS].
- A read re-asked during the survey turn "held the line the stream needs every STREAM_S for 50-300
  ms ... the turn jumped 10-20 deg" (`x3plus/roboarm/x3pick.py`, 2026-10-21) [OURS].
