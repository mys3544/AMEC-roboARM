# 1. The M3 Pro's base board, its arm servos and its firmware

Scope: which board and MCU run the M3 Pro's base and arm, which servos the arm has and what they
can report, whether the board's firmware is published, how it is flashed, backed up and recovered,
and what adding a joint-state publisher to it would cost on the servo bus. Workarounds without new
firmware are in note 2, the lidar stamps in note 3, what read-back buys and the robot checklist in
note 4. Labels: [DOC] a vendor document (Yahboom's course or product pages, ST, Feetech),
[CODE] source code read (vendor sample code shown in a course counts as [CODE, vendor sample]),
[CLAIM] a statement nobody measured, [2nd] a search summary or a forum, [OURS] the repo's own
records of our robots, "est." our arithmetic. Nothing was run on the robot (research only,
2026-10-27); Yahboom's course pages were read through their JSON API
(`https://api.yahboom.net/api/build?id=<chapter>`), which returns the same HTML the course shows.

## 1. One board does everything: an STM32H743 (Cortex-M7) running micro-ROS Humble

### Takeaway
The M3 Pro has a single control board: an STM32H743 that runs the motors, encoders, IMU, both
lidars, the OLED, the buzzer and the arm's bus servos, and talks to the Jetson through an on-board
CP2104 USB-serial chip at 2 Mbaud with micro-ROS (Humble). There is no separate servo controller:
the arm's servos hang directly on the MCU's USART3 through the board's 6.8 V "S3" bus-servo port.

Evidence:
- "Search and select the STM32H743VGT6 chip" when creating a project for this board
  ([Set up the STM32CUBEIDE development environment](https://www.yahboom.net/study/ROSMASTER-M3PRO/20494));
  "the STM32 control board integrates the STM32H743 chip"
  ([Post a topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20518)) [DOC]. The micro-ROS
  library is cross-compiled with `-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard`
  ([Compile the microros driver library](https://www.yahboom.net/study/ROSMASTER-M3PRO/20497)) [DOC].
- Conflict: the flash chapter says the board "has 2M Flash space" in two 1 MB banks
  ([Flash access data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20512)) [DOC], while by ST's
  part-number code a "VG" H743 has 1 MB (a "VI" has 2 MB) [2nd, not re-checked today]. The chip
  marking or the bootloader's device ID settles it (note 4 checklist).
- Pin table: bus servo on PD8/PD9 = serial port 3; left-rear lidar on serial port 4 (PC10/PC11),
  right-front lidar on serial port 5 (PC12/PD2); "Burning and communication interface PA9/PA10
  Serial port 1"; debug log on serial port 7 (PE7/PE8); a "Control interface: Serial port 6, can be
  connected to TTL module and send protocol to control the robot" (PC6/PC7); FDCAN on PD0/PD1;
  SWD on PA13/PA14; BOOT0 and RESET buttons; 25 MHz crystal; IMU on SPI2
  ([Introduction to the Control Board](https://www.yahboom.net/study/ROSMASTER-M3PRO/20493)) [DOC].
  The table lists no direction-control pin for the servo bus.
- Three bus-servo ports: "S3 is a 6.8V bus servo interface, and S4 and S5 are 12V bus servo
  interfaces. Since the M3PRO's robotic arm bus servo is 6.8V, we will use S3"
  ([Subscribe to the bus servo topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20522)) [DOC].
- USB-serial: "the CP2104 serial port chip is an onboard component"
  ([Serial communication](https://www.yahboom.net/study/ROSMASTER-M3PRO/20503)) [DOC]; our robot
  shows a CP210x as `/dev/myserial`, held by the agent [OURS, project memory m3pro-robot].
- micro-ROS: the course builds the library from `micro_ros_setup` branch `humble` with a custom
  serial transport on USART1 at 2 Mbaud with DMA, FreeRTOS, `RMW_UXRCE_MAX_PUBLISHERS=10`,
  `MAX_SUBSCRIPTIONS=10`, `MAX_NODES=1`, `MAX_SERVICES=1`
  ([20497](https://www.yahboom.net/study/ROSMASTER-M3PRO/20497), [20518](https://www.yahboom.net/study/ROSMASTER-M3PRO/20518));
  the agent is `microros/micro-ros-agent:humble serial --dev /dev/myserial -b 2000000`
  ([Install and start the microros agent](https://www.yahboom.net/study/ROSMASTER-M3PRO/20498)) [DOC].
  Whether the factory firmware was built with the same limits is not known.
- The factory node `/YB_Node` subscribes `/arm6_joints`, `/arm_joint`, `/beep`, `/cmd_vel`, `/rgb`
  and publishes `/battery`, `/imu/data_raw`, `/odom_raw`, `/scan0`, `/scan1`
  ([ROS control](https://www.yahboom.net/study/ROSMASTER-M3PRO/15792)) [DOC]; our `ros2 topic list`
  agreed: no servo angle comes back [OURS, NEXT.md 2026-10-23].

## 2. The arm's servos speak a Feetech-SCS-style protocol and can report their position

### Takeaway
Yahboom's own STM32 code drives the bus servos with packets that are byte-for-byte Feetech's SCS
protocol: header `FF FF`, WRITE (`0x03`) of the goal position at address `0x2A`, READ (`0x02`) of
two bytes at address `0x38` for the present position, an inverted-sum checksum, big-endian values.
So the servos answer position reads, and probably (unverified) the rest of the SCS table: speed,
load, voltage, temperature. The M3's arm calibration uses the same pulse map as the X3's arm
(2000 = 90 deg for J1-J4, 1486 = 90 deg for J5, 3100 = shut gripper), so it is the same Yahboom
servo family, at 6.8 V. The exact servo model on the M3 is not named anywhere we found.

Evidence:
- Yahboom's STM32 course for its sibling boards (ROSMASTER M1 Pro / A1 Pro, "Control Serial Port
  Servos"): `UartServo_Ctrl()` builds `{0xff, 0xff, id, 0x07, 0x03, 0x2a, pos_H, pos_L, time_H,
  time_L, checksum}` with `checksum = ~(id + len + cmd + addr + ...) & 0xff`; `UartServo_Get_Angle()`
  sends `{0xff, 0xff, id, 0x04, 0x02, 0x38, 0x02, checksum}` ("Request current position of servo");
  `UartServo_Rx_Parse()` takes an 8-byte reply and reads `Rx_Data[5] << 8 | Rx_Data[6]`; the header
  defines `MEDIAN_VALUE 2000`, `MID_VAL_ID6 3100`, `MID_ID5_MAX 3700`, `MID_ID5_MIN 380`,
  `MID_VAL_ID5 1486`, `MAX_PULSE 4000`, `MIN_PULSE 96`, and declares `UartServo_Sync_Write`,
  `UartServo_Set_Torque`, `UartServo_Set_ID`
  ([Control Serial Port Servos, M1 Pro course](https://www.yahboom.net/study/ROSMASTER-M1PRO/20707);
  code shown as images there, read 2026-10-27) [CODE, vendor sample]. The board in that chapter is
  the STM32F103 one (servo on USART3, PC10/PC11), not the M3's H743; the protocol is the servos'.
- Feetech's SCSCL table: `SCSCL_TORQUE_ENABLE 40`, `SCSCL_GOAL_POSITION_L 42`,
  `SCSCL_PRESENT_POSITION_L 56`, `SCSCL_PRESENT_SPEED_L 58`, `SCSCL_PRESENT_LOAD_L 60`,
  `SCSCL_PRESENT_VOLTAGE 62`, `SCSCL_PRESENT_TEMPERATURE 63`, `SCSCL_MOVING 66`,
  `SCSCL_PRESENT_CURRENT_L 69`; instructions `INST_READ 0x02`, `INST_WRITE 0x03`,
  `INST_SYNC_WRITE 0x83`, `INST_SYNC_READ 0x82`
  ([FTServo_Arduino SCSCL.h](https://github.com/ftservo/FTServo_Arduino/blob/main/src/SCSCL.h),
  [INST.h](https://github.com/ftservo/FTServo_Arduino/blob/main/src/INST.h)) [CODE]. Addresses 42
  and 56 match Yahboom's packets exactly. Feetech ships SyncRead examples only for its SMS/STS and
  HLSCL lines, not for SCSCL ([examples tree](https://github.com/ftservo/FTServo_Arduino)) [CODE]:
  do not count on sync read.
- The M3's arm mid values: `arm_mid_value = [2000, 2000, 2000, 2000, 1486, 3100]` in
  `config_robot.py`; Yahboom's screenshot of its read-out shows `arm_mid: 2025,2077,1957,2017,1419,3005`
  on a calibrated unit ([Modify the firmware parameters](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771)) [DOC].
  The X3's Rosmaster_Lib maps J1-J4 0-180 deg to pulses 3100-900, J5 0-270 deg to 380-3700, and
  the gripper 0-180 to 900-3100 (`__arm_convert_value`, Rosmaster_Lib V3.3.9 in
  `AMEC releases/RoboARM-X3Plus-2026-09-30/vendor/Rosmaster_Lib`) [CODE]: 90 deg is 2000 and 1486,
  180 deg on the gripper is 3100 -- the same map.
- Yahboom's 15 kg bus servo: "high precision potentiometers inside", "supports reading back the
  current position and status", "3 pin serial bus interface"
  ([RobotShop listing](https://www.robotshop.com/products/yahboom-yahboom-15kg-serial-bus-smart-servo-robotic-arm-w-english-manual)) [DOC];
  6-7.4 V, 300 deg, 115.2 kbps, potentiometer, "Supports angle readback"
  ([servodatabase YB-SD15M](https://servodatabase.com/servo/yahboom/yb-sd15m)) [2nd]. Whether the
  M3's 6.8 V servos are this model is not stated by Yahboom; its product page and parameter image
  name no servo ([store page](https://www.yahboom.store/products/rosmaster-m3-pro)) [DOC].
- est.: 2200 pulses per 180 deg = 12.2 pulses/deg, so one pulse is 0.082 deg.

## 3. The factory firmware is published as a hex file only; the course's sample projects are source

### Takeaway
Yahboom publishes the M3 Pro's factory firmware as `microROS_STM32-FW_V1.1.3.hex` (971 kB,
2025-08-28) and the course's sample projects (`Board_Samples/...`) as source. The factory
firmware's own source was not found: not on GitHub, not in the course, and the course itself says
some firmware comes as "only a hex file". So "add a joint publisher to the factory firmware" is not
possible as such; the options are Yahboom's source (ask them), or a replacement firmware assembled
from the samples, which re-implements the base as well.

Evidence:
- Google Drive linked from the course and from the GitHub repo
  ([Annex_Download_Link.txt](https://github.com/YahboomTechnology/ROSMASTER-M3PRO/blob/main/Annex_Download_Link.txt)):
  `6.Firmware_DriverFile/Control board factory firmware/microROS_STM32-FW_V1.1.3.hex`, 971 kB,
  dated 28 Aug 2025; next to it `CP2104 serial port driver`, `STM32CUBEIDE`, `STM32CUBEPRG`
  (listing seen 2026-10-27) [DOC]. `1.Code` holds `ROSMASTER M3 PRO-Code.zip` (1.21 GB) and
  `ros2_ws.zip` (34.4 MB); Drive cannot preview the big zip and it was not downloaded (a download
  needs the user's approval), so whether it contains the factory firmware's source is unknown.
- GitHub: `YahboomTechnology/ROSMASTER-M3PRO` holds 288 PDFs of the course (plus a few Dify
  workflow files and spreadsheets) and no firmware or robot source (tree
  listed through the GitHub API, 2026-10-27); the same holds for ROSMASTER-M3, M1, M1PRO, A1PRO,
  MicroROS-Board and ROS-robot-expansion-board [CODE: repository trees]. A web search for
  `arm6_joints` / `YB_Node` firmware found no source anywhere [2nd].
- The course: "If the firmware only has a hex file and does not have the full functional source
  code, you can use STM32CubeProgrammer"
  ([Burning STM32 firmware using SWD](https://www.yahboom.net/study/ROSMASTER-M3PRO/20495)); the
  micro-ROS samples are at "Board_Samples/Microros_Samples/Publisher", ".../Subscriber_uart_servo"
  and so on ([20518](https://www.yahboom.net/study/ROSMASTER-M3PRO/20518),
  [20522](https://www.yahboom.net/study/ROSMASTER-M3PRO/20522)) [DOC]. The samples cover each
  piece (odometry, IMU, lidar, arm subscriber, flash storage, PID) but not the factory protocol on
  the USB port that `config_robot.py` and `calibrate_arm.py` use (note 2).
- The factory version on our robot is not known; Yahboom's own screenshot shows `version: 1.1.2`
  ([15771](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771)) [DOC].

## 4. A joint publisher would cost 10-15 ms of servo-bus time per sweep: 20 Hz fits beside commands

### Takeaway
On a 115200-baud half-duplex bus one position read is 16 bytes on the wire, ~1.4 ms plus the
servo's reply delay, so a sweep of six joints takes ~9-12 ms. A firmware publisher at 20 Hz uses
~20-25 % of the bus and leaves room for our 10 Hz commands (~3-6 ms each); 50 Hz would take
~55-60 % and only works if reads yield to commands. Over micro-ROS a six-joint message is
negligible at 2 Mbaud. The firmware has the static room for one more publisher if it was built
with the course's limits (5 publishers used of 10).

Evidence and arithmetic (all est. unless labelled):
- 115200 baud, 8N1: 10 bits a byte, 86.8 us a byte. READ request 8 bytes + reply 8 bytes
  (formats in section 2) = 1.39 ms on the wire; add the servo's return delay and the MCU's turnaround
  (unknown for Yahboom's servo; SCS-family servos have a configurable return delay) -> ~1.5-2 ms a
  joint, 9-12 ms for six.
- Commands: one `UartServo_Ctrl` is 11 bytes (0.95 ms), six of them 5.7 ms; an SCS sync write for
  six joints is 7 + 6 x 5 + 1 = 38 bytes, 3.3 ms. Which one the factory firmware sends is visible
  with a passive listener (note 4).
- Bus share at 20 Hz sweeps + 10 Hz commands: (10.5 x 20 + 4.5 x 10) / 1000 = ~25 %. At 50 Hz:
  ~57 %. The firmware must never start a read while a command is queued (one bus task that owns
  USART3 and serves commands first).
- micro-ROS side: a small custom message (six int16 + a stamp) is under ~60 bytes with XRCE
  headers, 0.3 ms at 2 Mbaud (200 kB/s); a `sensor_msgs/JointState` with six names and three float64
  arrays is ~250-300 bytes, ~1.5 ms. The board already sends ~4.1 kB scans at 7 Hz per lidar
  ([note 1 of Base localization](../Base%20localization%20while%20turning/1%20Error%20sources%20on%20the%20M3%20while%20it%20turns.md)).
- Prior art: Husarion's ROSbot firmware (STM32, micro-ROS) publishes its wheels as
  `sensor_msgs/JointState`, stamped with `rmw_uros_epoch_nanos()` when the clock is synchronised
  ([wheels.cpp](https://github.com/husarion/rosbot_ros2_firmware/blob/main/src/wheels.cpp),
  [microros.cpp](https://github.com/husarion/rosbot_ros2_firmware/blob/main/src/microros.cpp)) [CODE].
- The X3 shows what host polling through a board costs: each `get_uart_servo_value()` waits up to
  30 x 1 ms for the answer (Rosmaster_Lib V3.3.9) [CODE]; two reads need ~50 ms between them or both
  start failing, and a re-asked joint "held the line the stream needs ... for 50-300 ms"
  (`x3plus/roboarm/arm.py`, `x3plus/roboarm/x3pick.py`) [OURS]. A reader inside the firmware, or one
  that owns its own bus slot, avoids that.

## 5. Flashing is easy and recoverable; the backup and the stored calibration are the traps

### Takeaway
The board flashes over its Type-C port (CP2104 -> USART1) through the STM32's ROM bootloader
(hold BOOT0, press RESET) with STM32CubeProgrammer, or over SWD with an ST-Link. The ROM bootloader
cannot be overwritten, so a bad application image is always recoverable by re-flashing Yahboom's
hex. Two things are not automatically recoverable: the board's stored parameters (the arm's mid
values, the ROS domain ID, the PIDs and scales live in its flash and a full erase wipes them), and a
backup of the current image if the chip is read-protected. Write the parameters down first.

Evidence:
- "press and hold the BOOT button on the control board, press the RESET button again, and then
  release the BOOT button. The STM32 will enter the serial port programming mode" -- STM32CubeProgrammer
  in UART mode over the Type-C cable plugged into a computer
  ([Burning STM32 firmware using serial port](https://www.yahboom.net/study/ROSMASTER-M3PRO/20496));
  SWD with ST-LINK and STM32CubeIDE or STM32CubeProgrammer
  ([20495](https://www.yahboom.net/study/ROSMASTER-M3PRO/20495)) [DOC]. "Since the Type-C
  communication serial port used by the microros agent is multiplexed with the burning serial port,
  it is recommended to use the STlink tool"
  ([20522](https://www.yahboom.net/study/ROSMASTER-M3PRO/20522)) [DOC].
- Stored parameters: `set_ros_domain_id`, `set_ros_namespace`, `set_motor_pid_parm`,
  `set_imu_yaw_pid_parm`, `set_ros_scale_line`, `set_ros_scale_angular`, `set_arm_mid_value`, read
  back by `python3 ~/config_robot.py` with the agent stopped
  ([15771](https://www.yahboom.net/study/ROSMASTER-M3PRO/15771)); the H743's flash is 128 kB
  sectors and the samples keep user data in a sector of their own
  ([20512](https://www.yahboom.net/study/ROSMASTER-M3PRO/20512)) [DOC]. Which sector the factory
  firmware uses is unknown, so erase only what the new image needs, and re-enter the values after
  any full erase.
- The arm's mid values are what `calibrate_arm.py` measured with the servos limp; after a wipe the
  arm's zero is off until they are restored (`set_arm_mid_value`) or the calibration is redone
  ([Robotic Arm Calibration](https://www.yahboom.net/study/ROSMASTER-M3PRO/15767)) [DOC].
- Readout protection: at RDP level 1 the flash cannot be read through the bootloader or a debugger,
  and going back to level 0 mass-erases it; level 2 is permanent
  ([ST wiki, Security:RDP](https://wiki.st.com/stm32mcu/wiki/Security:RDP); ST community threads)
  [2nd, the ST page timed out on fetch]. Whether Yahboom sets RDP is unknown: a backup attempt
  (`STM32CubeProgrammer`, read 0x08000000) either works or says so, and changes nothing. Never touch
  the option bytes (RDP, BOOT_ADD).
- The 2026-10-15 board hang needed a full power cycle before `/arm6_joints` came back [OURS,
  project memory]: every step that stops the agent or resets the board should expect that.
