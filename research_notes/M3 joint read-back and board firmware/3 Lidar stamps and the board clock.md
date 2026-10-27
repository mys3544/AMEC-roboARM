# 3. The lidar stamps and the board's clock: what only new firmware can fix

Scope: the same board firmware that would publish joint states also stamps the two lidars' scans
and keeps the micro-ROS clock. This note separates what a firmware change would fix in the scans'
timing from what a host-side re-stamper (designed in
[Base localization note 2](../Base%20localization%20while%20turning/2%20Time%20stamps%20and%20clock%20sync.md))
can fix without one. The error sizes are in
[Base localization note 1](../Base%20localization%20while%20turning/1%20Error%20sources%20on%20the%20M3%20while%20it%20turns.md)
and are not repeated beyond what the comparison needs. Labels as in note 1.

## 1. Today the board stamps a rolling buffer when it sends it, on a clock synced once

### Takeaway
Yahboom's published sample (and, by our measurements, the shipped firmware) parses each T-mini Plus
packet into a fixed array of angle bins and publishes a copy of the whole array on a 140 ms timer,
stamped at that moment with the micro-ROS clock; `time_increment` and `scan_time` stay zero. That
gives three separate timing faults: (1) the board clock's offset from the Jetson, set once per agent
connection and drifting ~48 ms an hour; (2) the stamp is the send time, ~83 ms after the points'
mean time; (3) each scan's points span one revolution (~167 ms) and the seam between newest and
oldest moves from scan to scan. A host re-stamper fixes (1) and the mean of (2); only the firmware
knows (3).

Evidence:
- "Create a publisher timer with a publishing frequency of 7 Hz. `#define LIDAR_PUBLISHER_TIMEOUT
  (140)`"; in the callback `lidar_msg.ranges.data[i] = Lidar_Ranges[index]/1000.0;` then
  `timespec_t time_stamp = get_ros2_timestamp();` and `rcl_publish`
  ([Publish radar data topic](https://www.yahboom.net/study/ROSMASTER-M3PRO/20526)) [CODE, vendor
  sample]. The odometry and IMU samples stamp the same way at sending
  ([20524](https://www.yahboom.net/study/ROSMASTER-M3PRO/20524),
  [20525](https://www.yahboom.net/study/ROSMASTER-M3PRO/20525)) [CODE, vendor sample].
- Our scans: ~7.1 Hz, 500 points over 270 deg; stamps ahead of arrival by 55 ms (scans), 72
  (odometry), 74 (IMU) on 2026-09-30; 1.1 s ahead after ~23 h; an agent restart re-synced it
  ([Base localization note 1, sections 1 and 4](../Base%20localization%20while%20turning/1%20Error%20sources%20on%20the%20M3%20while%20it%20turns.md)) [OURS].
- `get_ros2_timestamp()`'s body is not shown in any chapter; the 1.1 s growth and the re-sync on
  restart mean it returns the micro-ROS epoch from a single sync [inference].

## 2. Firmware can stamp each lidar packet when it arrives and publish whole revolutions

### Takeaway
The MCU sees every lidar packet the moment its last byte lands in its UART. A firmware change can
stamp each packet there (a microsecond timer converted to the synced epoch), start a scan at the
packet the lidar marks as the start of a revolution, end it one revolution later, set the stamp to
the first point's time, `time_increment` to the lidar's sample period (1/4000 s) and `scan_time` to
the measured revolution period. Then the scans are true per-revolution scans, `laser_geometry`
interpolates each point's pose, and Cartographer de-skews -- with no guessing on the host. Nothing
on the host can recover the seam position or the arrival time of each packet, because the board
throws both away before sending.

Evidence:
- YDLIDAR's protocol marks packet types `CT_Normal = 0`, `CT_RingStart = 1` ("Starting package"),
  header `PH 0x55AA` ([ydlidar_protocol.h](https://github.com/YDLIDAR/YDLidar-SDK/blob/master/core/common/ydlidar_protocol.h)) [CODE].
- YDLIDAR's own host driver stamps points from arrival: `trans_delay = _serial->getByteTime()`,
  `m_PointTime = 1e9 / sample_rate`, and the delay of data still in the buffer is `size *
  trans_delay + delayTime` with `delayTime` counted in point times; each node carries `stamp` and
  `delayTime` ([YDlidarDriver.cpp](https://github.com/YDLIDAR/YDLidar-SDK/blob/master/src/YDlidarDriver.cpp)) [CODE].
  The board firmware is in the same position as that driver, but drops the stamps.
- The T-mini Plus ranges at 4000 Hz and spins at 6 Hz by default (0.54 deg a point)
  ([data sheet](https://cdn.soselectronic.com/productdata/6e/cf/9b94bb2f/t-mini-plus.pdf), as read
  for Base localization note 1) [DOC].
- Consumers that use per-point time: `laser_geometry` interpolates the target-frame transform
  between `stamp` and `stamp + (n-1) * time_increment`; Cartographer extrapolates a pose per point;
  slam_toolbox and AMCL use only the stamp
  ([Base localization note 1, section 2](../Base%20localization%20while%20turning/1%20Error%20sources%20on%20the%20M3%20while%20it%20turns.md)) [CODE, read there].
- Sample's own artefacts a firmware fix would also remove: bins a packet skipped are "filled with
  the next packet's first distance", and points are placed at a fixed 0.541 deg from each packet's
  start angle ([Read Lidar data](https://www.yahboom.net/study/ROSMASTER-M3PRO/20511)) [CODE, vendor
  sample, as read for Base localization note 1].

## 3. Re-syncing the micro-ROS clock is one call, but only the firmware can make it

### Takeaway
micro-ROS exposes `rmw_uros_sync_session(timeout_ms)` (one NTP-style exchange with the agent) and
`rmw_uros_epoch_nanos()`; firmware typically calls the sync once after connecting -- Yahboom's
evidently does, and so does Husarion's ROSbot firmware. Calling it every 10-30 s from the ROS task
would hold the board-to-Jetson offset to the serial link's asymmetry (~1 ms, est.) instead of letting
it grow ~48 ms an hour. The host cannot trigger it; it can only measure the offset and re-stamp.

Evidence:
- `rmw_uros_epoch_synchronized()`, `rmw_uros_epoch_millis()`, `rmw_uros_epoch_nanos()`: "Returns the
  epoch time ... taking into account the offset computed during the time synchronization";
  `rmw_uros_sync_session(const int timeout_ms)`: "Synchronizes the session time using the NTP
  protocol" ([time_sync.h, humble](https://github.com/micro-ROS/rmw_microxrcedds/blob/humble/rmw_microxrcedds_c/include/rmw_microros/time_sync.h)) [CODE].
- One exchange, `time_offset = ((t0 + t3) - (t1 + t2)) / 2`, no averaging
  ([Base localization note 2, section 2](../Base%20localization%20while%20turning/2%20Time%20stamps%20and%20clock%20sync.md),
  Micro XRCE-DDS `session.c`) [CODE, read there].
- Husarion: `RCCHECK(rmw_uros_sync_session(1000));` at the end of `microros_init()`, and wheel states
  stamped with `rmw_uros_epoch_nanos()` only `if (rmw_uros_epoch_synchronized())`
  ([microros.cpp](https://github.com/husarion/rosbot_ros2_firmware/blob/main/src/microros.cpp),
  [wheels.cpp](https://github.com/husarion/rosbot_ros2_firmware/blob/main/src/wheels.cpp)) [CODE]:
  sync once, as Yahboom's appears to.
- est.: at 2 Mbaud a sync request and reply are a few dozen bytes, well under 1 ms each way; a
  call every 10 s costs nothing measurable. Single-sample offsets jitter with the agent's scheduling;
  taking the best of a few exchanges (smallest round trip) is the usual fix (the Micro XRCE-DDS docs
  suggest averaging; Base localization note 2).

## 4. What each side can fix

### Takeaway
The host re-stamper is worth building first because it fixes the two biggest terms (offset and mean
lag) for everything on the board at once, with no risk. Firmware is the only way to per-point time,
whole revolutions and a clock that does not drift -- worth doing in the same flash as the joint
publisher, never alone.

| timing fault | host re-stamper (no firmware) | firmware change |
|---|---|---|
| board clock offset (55-78 ms) and drift (~48 ms/h) | removed to a few ms by tracking `stamp - arrival` (minimum-delay filter) | removed at the source by periodic `rmw_uros_sync_session` |
| stamp at sending, points ~83 ms older on average | shifted by a measured constant (calibrated by turning, Base localization note 2 section 4) | stamp = first point's arrival time; exact |
| points of one scan span 167 ms; the seam moves | not recoverable from the message (see the idea below) | whole revolutions from `CT_RingStart`, `time_increment` = 250 us |
| skipped bins refilled, fixed 0.541 deg spacing | not recoverable | true angles from each packet |
| two lidars merged with one stamp | per-lidar re-stamp before merging | per-lidar stamps at arrival (already two topics) |

An untested host-side idea (est.): if the shipped firmware does what the sample does, ~16 % of a
revolution (27 of 167 ms, ~100 of 666 bins) was not refreshed since the previous scan, and that
sector sits just ahead of the write head. Bins whose range and intensity are both bit-identical to
the previous scan mark it; its edge gives the write head's angle, and with it each bin's age (angle
behind the head / 360 x 167 ms). When the head is in the 90 deg the board does not publish, the
sector is partly or wholly invisible and the scan cannot be dated this way. Test it on a recorded
bag: a wall seen standing still repeats single values by chance, but a run of tens of exact
repeats in both range and intensity should not happen by chance.
