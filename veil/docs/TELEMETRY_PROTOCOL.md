# Telemetry protocol

Layout id: magic byte `0xC7`. Protocol version is 1. It lives in the top two bits of byte 1, so the packet stays 20 bytes: `byte1 = (version << 6) | type`. `decode` rejects any version other than 1. Little-endian.

## Layout

| Offset | Size | Type | Name | Unit and range |
| --- | --- | --- | --- | --- |
| 0 | 1 | u8 | magic | Must be `0xC7` |
| 1 | 1 | u8 | version and type | Top 2 bits are the version (now 1). Low 6 bits are the type: 1 hazard, 2 pose, 3 heartbeat. |
| 2 | 2 | u16 | sequence | 0–65535. Order uses 16-bit serial arithmetic (RFC 1982): a candidate is newer when `(candidate - current) mod 65536 < 32768`. Sequence 0 is newer than 65535. An equal sequence is a duplicate unless the timestamp is newer, which is how a retry refreshes a fix. |
| 4 | 4 | u32 | t_ms | Mission milliseconds, 0–4294967295. On a heartbeat this is the uptime |
| 8 | 2 | u16 | track_id | 0–65535. Pose uses 0. Heartbeat uses the count of confirmed tracks |
| 10 | 1 | u8 | class | Hazard: 1 rock, 2 pit. Pose and heartbeat: 0. Other hazard classes are rejected |
| 11 | 1 | u8 | flags | Hazard and pose write 0. Heartbeat writes the mode: 0 RUN, 1 SAFE |
| 12 | 1 | u8 | agreement | Hazard: 0–100, share of the geometric window that agreed. Heartbeat: queue depth, capped at 100 |
| 13 | 2 | i16 | x_cm | Centimeters, −32768–32767 (−327.68 m to 327.67 m) |
| 15 | 2 | i16 | y_cm | Centimeters, same range |
| 17 | 1 | u8 | sigma_cm | Centimeters, 0–255 (0–2.55 m) |
| 18 | 1 | u8 | radius_cm | Centimeters, 0–255 (0–2.55 m) |
| 19 | 1 | u8 | crc8 | CRC-8 over bytes 0–18. Polynomial `0x07`, initial value `0x00`, not reflected, xorout `0x00`. |

`encode` raises `ValueError` when a value does not fit. It does not clamp and it does not wrap. `decode` returns `None` for the wrong length, a bad magic, a bad CRC, a version other than 1, an unknown type, or a hazard whose class is not rock or pit. The ground map never sees those packets.

## Checksum

CRC-8 over bytes 0–18, polynomial `0x07`, initial value 0, not reflected, xorout 0. A swap of two different body bytes leaves an XOR checksum unchanged and is rejected by this CRC. `tests/test_packet.py` builds that case.

## Example

`encode(1, 4, 1500, 2, 2, 0, 80, -1.25, 5.0, 0.12, 0.4)` is

```text
c7 41 0400 dc050000 0200 02 00 50 83ff f401 0c 28 87
```

Byte 1 is `0x41`: version 1 in the top two bits, type 1 (hazard) in the low six. Decoded: sequence 4, t = 1500 ms, track 2, pit, agreement 80, x = −1.25 m, y = 5.00 m, sigma = 0.12 m, radius = 0.40 m.

## Delivery

The sender keeps one outstanding packet per track. The sequence stays the same until the 2-byte little-endian ack arrives. The payload may be rewritten with a later timestamp before that ack. The ground map applies a same-sequence packet only when the timestamp is newer. A later sequence, including the wrap from 65535 to 0, replaces the fix when its timestamp is not older. An older sequence is dropped.

This is at-least-once transmission with receiver-side collapse on track id plus 16-bit serial order. It is not exactly-once. A lost ack causes another copy. Sequence comparison uses RFC 1982: `(candidate - current) mod 65536 < 32768` means candidate is newer.

Pose priority is 0.01. Hazards sort higher. `packets(budget)` stops at the 80-byte burst, four packets. Bytes counted in the report are the 20-byte payloads, including attempts dropped by the loss model, and not UDP or IP headers.

Two uncompressed baselines are counted on the scout, neither is sent:

- naive JSON, one object per detection
- five little-endian float32 values per detection: class code, x, y, sigma (0 until the track agrees), and timestamp, 20 bytes each

The 20-byte packet carries more than that bare float32 record. In the same 20 bytes it also has a magic, a version, a type, a sequence, a track id, flags, an agreement score, a radius, and a CRC-8. A third baseline counts five float32 values only for hazards the tracker has confirmed, 20 bytes each, so the comparison can separate the two savings. On the long acceptance run, every detection was 9020 float32 bytes, the four confirmed hazards were 80 bytes either as float32 or as first-send hazard packets, and the radio total was 3060 bytes. The large saving against "send every detection" is onboard confirmation. Packing does not shrink a confirmed hazard below 20 bytes. Pose, heartbeat, and retransmission bytes are why the radio total is larger than the confirmed-only baseline.

The scout does not transmit while it knows the link is down. `wasted_blackout_bytes` counts a send that is attempted anyway. On the normal path that field stays 0. A direct `link.send` during a blackout still increments it and does not draw the loss generator.

Each hazard leaves the queue after 12 transmission attempts or after `pending_limit_s` (30 s by default, 8 s in the closed-link stress run) and is not queued again. The sender does not retry forever. Pose and heartbeat slots are replaced in place and are not counted as lost hazards.

The pending queue holds at most 8 slots. When it is over that, the lowest-value slot is dropped. Heartbeats are worth 0.001, poses 0.01, and hazards more than either, so a hazard is kept and a pose is dropped first.

## Heartbeat and link state

Type 3 uses the same 20 bytes. `flags` is the mode (0 RUN, 1 SAFE). `agreement` is the queue depth. `track_id` is the confirmed-track count. `t_ms` is the uptime in milliseconds. `x` and `y` are the scout. Class is 0. The ground does not turn a heartbeat into a hazard.

Ground link state uses the simulated time since the last accepted heartbeat:

- UP when that age is under 1.0 s
- DEGRADED when the age is under 3.0 s
- DOWN otherwise, including when no heartbeat has arrived

## Watchdog

If `observe` has not run for more than 1.0 simulated second, the onboard computer sets SAFE. Heartbeats continue and carry that mode. New confirmed hazards are not queued. The next `observe` sets RUN and confirmation resumes.

## Smaller packet

A 16-byte layout is possible: a 16-bit time delta instead of the absolute millisecond counter, class and flags packed into one byte, and no agreement byte. That drops the confirmation score the map displays, and the time delta wraps every 65.536 s. The 20-byte packet stays the only format on the wire. A compact encoder was not added, so there is no second measured size.

## What the ground will not do

- Create a hazard from a failed checksum, a short buffer, or an unknown class.
- Move the scout from a pose older than the newest accepted pose.
- Treat a pose as a hazard.
