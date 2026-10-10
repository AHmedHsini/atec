# Testing and acceptance

From the `veil` directory:

```bash
python -m pytest tests -q
python -m veil.accept
```

`python -m veil.accept` checks that NumPy, OpenCV, and `outputs/hazard_net.npz` exist, runs the same tests, runs the loss sweep and the blackout-recovery sweep, and writes `artifacts/acceptance/SUMMARY.txt` plus `artifacts/acceptance/loss_sweep.json`. The process exit code is nonzero if pytest fails, if any seed at 30% loss with no blackout delivers fewer than 4 hazards, or if receipt latency does not change across the loss rates. 50% loss and above are printed and are not pass gates. A failed measurement is left failed.

## Scenarios

Both use seed 26, step 0.05 m, loss 0.3, and an 80-byte burst.

| Name | Frames | dt | Blackout | File |
| --- | --- | --- | --- | --- |
| acceptance-short | 140 | 0.1 s | 0.5–6.0 s | `artifacts/acceptance/acceptance-short.json` |
| acceptance-long | 130 | 0.2 s | 4–12 s | `artifacts/acceptance/acceptance-long.json` |
| acceptance-processes | 140 | 0.1 s | 0.5–6.0 s | `artifacts/acceptance/acceptance-processes.json` |

`acceptance-processes` is the same short drive, with onboard, radio, and ground in separate processes. The other scenario tests stay in one process so the suite stays fast. `python run_mission.py` uses the separate processes unless `--in-process` is set.

The long scenario is the one the live demo uses, without waiting on the clock. The live command writes `artifacts/live/report.json` and does not write the acceptance files.

## Pass rule

A scenario passes only when all of these are true:

- 4 of 4 known hazards delivered, 0 false, 0 missed
- at least one track rejected for `inconsistent_ground`
- mean match error under 0.25 m
- application bytes sent are less than the naive JSON byte count
- application bytes sent equal `packets_sent * 20`

Expired tracks are reported separately and do not satisfy the rejection count.

## Metrics

- Localization error: each delivered hazard is matched once to the nearest same-class truth inside 0.6 m. Mean and max are over those matches.
- Frames per second: mission frames divided by time inside `detect` only. Sleep, radio, and JPEG encoding are outside that timer.
- Bytes: 20-byte payloads, including dropped attempts. UDP and IP headers are not included. `naive_json_bytes` is one JSON object per detection. `float32_bytes` is five float32 values per detection (class, x, y, sigma, timestamp). `float32_confirmed_bytes` is 20 bytes per newly confirmed hazard only. `hazard_bytes`, `pose_bytes`, `heartbeat_bytes`, and `retransmission_bytes` split what was offered. `bytes_per_delivered_hazard` is `hazard_bytes` divided by matched hazards. `wasted_blackout_bytes` counts sends attempted while the link is closed.
- Confirm-to-receipt time: ground receive time minus the simulated time when the track first became confirmed. The loss sweep uses no blackout, so this time can move with the loss rate. The settle loop advances simulated time by `dt` on every retry.
- Retransmissions per hazard: send attempts after the first, averaged over hazard sequences.
- Blackout recovery: at 30% loss, time from the simulated step the link reopens until the first queued hazard is on the map, and until every hazard that was queued at that step is on the map.
- Stress: the link never reopens and `pending_limit_s` is 8 s. `lost_hazards` counts hazards dropped for age or for the 12-attempt cap. The sender holds while the link is down, so wasted bytes stay 0.
- Frame source: a PNG recording of the drive must deliver 4 hazards and 0 false. The mp4 result is recorded beside it.
- Determinism: two in-process runs with the same seed and no loss write the same hex packet log. Two process runs at loss 0 do the same after acks are given 20 ms to land.
- Model accuracy: fraction correct on `generate(80, seed=1007)`, which is not the training seed. The confusion matrix uses the same held-out generator. Both are synthetic.

## How to read a failure

Open the scenario JSON. `passed: false` is the result of that run, not a stale file. `rejected_inconsistent` and `expired_tracks` say whether the lens path was judged or only went quiet. `false_hazards_delivered` means a confirmed packet did not match a known hazard within 0.6 m.

Unit tests cover geometry, packet limits, track outcomes, ground ordering, seeded loss, blackout, ack retention, model loading, the bounded queue, the heartbeat, the watchdog, the PNG source, and the process split. They do not replace the full `run()` scenarios.

## Measured sweep

`python -m veil.accept` wrote these figures. A later run can move a latency by one 0.1 s step. 10% and 30% receipt medians are both 0.0 s because a packet that lands on the first try is delivered in the same simulated step it was confirmed. The p95 values, and the medians from 50% upward, are what change with loss.

| Loss | Delivered mean | Min | Receipt median s | Receipt p95 s | Retries per hazard | Informational |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 4.0 | 4 | 0.0 | 0.005 | 0.2 | no |
| 0.3 | 4.0 | 4 | 0.0 | 0.2 | 1.525 | no |
| 0.5 | 4.0 | 4 | 0.05 | 0.31 | 2.5 | yes |
| 0.7 | 3.9 | 3 | 0.2 | 0.7 | 6.025 | yes |
| 0.9 | 3.5 | 2 | 0.4 | 1.03 | 10.45 | yes |

Recovery at 30% loss, blackout 0.5–6.0 s: first queued hazard median 0.0 s, all queued hazards median 0.1 s, minimum delivered 4.

Stress, link never reopens, pending limit 8 s: 4 hazards lost, 0 delivered, 0 bytes wasted during the blackout.

Long run byte split: hazard 80, pose 880, heartbeat 520, retransmission 1580, sent 3060, float32 of every detection 9020, float32 of confirmed hazards 80, bytes per delivered hazard 20.

PNG sequence: 4 hazards, 0 false, mean error 0.033 m, same as the live simulator, and the PNG pixels match. mp4v: 4 hazards, 0 false, mean error 0.035 m. The 30% packet log is not byte-identical to the simulator run. Process mode: 4/4, 0 false, 1 rejected. Two process logs at 0% loss, 40 frames, were identical (28 packets).
