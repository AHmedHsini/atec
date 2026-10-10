# Limitations

- The camera, rocks, pits, and lens spot are painted by `veil/sim.py`. Nothing here was measured on the Moon.
- Region proposals are brightness thresholds (dark under 60, bright over 175, area at least 28 px). The network only names those blobs. Calling it a detector over the whole image would be wrong.
- Held-out accuracy is on more synthetic patches from the same painter, with a different seed. Similar lighting can make that number optimistic. It is not field performance.
- The lens spot is still a fixed image-space disk. Rejection uses the sliding ground point, not a hardcoded pixel test, but the simulator still paints that disk at one pixel.
- Coordinates assume a flat floor and a known mast height and pitch. A slope or a wrong mast angle moves every fix.
- Stored hazard position is x, y on z = 0. Object height is not estimated.
- Sigma growth (0.05 m/s while the link is down) and the 1.2 s coast limit are design choices. The spec requires an age or uncertainty display, not these constants.
- The 20-byte layout stores version in the top two bits of the type byte. Versions other than 1 are rejected. There is no room for a second header.
- Sequence numbers are 16-bit and compared with RFC 1982 serial arithmetic. Half the number space (32768) is the newer/older boundary.
- Delivery is at-least-once. A lost ack is retried. Duplicates collapse on the ground.
- The loss sweep uses a 4 ms pace. It is not the realtime dashboard clock. The pass gate is the no-blackout 30% row: the minimum delivered hazards across seeds is 4, and the median receipt latency is not the same at every loss rate. 50% and above are informational.
- A hazard is dropped after 12 send attempts or after the pending limit. The default limit is 30 s so the scripted blackouts still deliver. The stress case uses 8 s and a link that never returns, and those hazards are lost on purpose.
- The scout holds while it knows the link is down. Bytes counted as wasted are only sends that ignore that hold.
- PNG playback matches the simulator pixel for pixel. An mp4 from the `mp4v` codec does not. A packet log at 30% loss can still differ between two runs when an ack arrives on a different step.
- The live command uses three processes. A stalled child is terminated by the launcher. The fast tests stay in one process. Cross-process byte identity was measured at 0% loss with a 20 ms ack wait. It is not claimed at 30% loss, where a dropped ack changes the retry log.
- The 16-byte layout described in the protocol document is not implemented.
- The map grid does not pan or zoom. Dragging the lander rebases the displayed distances. Dragging the scout pauses the script.
- `git_revision` is null when git cannot resolve `HEAD`.
