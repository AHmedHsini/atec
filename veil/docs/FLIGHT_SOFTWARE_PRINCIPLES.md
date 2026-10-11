# Flight-software practices in this Python scout

This is a mapping, not a claim that VEIL is flight software. Python, a garbage collector, and UDP on a laptop do not meet NASA flight standards. The Power of 10 rules (simple control flow, bounded loops, no dynamic allocation after init, limited pointers) are only partly visible here.

| Practice | Where | Evidence |
| --- | --- | --- |
| Bounded queue | `Onboard.max_pending` is 8 | `tests/test_systems.py` queue trim |
| Watchdog and SAFE | perception stall over 1 s stops new hazards | heartbeat test in `test_systems.py` |
| CRC and version | CRC-8, version 1 in the type byte | `tests/test_packet.py`, `tests/test_faults.py` |
| Sequence wrap | RFC 1982 | `tests/test_ground.py` |
| Explicit reject and expire | tracker outcomes | `tests/test_tracks.py` |
| Input checks | packet encode raises, camera JSON must be an object, bad video is counted | `tests/test_faults.py`, `tests/test_ingest.py` |
| Seeded loss | `random.Random(seed)` | acceptance sweep |
| Separation | onboard, radio, ground | `tests/test_systems.py` process run |
| No silent map rollback | older packets ignored | `tests/test_ground.py` |

What Python does not provide: a fixed memory budget, a deadline scheduler, a certified compiler, or a proof that the heap stops growing. The soak note in `python -m veil.bench` reports RSS for a short run only.
