# Released-model integration evidence

These are **bounded transfer probes, not completed foundation-VLA folds**. The
successful cloth comparison in the parent directory uses original, compact,
state-assisted task controllers. Do not pool these experiments.

The joint trial runs SmolVLA LIBERO on the left arm and X-VLA Soft-FOLD on the
right. Both receive their own measured state and local RGB at the same pre-step
tick. Hybrid peer messages cross the actual Spacetime channel; Qwen reviews
local and delivered context. The full instruction is checked against each
model's prepared token tensor immediately before inference. Raw predictions,
bounded actions and subsequent measured robot state are retained.

| Measured property | Joint trial |
| --- | ---: |
| SmolVLA inference calls | 12 |
| X-VLA inference calls | 12 |
| Qwen generations | 11 |
| Delivered Spacetime packets | 24 |
| Complete token round trips | 24/24 |
| Physical grasps | 0 |
| Physical fold | No |

The trial spans 48 physics ticks, or 1.92 simulated seconds; total wall time was
113.9 seconds. Mean CPU inference was 1.87 seconds for SmolVLA and 3.49 seconds
for X-VLA with three Torch threads. Those are local measurements, not an isolated
latency comparison. Inference and database requests pause simulated time.

Inspect [the report](team/report.json), [audit](team/audit.json),
[actual video](team/overview.mp4), and [final frame](team/final.png).
The earlier standalone probes and unsuccessful first X-VLA cache lookup remain
under `standalone/`.

The adapters discard predicted rotations and limit each Cartesian step to
18 mm. X-VLA's native bimanual poses and three-camera input differ from our
partial observation setup; peer proprioception is unknown, and coordinate
transforms are uncalibrated. SmolVLA's LIBERO checkpoint has not been adapted to
this garment task. The outcome does not measure either model's capability in
its native environment.

See [pinned identities and commands](../../../checkpoints/cloth-policies/README.md)
and [calibration/training dependencies](../../../checkpoints/cloth-policies/foundation-adaptation.md).
Foundation weights are downloaded from the public publishers and excluded from Git.
