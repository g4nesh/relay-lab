# Local observation communication, version 2

The communication checkpoint is a **trained standardized PCA codec**, fitted to
original numeric measurements from this MuJoCo workcell. It is not a visual
encoder, a foundation VLA hidden state, or a learned end-to-end motor protocol.
The two actor networks remain separate from this representation.

The common observation has 23 numeric features:

| Measurement | Dimensions |
| --- | ---: |
| Own end-effector position | 3 |
| Visible local hem, sleeve and shoulder positions | 9 |
| Own attachment present | 1 |
| Locally observed obstacle position, radius and presence | 5 |
| Visibility masks for the three local landmarks | 3 |
| Measured sleeve attachment and hem attachment | 2 |

Missing landmark positions are encoded as zero with a separate visibility mask.
The sleeve/hem bits come from the robot's actual local clamp attachment. They
are never inferred from the requested phase. A sleeve grasp therefore cannot
silently satisfy a peer's requirement for a hem grasp.

The three compared channels receive the same measurement:

- **Embeddings:** an eight-dimensional learned projection; all facts are
  reconstructed approximately at the recipient.
- **Structured text:** all 23 measured features serialized at full numeric
  precision. This is deliberately a strong baseline.
- **Hybrid:** the same learned projection plus exact obstacle, attachment and
  visibility fields. End-effector and cloth coordinates still use the learned
  reconstruction.

All channels preserve the original observation timestamp, image hash, model
identity and sender/recipient scope. The seeded delay/drop schedule depends on
the sender and observation tick, independently of the representation and its
payload size. Newer observations cannot be overwritten by late old packets.
The database acknowledges durable storage before a packet enters the local
delivery queue. Synthetic delay is measured in simulation ticks; synchronous
database wall time is measured separately. Retrieval uses exact packet identity,
not approximate nearest-neighbor search.

## Reproduce the frozen codec

```bash
uv run python scripts/train_cloth_codec.py \
  --calibration checkpoints/cloth-policies/communication-training.json \
  --collection checkpoints/cloth-policies/collection.json \
  --trajectory checkpoints/cloth-policies/communication-trajectory-10004.json \
  --output checkpoints/cloth-policies
```

The initial calibration observations use training seeds 10000–10003. Their old
21-dimensional records do not identify the attachment. Only records with an
explicitly absent grasp can be safely migrated by adding two zero bits. Old
positive-grasp records are excluded and counted; no target is guessed.

The complete original training trajectory uses seed 10004. Its version-2
measurements are recollected by replaying recorded physical actions with the
updated sensor. Final physical vertices must match the source trajectory before
the collector writes the resulting data. The retained source JSON records that
provenance and each arm's measured local attachment.

`communication-training-v2.json` and `.npy` preserve the exact combined fitting
inputs. `communication-fit.json` records source hashes, the training seed set,
migration exclusions, model identity and per-feature reconstruction errors.
Those errors are **training reconstruction**, not held-out task performance.
The pipeline rejects evaluation seeds and does not refit on benchmark episodes.
Changed prior codec files are preserved under `archive/` by their original hash.

Each new physical run copies its exact actor and communication checkpoints into
`RUN/models/`. The auditor uses these immutable copies before mutable workspace
paths:

```bash
uv run python scripts/audit_cloth_run.py RUN --spacetime --require-success
```

Audit success means the recorded local observations, transmitted representations,
causal deliveries, raw network actions, applied holds, durable packets and final
garment geometry agree. It is not independent execution attestation. The trace
does not retain every original RGB frame or the complete physics state at every
tick; neural outputs and cloth speeds remain recorded measurements.
