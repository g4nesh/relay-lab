# Verified learned factory run

This recording completed **three workpieces through all five robots** with actual
local Qwen, the original Factory VLA and real SpacetimeDB. Every cargo item reached
the dispatch pallet after loading, transport, fixture processing and unloading.

```bash
uv run --extra learned relay-lab factory --replay examples/factory-vla/replay.json.gz
```

Open <http://127.0.0.1:8767/?replay=1>. Playback is explicitly recorded, with a
seekable timeline, speed control, robot inspection and the actual latent records.
The live dashboard remains available at <http://127.0.0.1:8767>.

![Completed learned factory task](completed.png)

## Recorded results

| Measurement | Observed value |
|---|---:|
| Workpieces completed | 3 / 3 |
| Verified commands | 27 / 27 |
| Simulation ticks | 3,257 |
| Neural motor actions | 4,390 |
| Qwen generations | 27 |
| Committed learned vectors | 3,370 |
| Actions using learned peer fusion | 338 |
| Failed skills / unconfirmed observations | 0 / 0 |
| Wall time, including Qwen and database work | 155.47 s |
| Spacetime vector refresh, p50 / p95 | 5.46 / 7.63 ms |
| Client peer selection, p50 / p95 | 0.044 / 0.056 ms |

Checkpoint SHA-256:
`a31f6dd477c03ae6848f1a96fd79004582d02bfbcdef84f1fb8825f2e99e1f5b`.
Run ID: `factory-cee4121817c14374`; seed 7; cargo pose jitter 0.02 m.

`audit.json` independently compares all 3,257 committed frames, outcomes,
messages and vectors to the local database and checks every fusion action's
prior eligible source record. It is an evidence-consistency audit, not a signed
attestation that model execution occurred. Model/input/weight ablation tests and
the explicit inference boundary provide complementary checks.

## Embedding comparison and larger workload

`ablation.json` uses the same weights, seed 31, rules planner, memory coordinator,
pose jitter and 6,000-tick budget for both three-workpiece runs. Sharing off
finishes in 3,242 ticks; sharing on finishes in 3,253 ticks with 337 fusion actions.
Both succeed. This demonstrates actual communication use; it does **not** show a
latency or success-rate benefit. Wall timings overlap other local work and are
not isolated benchmarks.

A six-workpiece learned run, seed 42 with sharing enabled, finishes in 6,505 ticks
and 54 commands with 698 fusion actions. Its original 6,000-tick attempt reached
5/6 before exhausting the budget; that result is retained. The application now
defaults to a 10,000-tick run budget. Physical success thresholds and individual
skill timeouts were unchanged.

`retrieval-benchmark.json` is a separate synthetic 80-record transport benchmark;
its timings must not be described as learned task performance or native ANN search.

The live dashboard camera, pause/resume/reset controls and recorded playback
were checked in the browser. The complete Python suite passed **379 tests**,
including actual Spacetime transaction and failure cases. Ruff, TypeScript and
the bundled JavaScript checks passed.

## Evidence files

- `events.jsonl.gz`: complete decisions, raw/applied neural actions, embeddings,
  measured outcomes and committed world states.
- `replay.json.gz`: sampled committed frames, preserving task-stage transitions
  and the final completion; unconfirmed observations are never played as commits.
- `summary.json`: final observed task, model provenance, timings and counters.
- `audit.json`: durable database comparison and causal latent checks.
- `ablation.json`: paired three-workpiece checks and six-workpiece validation.

These are narrow task-trained models using calibrated fiducial cameras,
kinematic robot links and idealized proximity-checked grasps. They are not
foundation VLAs or evidence of humanoid walking or hardware readiness. See
[the model and communication contract](../../docs/factory-vla.md).
