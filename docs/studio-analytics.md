# Relay Studio analytics

Start Studio as described in [the setup guide](studio.md), sign in, and select
**Analytics**, or open `http://127.0.0.1:8770/#analytics`. Both employees and
managers can view aggregate analytics. Model management and releases still
require a manager; analytics does not expose member identities, requests,
uploaded paths, prompts, or model endpoint credentials.

## What the dashboard measures

| Population | Unit | Useful comparison | Boundary |
|---|---|---|---|
| Factory private-routing experiment | Whole simulated task per seed | Embeddings, structured text, no private messages; 12 matched seeds each | Task-trained models; in-memory backend; public scheduling still exists in the control |
| Two-arm cloth experiment | Whole task per seed/condition/channel | Text, embeddings, hybrid with clear/delayed conditions | State-assisted task policies; separate no-message diagnostics are not pooled |
| Learned adapters | Held-out arm decisions or delayed frozen-state decisions | Hidden-peer accuracy and corrupted/no-message controls | Correlated decisions, not independent robot tasks; bounded live trial did not fold the shirt |
| Studio communication | Unique committed session/exchange | End-to-end, database and Qwen p50/p95, packet bytes, timing coverage | Captured features or external reports; no physical action execution |
| Model governance | Current registry model configuration | Draft, released and retired counts | Approval is permission to use a registration, not proof of controller quality |
| Engineering effort | Explicit person-hour scenario | Manual integration assumptions versus Relay integration assumptions | Illustrative estimates; no recorded human-team baseline |

The factory evidence records **12/12** successes for text and embeddings, versus
**2/12** without private messages: an observed **83.3 percentage-point** difference.
The API independently checks pairing identities and hashes before showing that
difference. Neither the experiment nor the dashboard establishes that embeddings
beat structured text, or that Spacetime caused this outcome improvement.

The learned-controller live trial remains visible: zero grasps and 12 missed
hazard holds out of 14. A fast message exchange or good offline decision score
does not establish successful live manipulation.

## Operational telemetry

Spacetime remains the authoritative task/message journal. After exact readback,
Studio records payload-free measurements in the private local
`.relay-lab/studio/telemetry.sqlite3` store. Its key is `(session_id, exchange)`;
retries are not new samples. The log survives Studio restarts. Historical
exchanges whose after-acknowledgement timing was never recorded retain unknown
timings, rather than a zero or a new recovery latency.

The API inspects up to 500 recorded exchanges, plus at most 20 session histories
with 50 results each, deduplicated into a maximum 500-exchange window. The session
inventory is capped at 200. Counts report this coverage rather than lifetime
traffic. Failed uncommitted requests do not have timing samples; failed sessions
and unavailable evidence are separately identified. The archive verification
cohort is explicitly historical and excluded from current traffic.

Percentiles are computed from individual valid samples with linear interpolation.
No percentile is computed by averaging other percentiles. `database_ms` spans
prepare/commit/exact readback; `elapsed_ms` also includes generation, encoding and
optional prediction. `llm_ms` sums generation durations for that exchange. These
are different timing boundaries; they are not independent quantities to add.
Studio's elapsed time excludes waiting for the governance lock, local telemetry
persistence, and HTTP response delivery. It measures execution through the
prediction preview, not client-perceived latency under concurrent load.

Snapshots cache expensive reads for five seconds. Current registry counts are
revalidated for every request. Refresh updates the snapshot; export includes its
generation time, selected cohort/channel/condition, exact visible rows, source
hashes, definitions and scope. Different sections have explicitly different
populations. The raw source artifacts live under the repository's `examples/`
directories; private runtime telemetry is never committed to Git.

## Engineering-effort scenario

The calculator requires explicit robot/task counts and estimated hours. Initial
inputs are examples, not observed savings. It assumes each unordered robot pair
needs custom integration (`n × (n − 1) / 2` pairs); reduce pair-hours for architectures
that already have shared middleware.

```
manual_hours = robots × manual_per_robot_hours
             + pairs × manual_pair_hours + tasks × manual_task_hours
relay_hours  = robots × relay_per_robot_hours + tasks × relay_task_hours
hours_saved  = manual_hours − relay_hours
```

The estimate preserves negative savings. A zero manual baseline yields no
percentage. Hourly rate produces an explicitly modeled USD labor cost, not a
customer ROI claim. Hardware, training, safety validation, deployment and ongoing
maintenance are excluded; there is no inferred elapsed delivery time.

## Reproduction and extension

Use the existing factory and cloth experiment commands from their respective
guides to generate new evidence.

Authenticated API routes:

- `GET /api/analytics`: source manifest, measured aggregates and scenario definitions.
- `POST /api/analytics/estimate`: JSON matching `effort_model.defaults`, with the
  normal session CSRF header; returns calculations and assumptions without saving
  them or changing the workspace.

The run lineage, explicit evaluation configuration, and quality/latency separation
are inspired by [Tinker's evaluation workflow](https://tinker-docs.thinkingmachines.ai/cookbook/eval/benchmarks-guide/).
Tinker is not a robotics baseline here, and no numeric performance comparison with
Tinker is claimed.
