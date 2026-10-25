# Recorded cloth communication comparison

Observed highest completion: embeddings, structured_text, hybrid (100%). Small paired sample; not a general superiority claim.

Planner: **llm**. Backend: **spacetime**. Seeds: [103]. Conditions: ['delayed'].

| Mode | Completed | Mean simulated time | Mean worst-arm hem error | Mean packet size |
| --- | ---: | ---: | ---: | ---: |
| embeddings | 1/1 | 50.28 s | 8.19 cm | 634 bytes |
| structured_text | 1/1 | 45.04 s | 4.71 cm | 1097 bytes |
| hybrid | 1/1 | 50.28 s | 8.19 cm | 870 bytes |

All attempted episodes remain in the denominators. Check `comparison.json` for pairing hashes and `analysis.json` for per-episode holds, codec errors and timings.

Each episode contains an audit of causal message delivery, model identity, raw actions, recorded dynamics, durable Spacetime history and final geometry. An audit is consistency checking, not independent neural/physics execution attestation.

- Descriptive paired pilot; few seeds, one garment/fixture and one fixed intrusion schedule.
- Simulated duration reflects controller/communication behavior, not real-time inference speed.
- Payload bytes are serialized JSON including envelope; exclude HTTP and database readback traffic.
- Local wall-clock timings include shared-machine contention; not an isolated latency contest.
- Setup/runtime failures stay in success denominators; unavailable measurements are null, not zero, and missing counts are reported.
- State-assisted task policies and numeric PCA; no foundation-VLA or real-textile transfer conclusion.
