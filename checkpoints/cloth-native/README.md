# Native SmolVLA cloth pair

Two independent instances of pinned `HuggingFaceVLA/smolvla_libero`, with
separate left/right adaptations of its original native action network.
These are task-adapted foundation VLAs, not zero-shot controllers. The full
pretrained vision/language backbone and native action expert execute at runtime.
There is no teacher, demonstration lookup, or replacement action network in the
deployed control path.

The base model is downloaded separately. This directory contains approximately
10 MB of adapted native tensors per arm, their training reports, exact identities
in `manifest.json`, and upstream attribution/license information.

## Candidate and evidence

- Left: v4, original final expert MLP/norm/output adaptation, including 167
  one-step corrections from its own failed trajectory. The offline teacher
  supplied labels but executed no actions in that replay.
- Right: v3, independently fitted native modules and right-arm normalization;
  the yaw reference is centered on π to avoid an angular discontinuity.
- Both: native input projections inherited from the recorded warm start;
  1,859,552 original native parameters trained in the final stage. Full saved
  model inference was checked against the cached training computation.
- The fixed dashboard scene uses seed **10007**, which was used in adaptation.
  It is a development demonstration, not a held-out benchmark. Training reports
  retain larger errors on separate observations and all model lineage.

**One dashboard-launched fold passed at 1,164 physics ticks.** Both native action
audits passed, and fresh MuJoCo command replay reproduced the final cloth vertices
bitwise and all metrics exactly. See [`demo-result.json`](demo-result.json) for the
model/source hashes, measurements, coordination counters and limitations.
The run used 32 Qwen reviews and 291 durable Spacetime frame commits. It took
25.1 minutes on the local M3 Pro; this is not real-time control or a held-out result.
The right arm released a sleeve early, so the eventual successful geometry does
not establish robust grasp retention.

## Run

Use the existing Apple Silicon simulation host with the `vla` dependencies,
pinned foundation assets, Qwen runtime, and local SpacetimeDB installed. See
[native control](../../docs/native-vla-cloth.md) and
[Studio setup](../../docs/studio.md).

In Studio, open **Tasks → Prepare model pair**, review and release both exact
model revisions, then choose **Run folding task**. The worker loads these pinned
files and runs both full native models. Historical task-controller recordings
retain their original identities.

The native command audit checks recorded model outputs against executed
commands. Fresh-world MuJoCo replay checks their physical outcome, including
successful geometry and equality with the saved metrics. Neither audit reruns
neural inference. The local simulator uses a single-layer garment, idealized
marked-tab clamps and a temporary vacuum fixture; it does not validate real
robot or textile performance.

See `LICENSE` and `NOTICE` for upstream provenance and modifications.
