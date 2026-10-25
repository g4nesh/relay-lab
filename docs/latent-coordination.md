# Learned communication between different robot models

The learned communication experiment implements a **CoLA-inspired adapter between
frozen SmolVLA and X-VLA features**. It is separate from the existing eight-dimensional
PCA channel. Normalizing two unrelated model embeddings does not give their
coordinates a shared meaning; the receiver needs a learned translation.

This document describes the implementation and its evaluation limits. It does
not report a new folding result. The existing completed folds use the small
state-assisted task controllers. The recorded released-model transfer trial
executes both real VLAs but achieves no grasp or fold. See the
[existing experiment](../examples/cloth-coordination/README.md) and
[foundation adaptation constraints](../checkpoints/cloth-policies/foundation-adaptation.md).

## What the papers establish

**CoLA** freezes the VLA backbone, learns a 64-dimensional message encoder and
directed receiver decoders, and **replaces the native action head**. Its head is
a conditional diffusion U-Net producing ten-step joint chunks, with a separate
gripper classifier. The modules learn jointly by imitation on cached features.
The paper tests Octo and pi0.5 separately; those experiments do not establish a
mixed-family pair. Its stated limits include simulated demonstrations and
synchronous, zero-latency messages. A small Cartesian regression head in this
project is an adaptation, not a reproduction. [Method §§3.2–3.5 and limits §5](https://arxiv.org/html/2610.15337v1#S3)

**VLA-Fabric** retains complete local VLAs and learns three interaction paths:
common semantic context, receiver-specific peer context, and action-stage
interaction. In its pi0.5 realization, peer attention enters the evolving action
expert at each flow step. Training uses the native action objective; the
SingleVLA experiments adapt the language backbone, action heads, and interaction
modules while freezing vision. Teams are homogeneous with fixed roles. This is
more invasive than adding an output correction to two frozen policies. The
latter must not be called a VLA-Fabric reproduction. [Method and appendices A–B](https://arxiv.org/html/2610.23018v1#S3),
[limitations §7](https://arxiv.org/html/2610.23018v1#S7)

**Latent Telepathy** uses one shared, frozen perceptual encoder and trains a
receiver from reward. Its comparison holds the channel and receiver fixed while
changing message content. The reported task has a one-bit hidden hazard; its
physical-camera test evaluates decoding, not closed-loop robot navigation.
It motivates a shared-encoder control and a content-controlled experiment,
but does not solve heterogeneous feature alignment. [Method and limitations](https://arxiv.org/html/2609.23269v2)

**Interlat** inserts a sender's hidden-state sequence into a receiving language
model through a learned attention/projection adapter, with additional training
for latent compression. Its evidence is on ALFWorld and MATH, primarily with two
agents. It requires internal-state access and acknowledges reduced
interpretability. It motivates explicit learned alignment and mismatched-message
tests; it is not evidence for continuous cloth control. [Paper §§3 and 7](https://aclanthology.org/2026.acl-long.1248/)

## Local adapter contract

For arm `i`, let `f_i` be the pooled hidden representation computed by its actual
released VLA from its own camera and instruction. Let `x_i` contain **23 local
simulator measurements and a 16-component phase one-hot vector**. The
implementation is:

```text
message_i = learned_encoder_i(f_i)                   # 64 tanh-bounded numbers
local_i = learned_local_projection_i(f_i)            # 32 numbers
context_j_to_i = learned_decoder_j_to_i(message_j)   # 32 numbers, directed
action_i = learned_head_i(local_i, x_i, context_j_to_i, peer_facts, age, valid)
```

The selected SmolVLA pooled representation has 960 components and the selected
X-VLA checkpoint's pooled representation has 1,024. Each extractor must identify
its layer, pooling, mask, checkpoint revision, and input preprocessing. A common
message width is an interface shape, not an
assertion that the raw latent spaces are aligned. The learned encoders and
decoders provide that alignment for the trained task distribution.

The small MLP head predicts three Cartesian motion values, a gripper logit,
and a coordinated-hold logit. It replaces native action generation in this mode.
Calling a native VLA once to obtain its feature does not make the subsequent
head its original action policy. A frozen feature obtained from a real forward
pass is still useful foundation-model transfer, provided this distinction is
visible in reports.

The feature extractors consume only the arm's own RGB and language. The action
head additionally receives ideal local landmark, gripper and obstacle
measurements, and a teacher-supplied phase instruction. This is a **state-assisted
controller**, not an end-to-end visual folding policy. The privileged original
geometric teacher provides nominal motion and explicit labels to hold both arms
when either obstacle is present. Those labels supervise training and evaluation;
they are not hidden peer measurements supplied to the actor.

| Mode | Information delivered to the receiver |
| --- | --- |
| `none` | No private peer message; its independently trained head uses local inputs. |
| `structured_text` | Canonical JSON text containing the 23 measured peer features, parsed into the head's explicit fact inputs. |
| `embeddings` | A learned 64D message from the peer's actual frozen RGB/language features; no reconstructed simulator facts. |
| `hybrid` | The learned message plus the existing 11 exact safety fields, including obstacle and grasp measurements. |

All four modes execute their local feature extractor and sender path for matched
local preprocessing. Sender weights affect the partner's task loss only when a
latent channel is enabled. The structured-text path uses a parser, not an LLM
rewriting the JSON. The latent payload is transmitted directly into the learned
receiver, rather than decoded into a textual claim about the world.

## Training on the available machine

The dataset preserves each arm's image, measured local state, instruction,
teacher action, episode identity, and observation tick together. It splits by
complete episode before training. The collection is an original action-imitation
diagnostic, not a corpus of successful autonomous folds.

Feature extraction runs one frozen backbone at a time and caches to disk. Exact
image/instruction duplicates are reused only within the same pinned extractor.
The small senders, directed receivers, local projections, and action heads train
jointly on CPU from these caches. An arm's action loss reaches the peer's sender
through the directed channel.
This avoids holding gradients and optimizer states for both foundation models
in 18 GB of unified memory. Measure peak memory and time rather than assuming
a whole-model fine-tuning configuration will fit.

The implemented objective is nominal Cartesian MSE, plus 0.3 times gripper
binary cross-entropy, plus coordinated-hold binary cross-entropy. Four modes
train separate checkpoints with matched initialization and a maximum of 400
epochs; each selects the lowest whole-development-episode loss. The held-out
episode is evaluated after selection. The report includes a separately trained
no-message model and fixed-checkpoint zero/shuffle interventions.

The zero-payload intervention retains a valid packet indicator and can retain
decoder bias. It is intentionally distinct from removing a packet, expiring it,
or retraining without communication. The shuffle intervention cyclically
reassigns peer payloads within the held-out batch; it is a diagnostic of content
use, not a new independent task sample.

**Training uses fresh paired observations only.** It does not train on historical
peer rows, message dropout, or variable age. The age input and transport expiry
checks do not establish learned delay robustness. A future delayed-training
example must use the actual earlier peer observation; attaching an old timestamp
to a current feature would leak future information.

The first evaluation is deliberately a **one-step coordination diagnostic**.
Its split uses original training seeds 10010 and 10011, development seed
11, and held-out seed 104. At sampled physical states, independently vary the
obstacle condition between clear, left-obstructed, and right-obstructed. The
collector does not advance physics between these variants. It explicitly removes
peer-side obstacle geometry and shadows from the receiver's camera rendering,
then checks that the receiver's pixels and local observations match exactly.
This is a controlled observation mask, not natural camera occlusion.
Compare Cartesian error, gripper classification, and correct holding behavior
for the next action. This tests whether the receiver learns to use private peer
information; it does not measure long-horizon folding success. Keep these
results separate from the existing complete-fold benchmark. Multiple snapshots
from one episode remain correlated; one held-out seed is a small diagnostic.

Run the stages separately so the two frozen backbones need not occupy memory
together. These commands create new directories and do not overwrite existing
feature caches or adapter checkpoints:

```bash
.venv/bin/python -m relay_lab.cloth.latent_dataset --workers 4 --output runs/latent-data
.venv/bin/python -m relay_lab.cloth.latent_training cache \
  --dataset runs/latent-data --output runs/latent-features --family smolvla
.venv/bin/python -m relay_lab.cloth.latent_training cache \
  --dataset runs/latent-data --output runs/latent-features --family xvla
.venv/bin/python -m relay_lab.cloth.latent_training train \
  --dataset runs/latent-data --cache runs/latent-features \
  --output runs/latent-adapters --epochs 400
```

MuJoCo rendering needs local graphics access. Both pinned released model caches
must be present. Training writes four checkpoints and `evaluation.json`; it does
not run a complete learned-policy cloth rollout.

## Causal delivery through SpacetimeDB

The database transports and journals messages. It does not align representation
spaces, remove model inference latency, or make stale observations current.
For two fixed peers, retrieve the latest compatible delivered packet; semantic
nearest-neighbor search is not needed to identify the teammate's current state.

Each envelope should bind the run, sender, recipient, observation tick,
sequence, expiry, feature schema, backbone identity, adapter checkpoint, and
payload digest. Validate finite values and dimensions before decoding. Reject
packets from another run or incompatible model pair, future timestamps,
duplicates, regressions, and expired observations. Preserve rejected-packet
reasons for the audit.

The receiver gets packet age and an explicit validity mask. Both the message and
decoded context are masked, so decoder bias does not fabricate a peer state from
missing input. The runtime applies an explicit conservative hold when a message
is missing or expired for communication-enabled modes. The no-message baseline
does not use that hold gate. This is a hand-authored availability guard, not a
learned stale-message policy, and must be separated from learned performance.

Report simulation-time delay separately from wall-clock model latency, database
commit/read latency, and end-to-end decision latency. If physics pauses during
inference or database calls, call it a paused simulation. Do not claim a
real-time distributed controller from these measurements.

The additive database schema is `relay.cloth.learned-latent.v1`. Its reducer
requires the learned adapter kind, exact 960/1024 source dimensions, 64 bounded
message values, and both extractor identities. Existing PCA channels retain
their original schema. The local module was published without clearing data;
the 53 learned-channel and reducer checks passed against it:

```bash
RELAY_SPACETIME_INTEGRATION=1 .venv/bin/pytest -q \
  tests/test_cloth_latent_channel.py tests/test_cloth_latent_spacetime.py
```

Those checks establish transport validation and failure handling. Their small
synthetic training fixtures are deliberately separate from the actual VLA
feature extraction and cloth coordination experiment.

## Comparisons that answer different questions

Keep the initial cloth geometry, obstacles, local views, task instruction,
network random stream, execution horizon, physical score, and attempt count
matched. Use new held-out episodes and retain failures.

| Comparison | Question |
| --- | --- |
| Learned messages versus separately trained no-message model | Does access to communication improve what the team can learn? |
| Delivered messages versus zeroed messages in one fixed checkpoint | Does this trained controller depend on its peer channel? |
| Delivered versus shuffled, equally aged peer messages | Does the content matter beyond a generic additional input? |
| Real frozen VLA features versus zeroed or shuffled features | Does foundation perception contribute beyond local state? |
| Latent versus structured text versus hybrid | Which interface works under this task and resource budget? |
| Age-aware versus age-blind receiver | Does explicit staleness handling help under the same delays? |

For structured text, declare precisely which local facts are available and how
the receiver consumes them. A parser receiving ideal simulator landmarks and
a neural model receiving image features have different information and compute
budgets; that can be a useful system comparison, but not a content-only causal
claim. Also report an equal-byte or equal-rate comparison when claiming a
communication-efficiency advantage. Count the full serialized envelope, not
just `64 * 4` payload bytes.

Keep Qwen's role identical across modes. If its prompt contains peer facts, it
is another communication channel and belongs in the treatment definition.
Latent-only runs cannot receive a hidden exact peer-state side channel through
the planner. Hybrid runs should declare their exact additional fields.

Offline action error, gripper accuracy, intervention sensitivity, and packet
audit results are diagnostics. The primary task result remains a physically
completed fold, with fixture and grippers released, under the existing score.
An action changing after a latent changes proves a computational connection;
it does not by itself prove better coordination.

## What remains distinct from the papers

The local research question is whether **trained translation between different
frozen VLA families remains useful when private observations become stale during
deformable manipulation**. That combination is a testable scope, not a verified
priority claim. A defensible result needs comparison against a well-trained
text interface and no-message controller, with backbone and action-head effects
separated from channel effects.

A faithful VLA-Fabric port is a later, separate implementation: expose compatible
semantic, peer-context, and action-generation insertion points in each model;
train them with the models' own objectives; and compare each interaction path.
The present CoLA-inspired replacement-head experiment must not be renamed to
imply that those internal paths or published results have been reproduced.
