# Experimental pretrained SmolVLA

This document records the original pretrained-policy investigation. The active
default is now the task-adapted compact Relay VLA plus a local Qwen mediator;
see the README. SmolVLA remains available as an unadapted comparison.

Relay Lab uses **`HuggingFaceVLA/smolvla_libero`**, an Apache-2.0 pretrained
vision-language-action policy from Hugging Face's VLA research organization.
Its Cartesian action interface is a useful starting point for our grippers.
This is a model integration experiment; the checkpoint has not been trained to
solve Relay Lab.

| Item | Pinned choice |
| --- | --- |
| Checkpoint | `HuggingFaceVLA/smolvla_libero` |
| Revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| Parameters | 604,934,176, approximately 605M |
| Download size | Approximately 1.22 GB for the checkpoint repository |
| License | Apache-2.0 |
| Runtime | `lerobot[smolvla]==0.6.1`, Python 3.12 or newer |

The accompanying configuration and tokenizer are from
`HuggingFaceTB/SmolVLM2-500M-Instruct`, revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Its separate model weights are not
needed when loading the complete VLA checkpoint.

The familiar **450M** figure describes the SmolVLA base variant. This LIBERO
checkpoint is larger. The base checkpoint, `lerobot/smolvla_base`, produces six
SO100 joint commands; treating those numbers as XYZ movement would be incorrect.
The selected checkpoint already predicts end-effector movement.

Sources: [model card](https://huggingface.co/HuggingFaceVLA/smolvla_libero),
[checkpoint metadata](https://huggingface.co/api/models/HuggingFaceVLA/smolvla_libero),
[tokenizer metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM2-500M-Instruct),
[base model configuration](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json),
[published runtime](https://pypi.org/project/lerobot/0.6.1/).

## Inputs and actions

| Model field | Meaning |
| --- | --- |
| `observation.images.image` | External camera RGB, three channels, 256 × 256 |
| `observation.images.image2` | Wrist camera RGB, three channels, 256 × 256 |
| `observation.state` | Eight values: world XYZ, axis-angle orientation XYZ, two signed finger positions in metres |
| `task` | Local instruction; checkpoint default is 48 tokens. Relay Lab now uses a 512-token budget for the instruction and execution feedback, rejecting overflow instead of truncating. |
| `action` | Seven values: translation XYZ, rotation XYZ, gripper |

Image tensors use channel-first layout and values in `[0, 1]`. Load the saved
preprocessor and postprocessor: they contain the checkpoint's state and action
statistics. Raw network outputs are normalized values, not simulator commands.
The checkpoint predicts chunks of 50 actions.

LIBERO's controller clips action values to `[-1, 1]`, then scales translation
by **0.05 metres** and rotation by **0.5 radians**. Its gripper convention is
**+1 closes, −1 opens**, the reverse of Relay Lab's gripper command. The state
uses world coordinates. Its two finger positions are approximately
`[+0.04, -0.04]` when fully open and near zero when closed.

Sources: [checkpoint configuration](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/main/config.json),
[state processor](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/processor/env_processor.py),
[controller scales](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json),
[gripper convention](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py),
[world-frame observations](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/robots/single_arm.py).

## Running on this Mac

An Apple M3 Pro with 18 GB unified memory is a realistic inference target.
Float32 model weights alone occupy about 2.42 GB; activations, image processing,
the simulator, and other applications need additional memory. This estimate is
not a measured memory peak or a real-time performance claim. LeRobot provides
an [official MPS inference example](https://github.com/huggingface/lerobot/blob/v0.6.1/examples/tutorial/smolvla/using_smolvla_example.py).

The pinned runtime requires PyTorch `>=2.7,<2.12`, Transformers `>=5.4,<5.6`,
and NumPy `>=2,<2.3`. Inference needs the `smolvla` extra; installing LIBERO or
downloading its demonstration dataset is unnecessary for our custom simulator.

Load the policy with `SmolVLAPolicy.from_pretrained`, and its processors with
`make_pre_post_processors`. Pin the revision for both. Set the processor device
to `mps` as well as the model device. Setting `config.load_vlm_weights=False`
avoids downloading the base VLM weights again when loading a complete policy
checkpoint. Configurations and tokenizer assets still need to be available.

Use `predict_action_chunk` with a separate action queue for each robot. Sharing
model weights saves memory, but sharing `select_action`'s internal queue would
mix the robots' actions. Replanning after fewer than 50 actions trades speed
for more frequent visual feedback.

With recursive feedback enabled, the next chunk also receives grounded text
from the coordinator: requested movement versus observed displacement, gripper
and attachment changes, local instruction events, and delivered peer reports.
Important changes discard queued actions early; routine updates wait for the
next scheduled prediction. Logs retain the exact normalized prompt and its
token count for each chunk. The 512-token limit is identical with feedback on
or off. This is an experimental input-format change, not evidence that the
pretrained checkpoint learned to use execution feedback effectively.

Source: [pinned policy implementation](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py),
[processor loader](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/factory.py),
[dependency declarations](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml).

## What alignment does not solve

Relay Lab has different cameras, objects, grippers, grasp mechanics, and control
timing. It supports translation with a fixed tool orientation, so projecting a
seven-dimensional action onto its available controls discards rotation. Its
motion limit can also shorten a model command.

Any offset between Relay Lab's table-relative coordinates and the model's world
coordinates is an explicit adapter convention, not a learned calibration.
The current adapter adds `(0.35, 0, 0.90)` metres to the picker's XYZ and
`(-0.35, 0, 0.90)` metres to the packer's XYZ. It preserves the world axes, so
output translation requires no inverse rotation. Both tools have a fixed
axis-angle orientation of `[pi, 0, 0]`.

For finger state, the adapter takes the simulator's inner jaw span and supplies
`[span / 2, -span / 2]`. This is an approximation of Panda finger joints, not a
Panda kinematics model. The current geometry has an 86 mm open span and a 60 mm
closed span, including when no parcel is held. Model gripper predictions above
`0.3` close the jaws, below `-0.3` open them, and values in between retain their
state.

LIBERO itself includes multiple table heights: main tables use 0.90 m while
living-room tables use 0.41 m. A downward tool frame is approximately
axis-angle `[pi, 0, 0]`; that convention must agree with the simulator's tool
frame. Supplying plausible numbers does not establish transfer to a new robot.

Sources: [main table](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/problems/libero_tabletop_manipulation.py),
[living-room table](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/problems/libero_living_room_tabletop_manipulation.py).

A successful load and finite model actions prove that inference is connected.
They do not prove that the policy can complete the relay. A failed learned
rollout must remain a failure, without silently switching to scripted actions.

## Adapting to our own task

If pretrained rollouts fail, collect demonstrations in our environment using
the existing scripted controller or teleoperation. Save each robot's allowed
camera images, local instruction, proprioception, and **actually executed**
action. Use the same coordinate and gripper conventions during training and
inference. Keep demonstration seeds separate from evaluation seeds.

Fine-tune the action expert and state projection, then evaluate the learned
policy without script assistance. Mac inference feasibility does not guarantee
comfortable fine-tuning; training memory and speed require a separate check.
No RoboPoly or DuoMind demonstration dataset is required for this path.

## Local measurements

Measured on this machine: Apple M3 Pro, 18 GB unified memory, Python 3.13,
PyTorch 2.11.0, MPS, float32, ten denoising steps, five executed actions per
chunk. The pinned 1,218,047,032-byte checkpoint passed its full SHA-256 check:
`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`.

The initial integration probe is saved in `runs/smolvla-demo/`:

| Measurement | Result |
| --- | --- |
| Model load, weights cached | 12.47 seconds |
| Decisions / model calls | 60 / 24 (12 per robot) |
| First chunk, including GPU warmup | 7.23 seconds |
| Median subsequent chunk | 0.475 seconds |
| Total inference time | 18.92 seconds |
| Whole run, including load and recording | 35.94 seconds |
| End-of-run MPS tensor allocation | 2.57 GiB |
| End-of-run MPS driver allocation | 4.29 GiB |
| Outcome | Step budget reached; 0 of 3 parcels packed |

Memory values are current allocation snapshots, not measured peaks; MPS driver
allocation includes its tensor allocation. Do not add them together. Video plays
at simulated 10 Hz, independent of inference wall time. These timings describe
this one local run, not a general hardware benchmark.

The 120 robot actions were audited against raw predictions, action mapping,
workspace clipping, and recorded end-effector changes. Each robot maintained
its own action queue and random stream. Both messages were delivered. The picker
never approached the requested coral parcel and kept its gripper open. The packer
moved despite its wait instruction and attempted empty grasps. These observations
identify action-policy transfer as the next problem to address; they do not show
a communication benefit or failure. No scripted controller intervened.


The full seed-7 attempt is saved in `runs/smolvla-full/`. It used the same
600-decision budget as the scripted reference, with communication enabled:

| Measurement | Result |
| --- | --- |
| Outcome | Step budget reached; 0 of 3 parcels packed |
| Simulated / wall time | 60.0 / 139.98 seconds |
| Model load, weights cached | 9.96 seconds |
| Model calls | 243: picker 123, packer 120 |
| Median chunk after the first call | 0.455 seconds |
| Total inference time | 114.77 seconds |
| End-of-run MPS tensor / driver allocation | 2.57 / 4.29 GiB |
| Messages sent / delivered | 20 / 20 |

All 1,200 robot actions were finite and within the motion bound, and logged
executed deltas matched the actual position changes. The picker grasped and
released the wrong mint parcel four times; the packer never grasped a parcel.
The requested coral parcel stayed in the source area. The three additional
picker predictions came from instruction changes discarding pending chunks.
`audit.json`, `model.json`, and `steps.jsonl` preserve the measurements and actions.

For a simulator regression check, the current scripted reference in
`runs/lab-reference/` completed this seed in 311 decisions. That shows the task
is achievable by the reference controller; it is not a learned-policy result.
