# From the working cloth controllers to adapted foundation VLAs

The completed task-policy benchmark uses the two small state-assisted controllers
in this directory. The released SmolVLA and X-VLA trials are separate experiments.
They establish real model inference, physical action execution and communication
wiring. An unsuccessful bounded transfer trial does not measure either model's
capability in its native environment, and a successful small-controller fold
does not establish a successful foundation-model fold.

## Native conventions and current differences

The official X-VLA Soft-FOLD client supplies a high camera and two wrist cameras,
both arms' pose blocks, domain 5, and absolute 20D pose/gripper chunks. It uses a
15 Hz deployment loop and converts six-dimensional rotations into Euler angles
before commanding its hardware. Its optional relative-to-absolute conversion is
disabled in the published client. Position inputs come directly from the two
robot ROS pose topics; that code supplies no transform into this project's
table frame. Its initial gripper state is a hardware joint position, while later
cached predicted proprioception uses the model's output. Sources: [deployment
client](https://github.com/2toinf/X-VLA/blob/main/evaluation/SoftFold-Agilex/deploy/client_eef6d_xvla.py),
[pose conversions](https://github.com/2toinf/X-VLA/blob/main/evaluation/SoftFold-Agilex/deploy/utils/rotation.py).

Our X-VLA adapter instead supplies one robot's local camera and measured xyz,
masks the absent cameras, zeros unknown peer proprioception, approximates a fixed
downward orientation, and uses binary grasp state as gripper proprioception.
Only the selected arm's predicted xyz/gripper block is executed. The other block
and rotations remain in the raw output log. These are explicit transfer
approximations, not calibrated native Soft-FOLD evaluation.

Our SmolVLA adapter uses the pinned LIBERO checkpoint and its saved preprocessing
and action unnormalization. The same local image currently fills its two camera
slots, and table coordinates stand in for its native robot coordinates. Predicted
translation is interpreted as a bounded delta; rotations are discarded. Its
[checkpoint card](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102)
does not supply a calibration to our custom robot or evidence of cloth training.

## Necessary next work

1. Establish one explicit pose/action convention per model: transforms from each
   MuJoCo arm base, measured tool orientation, gripper opening/sign, camera
   intrinsics/extrinsics and action time horizon. Validate round-trip transforms
   and commanded motion before attributing failures to learned control.
2. Export aligned demonstrations from original training trajectories: local RGB,
   measured own pose/gripper, instruction, delivered peer context, future action
   chunk and timestamps. Seed 10004 has a successful physical reference trajectory
   that can be replayed for this. The 80 shuffled image contexts per arm used by
   the tiny controllers are not an aligned action-chunk dataset.
3. Fit a small adapter separately for each foundation model. A training-episode
   overfit check can validate the gradient path and action scale; it cannot
   establish generalization. Keep development seeds 0–99 and held-out seeds 100–9999
   separate from demonstration seeds 10000 and above.
4. Collect more successful trajectories with varied garment poses, partial views
   and changing conditions, then evaluate the adapted pair on untouched seeds.
   The [official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)
   suggests roughly 50 demonstrations as a starting point for its example tasks;
   that is guidance, not a guarantee for cloth folding.

## Local hardware feasibility

The 18 GB Mac can run the individual released models. Small-adapter training
sequentially, with batch 1, frozen visual/language encoders and cached frozen
features, is a plausible next experiment. Actual backward-step memory and time
must be measured before choosing a training budget. No adaptation training was
launched as part of the frozen communication comparison.

Full float32 AdamW training is a poor default here: approximately 0.879B X-VLA
parameters require about 14.1 GB for weights, gradients and two optimizer moments
before activations, temporary buffers or the operating system. The corresponding
estimate for the 0.605B SmolVLA checkpoint is about 9.7 GB. These are arithmetic
estimates, not measured training peaks, and the models should not be trained
together on this machine.

The installed X-VLA implementation exposes soft prompts and action projections
that can be trained while the large encoders are frozen. Official documentation
also recommends more extensive adaptation when compute permits; restricting
training to small modules is a memory tradeoff, not equivalent performance.
[X-VLA training documentation](https://huggingface.co/docs/lerobot/xvla).
SmolVLA has [official LoRA/PEFT support](https://huggingface.co/docs/lerobot/peft_training).
Its PEFT package is an additional optional dependency and is not required for
the inference probes delivered here.
