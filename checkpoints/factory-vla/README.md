# Factory VLA v1

An original task-trained vision/language/action model for Relay Works. This is
an 87,182-parameter learned controller, not a pretrained foundation VLA or a
fine-tuned SmolVLA. The same weights run with separate context for each of the
five robots.

The input boundary is two actual MuJoCo RGB camera views, a bounded natural
language skill instruction, the robot's own pose/velocity/gripper state, and an
optional compatible peer latent. The policy produces continuous Cartesian motor
deltas, a yaw delta and a learned gripper class. It receives no simulator object
positions, numeric targets, waypoints or training teacher. The simulator applies
speed/acceleration/reach/collision limits and checks grasp, release and task
success against physical state. Kinematic IK remains the embodiment adapter;
this model does not learn balance, contact-rich assembly or humanoid locomotion.

## Visual assumptions

The sensors mirror original factory surfaces and fixed physical colored
fiducials on cargo, docking beds, robot tools and pallet slots. Both camera views
are rendered independently of the command. This is a simplified inspection
camera scene, not the detailed browser renderer. Fixed tag offsets and camera
calibration are learned from original paired RGB/pose training examples. The
learned visual goal is retained during one instruction while the target is
stationary; motor inference still runs at every active simulation tick.

Vision is a learned pixel classifier, spatial pooling and calibrated 3D readout;
language routing and the motor network are learned. It is deliberately narrow
and should not be represented as open-world visual reasoning or pretrained
cross-embodiment generalization.

## Shared latent communication

A trained 32-dimensional normalized communication head encodes the policy's
visual goal, instruction/embodiment features and local tool/deck pose. Its trained
decoder exposes an approximate pose and intent. During a placement onto a peer
carrier, a matching, fresh, same-checkpoint peer vector is decoded and enters a
trained visual/peer fusion layer before the motor network. Incompatible records
are ignored. Tests assert that a valid peer vector changes the actual motor
output, not just a log entry.

The decoder confidence is aggregate held-out coverage within 65 mm, not a
per-message posterior probability. Decoded text is a model prediction; the
simulator and coordination backend remain responsible for verified facts,
custody and task success. Spacetime stores typed versioned vectors; retrieval
and transport details are documented in the main factory architecture guide.

## Original training and reproducibility

`policy.json` records the checkpoint hash, training seeds, source/data hashes,
step counts and refinement lineage. `evaluation.json` records held-out RGB
calibration, motor/codec error and modality checks for that exact checkpoint.
All data are generated in this repository; no RoboPoly demonstration data,
DuoMind code or external weights are included in this model.

```sh
python -m relay_lab.factory.train_vla collect --scenes 800 --seed 8123
python -m relay_lab.factory.train_vla train --seed 91 --pixel-steps 3000 --control-steps 24000 --codec-steps 5000
python -m relay_lab.factory.train_vla refine --steps 6000 --seed 919
python -m relay_lab.factory.train_vla evaluate --scenes 100 --seed 49271
```

Training uses a simulator-only teacher. The refinement adds observed difficult
lane/clearance junction states and balanced loaded/empty carrier states. The
teacher module is never imported by the runtime policy. On macOS, RGB collection
and evaluation require a process with access to the local graphics session.

The final checkpoint is a validated experimental controller for this factory
layout, not a hardware-ready robot controller. Changing cameras, colors, scale,
robot geometry or allowable tasks requires new training and evaluation.
