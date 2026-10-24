# Private inspection routing codec

This original 20,907-parameter CNN turns a private 64×64 MuJoCo inspection-camera
crop into a learned, normalized 32-dimensional representation. Its receiver head
predicts route A, route B or an unreadable indicator. It is a task-specific visual
communication codec, not a pretrained model or a replacement for the factory
motor VLA. The existing motor checkpoint is unchanged.

The crop shows a physical industrial indicator with independently randomized
pose, illumination, metal/background colors, status lights and sensor noise.
Only the humanoid inspection worker may observe this camera in the routing task.
The private route value is used by the simulator to render the indicator, and
never enters `RoutingCodec.encode`. Only pixels enter that inference function.

The two communication representations deliberately share perception and decision
quality:

- Embeddings: sender runs `encode(rgb)`; receiver runs `decode(vector)`.
- Text: sender runs the identical `decode(encode(rgb))`, then sends its exact
  structured JSON result, for example `{"route":"A"}`.
- No communication: the receiver must act without the private visual result.

The encoder envelope includes its latent, version/checkpoint IDs and a source RGB
hash. It contains no decoded route. Runtime transport supplies job/command and
observation-tick provenance, expiration and delivery rules. A packet contains one
representation only. The decoder abstains on unreadable crops or a softmax score
below 0.90; the score is not a calibrated probability.

Original data use disjoint seed ranges: 100000+ for 3,072 training crops, 200000+ for
768 validation crops, and 300000+ for768 test crops. The test set contains 512 valid
routes and 256 unreadable cases. `evaluation.json` records exact results and the
checkpoint hash; `verification_rgb.npz` contains 12 real held-out crops for tests
without a graphics context. Larger generated datasets remain in the ignored
`.cache/factory-routing` directory.

```sh
python -m relay_lab.factory.train_routing_codec collect
python -m relay_lab.factory.train_routing_codec train
python -m relay_lab.factory.train_routing_codec evaluate
```

The held-out test classified all 512 routes and 256 unreadable cases correctly;
text and vector decisions agreed on every valid cue. This establishes sensor and
representation parity, not superiority of embeddings. The exact text fact costs
13 payload bytes, while a JSON 32-vector averages about 650 bytes before the common
transport envelope. End-to-end task success and latency must be measured by the
paired factory experiment, with identical observations and execution budgets.
