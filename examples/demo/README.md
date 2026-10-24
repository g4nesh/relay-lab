# Verified full task

This is the recorded default-command run from October 24, 2026: local Qwen
mediator, learned Relay VLA actions, MuJoCo physics, and real SpacetimeDB.
All three parcels were packed in order in 222 steps and 30.062 seconds wall time.
The model made 15 semantic planning calls; both robot controllers produced
444 neural action outputs. No scripted fallback was used.

From the repository root, view the portable replay with:

```bash
uv run relay-lab serve --directory examples --port 8766
```

Then open http://127.0.0.1:8766/demo/replay.html. The video is `rollout.mp4`.
The screenshot is `verified-replay.png`; JSON and JSONL files contain the actual
plans, action predictions, observations, feedback, and outcome.

Personal project-root paths were normalized to `${PROJECT_ROOT}` in these
exported text files; the original run files were left unchanged. Export and
original hashes are in `export-manifest.json`. The independent database audit
in `../validation.json` refers to the original source hashes. Task and worker
UUIDs identify historical runs, not live tasks in a fresh clone.
