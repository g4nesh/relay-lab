"""Registration and communication probes, never an implicit hardware controller.

The factory coordinator already supports typed peer messages and arbitrary task
metadata. Reuse those reducers, ownership checks and exact-readback barriers;
there is no new database or schema. Model files are stored, never unpickled.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from relay_lab.factory.spacetime import SpacetimeCoordinator
from relay_lab.llm import MODEL_ID, MODEL_REVISION, LocalLLM
from relay_lab.spacetime import _identifier

from .artifacts import ArtifactRegistry

MODES = ("mediated", "orchestrated", "latent", "hybrid")
ROBOTS = ("arm", "mobile", "humanoid", "drone", "custom")
FAMILIES = ("smolvla", "xvla", "openvla", "groot", "custom")
ARMS = ("left", "right")
SOURCES = {
    "smolvla": ("HuggingFaceVLA/smolvla_libero", "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"),
    "xvla": ("lerobot/xvla-folding", "c87d6e8fcf47fb19d6aac8b67d35dd20d74bbc8d"),
}


def bounded(value, label, limit=200, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{label} must be {'0' if empty else '1'}–{limit} characters")
    return value.strip()


def endpoint(value):
    value = bounded(value, "Endpoint", 1000, empty=True)
    if value:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("http", "https", "ws", "wss")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("Use an HTTP(S) or WebSocket URL without embedded credentials")
        _ = parsed.port
    return value


def validate_workspace(value, *, complete=False):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Expected a Relay Studio version 1 workspace")
    agents = value.get("agents")
    if not isinstance(agents, list) or len(agents) > 2 or (complete and len(agents) != 2):
        raise ValueError("This workspace supports exactly two model slots; fill both to connect")
    result = {
        "version": 1,
        "name": bounded(value.get("name"), "Workspace name", 80),
        "objective": bounded(value.get("objective"), "Objective", 1000),
        "mode": value.get("mode"),
        "direction": value.get("direction"),
        "agents": [],
    }
    if result["mode"] not in MODES or result["direction"] not in ("both", "left", "right"):
        raise ValueError("Unknown communication mode or direction")
    seen = set()
    for a in agents:
        if not isinstance(a, dict) or a.get("id") not in ARMS or a["id"] in seen:
            raise ValueError("Each model must occupy a different slot")
        seen.add(a["id"])
        if a.get("robot") not in ROBOTS or a.get("family") not in FAMILIES:
            raise ValueError("Unknown embodiment or model family")
        source = a.get("source")
        if not isinstance(source, dict) or source.get("kind") not in (
            "huggingface",
            "local",
            "upload",
            "endpoint",
        ):
            raise ValueError(
                "Choose a model repository, local weights, upload, or inference endpoint"
            )
        ref = bounded(source.get("ref"), "Model source", 1000)
        if source["kind"] == "huggingface" and not re.fullmatch(r"[\w.-]+/[\w.-]+", ref):
            raise ValueError("A model repository must have the form owner/model")
        if source["kind"] == "upload" and not re.fullmatch(r"[a-f0-9]{32}\.safetensors", ref):
            raise ValueError("Select an uploaded safetensors file")
        if source["kind"] == "endpoint":
            endpoint(ref)
        result["agents"].append(
            {
                "id": a["id"],
                "name": bounded(a.get("name"), "Robot name", 60),
                "robot": a["robot"],
                "family": a["family"],
                "source": {
                    "kind": source["kind"],
                    "ref": ref,
                    "revision": bounded(source.get("revision", ""), "Revision", 128, True),
                },
                "rig": endpoint(a.get("rig", "")),
                "action_contract": bounded(
                    a.get("action_contract", ""), "Action contract", 200, True
                ),
            }
        )
    result["agents"].sort(key=lambda a: ARMS.index(a["id"]))
    return result


def example_workspace():
    return {
        "version": 1,
        "name": "Two-arm workcell",
        "objective": "Coordinate both arms to fold a T-shirt. Hold if either arm sees an obstacle.",
        "mode": "hybrid",
        "direction": "both",
        "agents": [
            {
                "id": arm,
                "name": f"{'Left' if arm == 'left' else 'Right'} arm",
                "robot": "arm",
                "family": family,
                "source": {
                    "kind": "huggingface",
                    "ref": SOURCES[family][0],
                    "revision": SOURCES[family][1],
                },
                "rig": "",
                "action_contract": "Cartesian target xyz (m) + gripper boolean",
            }
            for arm, family in zip(ARMS, SOURCES)
        ],
    }


def matched_pair(workspace):
    return len(workspace["agents"]) == 2 and all(
        a["id"] == arm
        and a["family"] == family
        and a["robot"] == "arm"
        and a["source"]
        == {"kind": "huggingface", "ref": SOURCES[family][0], "revision": SOURCES[family][1]}
        for a, arm, family in zip(workspace["agents"], ARMS, SOURCES)
    )


def captured_report(local):
    """Describe only this robot's recorded view; no access to its peer's state."""
    obs = local["observation"]
    obstacle = obs.get("obstacle")
    if obstacle and obstacle.get("active"):
        hazard = (
            f"An ACTIVE OBSTACLE is visible in my local view at xyz {obstacle['position']} metres."
        )
    elif obstacle:
        hazard = "The obstacle visible in my local view is currently inactive."
    else:
        hazard = (
            "No obstacle is visible in my LOCAL view. I cannot see the other side; "
            "this does not establish that the shared path is clear."
        )
    return (
        f"Robot {obs.get('agent', 'unknown')}. Phase: {local['phase']}. "
        f"Grasped: {'yes' if obs.get('grasped') else 'no'}. {hazard}"
    )


class StudioService:
    def __init__(self, root, *, coordinator_factory=SpacetimeCoordinator, llm=None):
        self.root = Path(root).resolve()
        self.storage = self.root / ".relay-lab" / "studio"
        self.storage.mkdir(parents=True, exist_ok=True)
        self.uploads = self.storage / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.factory = coordinator_factory
        self.llm = llm or LocalLLM(python=self.root / ".venv-llm/bin/python", max_tokens=180)
        self.lock, self.sessions = threading.RLock(), {}
        self.artifacts = ArtifactRegistry(self.root)

    def workspace(self):
        path = self.storage / "workspace.json"
        if path.exists():
            return validate_workspace(json.loads(path.read_text()))
        return {**example_workspace(), "name": "Untitled team", "mode": "mediated", "agents": []}

    def validate(self, workspace, *, complete=False):
        value = validate_workspace(workspace, complete=complete)
        for a in value["agents"]:
            if (
                a["source"]["kind"] == "upload"
                and not (self.uploads / a["source"]["ref"]).is_file()
            ):
                raise ValueError("Uploaded model file is missing; upload it again")
        return value

    def save(self, workspace):
        value = self.validate(workspace)
        with self.lock:
            temporary = self.storage / "workspace.tmp"
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.replace(self.storage / "workspace.json")
        return value

    def capabilities(self):
        adapter = self.artifacts.describe()
        return {
            "backend": "SpacetimeDB",
            "sample_available": adapter["available"],
            "llm_available": (self.root / ".venv-llm/bin/python").is_file(),
            "hardware_execution": False,
            "max_models": 2,
            "adapter": adapter,
            "session_recovery": True,
            "history_window": 50,
        }

    def health(self):
        started = time.perf_counter()
        database = {"available": False, "status": "unavailable", "error": None}
        try:
            coordinator = self.factory()
            coordinator.client._sql("SELECT run_id FROM factory_run LIMIT 1")
            database.update(available=True, status="connected")
            config = getattr(coordinator.client, "config", None)
            if config:
                database.update(server=config.server, database=config.database)
        except Exception as error:  # noqa: BLE001 - expose a health failure, never fake readiness
            database["error"] = f"{type(error).__name__}: {error}"
        database["latency_ms"] = (time.perf_counter() - started) * 1000
        return {
            "database": database,
            "adapter": self.artifacts.describe(),
            "llm": {
                "available": (self.root / ".venv-llm/bin/python").is_file(),
                "loaded": getattr(self.llm, "_process", None) is not None,
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            "hardware_execution": False,
        }

    def diagnostics(self, workspace):
        checks = []
        try:
            workspace = self.validate(workspace, complete=True)
            checks.append(
                {
                    "id": "workspace",
                    "label": "Model registration",
                    "status": "ready",
                    "detail": "Two valid model registrations",
                }
            )
        except (ValueError, TypeError, KeyError) as error:
            checks.append(
                {
                    "id": "workspace",
                    "label": "Model registration",
                    "status": "blocked",
                    "detail": str(error),
                }
            )
            workspace = None
        health = self.health()
        checks.append(
            {
                "id": "database",
                "label": "SpacetimeDB",
                "status": "ready" if health["database"]["available"] else "blocked",
                "detail": "Authenticated read succeeded"
                if health["database"]["available"]
                else health["database"]["error"],
            }
        )
        mode = workspace["mode"] if workspace else None
        matched = bool(workspace and matched_pair(workspace))
        if mode in ("latent", "hybrid"):
            available = matched and health["adapter"]["available"]
            detail = "Pinned SmolVLA and X-VLA pair; bundle integrity verified"
            if not matched:
                detail = (
                    "A learned translation is required for other models, revisions or embodiments"
                )
            elif not available:
                detail = health["adapter"]["error"]
            if available:
                try:
                    self.artifacts.policy(mode)
                    detail += "; trained adapter loaded and extractor identities checked"
                except Exception as error:  # noqa: BLE001 - optional runtime diagnostic
                    available, detail = False, f"{type(error).__name__}: {error}"
            checks.append(
                {
                    "id": "adapter",
                    "label": "Learned adapter",
                    "status": "ready" if available else "blocked",
                    "detail": detail,
                }
            )
        if mode in ("mediated", "orchestrated", "hybrid"):
            checks.append(
                {
                    "id": "llm",
                    "label": "Qwen runtime",
                    "status": "ready" if health["llm"]["available"] else "blocked",
                    "detail": "Local interpreter present; generation is tested on exchange"
                    if health["llm"]["available"]
                    else "Install the isolated .venv-llm runtime",
                }
            )
        checks.append(
            {
                "id": "actuation",
                "label": "Execution boundary",
                "status": "info",
                "detail": "Communication and action previews only. Hardware is inactive.",
            }
        )
        return {
            "ready": not any(c["status"] == "blocked" for c in checks),
            "checks": checks,
            "mode": mode,
            "matched_pair": matched,
            "adapter": self.artifacts.describe(),
            "hardware_execution": False,
        }

    def connect(self, workspace):
        workspace = self.validate(workspace, complete=True)
        if workspace["mode"] in ("latent", "hybrid") and not matched_pair(workspace):
            raise ValueError(
                "Latent mode needs trained sender/receiver adapters. This build supports the pinned SmolVLA + X-VLA example pair only."
            )
        if workspace["mode"] in ("latent", "hybrid"):
            self.artifacts.policy(workspace["mode"])
        run_id = "studio-" + uuid.uuid4().hex[:16]
        coordinator = self.factory()
        state = {
            "tick": 0,
            "success": False,
            "task": "studio-communication",
            "jobs": {"channel": {"job_id": "channel", "stage": "registered"}},
            "workspace": workspace,
            "hardware_execution": False,
            "studio_protocol": 2,
            "created_at": time.time(),
        }
        coordinator.create(
            run_id,
            "studio-worker",
            [{"robot_id": a["id"], "capabilities": ["communicate"]} for a in workspace["agents"]],
            list(state["jobs"].values()),
            [],
            max_ticks=200,
            state=state,
        )
        with self.lock:
            self.sessions[run_id] = {
                "coordinator": coordinator,
                "workspace": workspace,
                "state": state,
                "tick": 0,
                "sequences": {a: 0 for a in ARMS},
                "results": [],
                "requests": {},
                "failed": False,
                "lock": threading.RLock(),
                "recovered": False,
            }
        return {
            "session_id": run_id,
            "backend": "SpacetimeDB",
            "status": "channel_open",
            "hardware_execution": False,
            "models_loaded": False,
        }

    def session(self, session_id):
        _identifier(session_id, "Studio session")
        if session_id not in self.sessions:
            self.recover(session_id)
        return self.sessions[session_id]

    def list_sessions(self):
        coordinator = self.factory()
        rows = coordinator.client._sql(
            "SELECT run_id, tick, status, phase FROM factory_run WHERE worker_id = 'studio-worker' LIMIT 200"
        )
        sessions = []
        for row in rows:
            _identifier(row["run_id"], "Studio session")
            states = coordinator.client._sql(
                f"SELECT state_json FROM factory_run WHERE run_id = '{row['run_id']}'"
            )
            if len(states) != 1:
                continue
            state = json.loads(states[0]["state_json"])
            if state.get("task") != "studio-communication":
                continue
            workspace = state["workspace"]
            sessions.append(
                {
                    "session_id": row["run_id"],
                    "name": workspace["name"],
                    "mode": workspace["mode"],
                    "direction": workspace["direction"],
                    "tick": row["tick"],
                    "status": row["status"],
                    "phase": row["phase"],
                    "resumable": row["phase"] == "ready" and state.get("studio_protocol") == 2,
                    "updated_at": state.get("updated_at", state.get("created_at")),
                }
            )
        sessions.sort(key=lambda s: s["updated_at"] or 0, reverse=True)
        return {"sessions": sessions, "backend": "SpacetimeDB", "limit": 200}

    def recover(self, session_id):
        _identifier(session_id, "Studio session")
        if not re.fullmatch(r"studio-[a-f0-9]{16}", session_id):
            raise ValueError("Unknown Studio session identifier")
        with self.lock:
            if session_id in self.sessions:
                return self.history(session_id)
            coordinator = self.factory()
            coordinator.run_id, coordinator.worker_id = session_id, "studio-worker"
            snapshot = coordinator.snapshot()
            state = snapshot["state"]
            if (
                snapshot["worker_id"] != "studio-worker"
                or state.get("task") != "studio-communication"
            ):
                raise ValueError("This run is not a Studio communication session")
            if snapshot["phase"] != "ready" or snapshot["status"] != "running":
                raise ValueError(
                    "Session is not safely resumable: an incomplete commit or terminal state requires a new channel"
                )
            if state.get("studio_protocol") != 2:
                raise ValueError(
                    "Legacy session has no durable request journal; open a new channel"
                )
            if not 0 <= snapshot["tick"] == snapshot["revision"] <= 200:
                raise ValueError("Invalid Studio recovery revision")
            workspace = self.validate(state["workspace"], complete=True)
            session = {
                "coordinator": coordinator,
                "workspace": workspace,
                "state": state,
                "tick": snapshot["tick"],
                "sequences": {a: 0 for a in ARMS},
                "results": [],
                "requests": {},
                "failed": False,
                "lock": threading.RLock(),
                "recovered": True,
            }
            for revision in range(snapshot["revision"]):
                batch = coordinator._batch(revision)
                if not batch or not batch["result_json"]:
                    raise ValueError("Cannot recover an uncommitted Studio exchange")
                committed = json.loads(batch["result_json"])
                durable = committed["state"].get("studio_exchange")
                if not durable or committed["state"]["tick"] != revision + 1:
                    raise ValueError("Studio exchange journal is incomplete")
                if (
                    committed["state"]["workspace"] != workspace
                    or json.loads(batch["prepare_json"]) != {"commands": [], "messages": []}
                    or not re.fullmatch(r"[a-f0-9]{64}", durable.get("signature", ""))
                ):
                    raise ValueError("Studio exchange recovery contract differs from registration")
                authoritative = self.read_messages(coordinator, session_id, tick=revision + 1)
                if authoritative != sorted(committed["messages"], key=lambda m: m["sender"]):
                    raise ValueError("Studio batch and message journal disagree")
                result = copy.deepcopy(durable["result"])
                result.update(
                    messages=committed["messages"], readback_verified=True, recovered=True
                )
                if result["exchange"] != revision + 1 or result["mode"] != workspace["mode"]:
                    raise ValueError("Studio recovery result has inconsistent scope")
                if durable["request_id"] in session["requests"]:
                    raise ValueError("Duplicate request ID in committed Studio journal")
                for message in committed["messages"]:
                    sender = message["sender"]
                    if message["sequence"] != session["sequences"][sender]:
                        raise ValueError("Studio recovery sequence is inconsistent")
                    session["sequences"][sender] += 1
                session["requests"][durable["request_id"]] = (durable["signature"], result)
                session["results"].append(result)
            if snapshot["revision"]:
                # Obtain a confirmed idempotent reducer ACK for the existing
                # final batch. SQL visibility alone is not our durability bar.
                revision = snapshot["revision"] - 1
                coordinator.client._mutate(
                    "commit_factory_batch",
                    [session_id, "studio-worker", revision, batch["result_json"]],
                    lambda: coordinator._batch(revision),
                    {"result_json": batch["result_json"]},
                )
            # Detect a writer that advanced the run while recovery was reading it.
            if coordinator.snapshot() != snapshot:
                raise ValueError("Studio session changed during recovery; retry")
            self.sessions[session_id] = session
        return self.history(session_id)

    @staticmethod
    def read_messages(coordinator, session_id, *, tick=None, since=0, until=None):
        condition = f"created_tick = {tick}" if tick is not None else f"created_tick > {since}"
        if until is not None:
            condition += f" AND created_tick <= {until}"
        rows = coordinator.client._sql(
            f"SELECT payload_json FROM factory_event WHERE run_id = '{session_id}' AND {condition}"
        )
        messages = [json.loads(r["payload_json"]) for r in rows]
        return sorted(
            (
                m
                for m in messages
                if (
                    m["payload"]["exchange"] == tick
                    if tick is not None
                    else m["payload"]["exchange"] > since
                )
                and (until is None or m["payload"]["exchange"] <= until)
            ),
            key=lambda m: (m["payload"]["exchange"], m["sender"]),
        )

    def history(self, session_id):
        _identifier(session_id, "Studio session")
        if session_id not in self.sessions:
            return self.inspect(session_id)
        session = self.sessions[session_id]
        tick = session["tick"]
        since = max(0, tick - 50)
        messages = self.read_messages(session["coordinator"], session_id, since=since, until=tick)
        return {
            "session_id": session_id,
            "messages": messages,
            "results": copy.deepcopy(session["results"][-50:]),
            "failed": session["failed"],
            "workspace": copy.deepcopy(session["workspace"]),
            "tick": tick,
            "status": "failed" if session["failed"] else "channel_open",
            "recovered": session["recovered"],
            "history_from_tick": since + 1,
            "history_window": 50,
            "resumable": not session["failed"] and tick < 200,
        }

    def inspect(self, session_id):
        """Read old, terminal or interrupted sessions without authorizing a write."""
        if not re.fullmatch(r"studio-[a-f0-9]{16}", session_id):
            raise ValueError("Unknown Studio session identifier")
        coordinator = self.factory()
        coordinator.run_id, coordinator.worker_id = session_id, "studio-worker"
        snapshot = coordinator.snapshot()
        state = snapshot["state"]
        if snapshot["worker_id"] != "studio-worker" or state.get("task") != "studio-communication":
            raise ValueError("This run is not a Studio communication session")
        tick = snapshot["tick"]
        if type(tick) is not int or tick < 0:
            raise ValueError("Invalid stored Studio clock")
        since = max(0, tick - 50)
        return {
            "session_id": session_id,
            "messages": self.read_messages(coordinator, session_id, since=since, until=tick),
            "results": [],
            "failed": snapshot["status"] == "failed",
            "workspace": validate_workspace(state["workspace"], complete=True),
            "tick": tick,
            "status": "inspect_only",
            "recovered": False,
            "resumable": snapshot["phase"] == "ready"
            and snapshot["status"] == "running"
            and state.get("studio_protocol") == 2,
            "phase": snapshot["phase"],
            "database_status": snapshot["status"],
            "history_from_tick": since + 1,
            "history_window": 50,
        }

    def sample(self, condition):
        """Validate cache provenance once; reuse actual held-out local observations."""
        if condition not in ("clear", "left_obstacle", "right_obstacle"):
            raise ValueError("Unknown captured condition")
        rows, sources, vectors, _ = self.artifacts.samples()
        snapshot = next(r["snapshot_id"] for r in rows if r["split"] == "test")
        selected = {
            r["agent"]: (i, r)
            for i, r in enumerate(rows)
            if r["snapshot_id"] == snapshot and r["counterfactual"] == condition
        }
        if set(selected) != set(ARMS):
            raise ValueError("Captured observation pair is incomplete")
        return {
            arm: {
                "features": vectors[i].tolist(),
                "observation": copy.deepcopy(row["observation"]),
                "phase": row["phase"],
                "row_id": row["id"],
                "rgb_index": row["rgb_index"],
                "rgb_sha256": row["rgb_sha256"],
                "observation_sha256": row["observation_sha256"],
                "extractor_id": sources[arm]["model"]["feature_space_id"],
            }
            for arm, (i, row) in selected.items()
        }

    def sample_image(self, condition, arm):
        from PIL import Image

        if arm not in ARMS:
            raise ValueError("Unknown captured camera")
        sample = self.sample(condition)
        _, _, _, rgb = self.artifacts.samples()
        output = io.BytesIO()
        Image.fromarray(rgb["rgb"][sample[arm]["rgb_index"]]).save(output, format="PNG")
        return output.getvalue()

    def exchange(self, session_id, request):
        # One ordered commit at a time. Reads remain available during Qwen inference.
        with self.session(session_id)["lock"]:
            return self._exchange(session_id, request)

    def _exchange(self, session_id, request):
        session = self.session(session_id)
        if session["failed"]:
            raise ValueError("This session stopped after a failed commit. Open a new channel.")
        request_id = bounded(request.get("request_id"), "Request ID", 80)
        signature = hashlib.sha256(
            json.dumps(request, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        previous = session["requests"].get(request_id)
        if previous:
            if previous[0] != signature:
                raise ValueError("A request ID cannot be reused with different observations")
            return previous[1]
        if session["tick"] >= 200:
            raise ValueError("This channel reached its 200-exchange budget; open a new channel")
        workspace, started = session["workspace"], time.perf_counter()
        mode, sample = workspace["mode"], None
        if request.get("source") not in ("captured", "manual"):
            raise ValueError("Choose captured observations or manual worker reports")
        if request.get("source") == "captured":
            if not matched_pair(workspace):
                raise ValueError(
                    "Captured features belong to the exact example checkpoints, not uploaded models"
                )
            sample = self.sample(request.get("condition", "left_obstacle"))
            reports = {a: captured_report(sample[a]) for a in ARMS}
        else:
            reports = request.get("reports")
            if not isinstance(reports, dict) or set(reports) != set(ARMS):
                raise ValueError("Supply a local report for each model")
            reports = {a: bounded(reports[a], "Local report", 1800) for a in ARMS}
        features = sample or request.get("features", {})
        contexts, generated, vectors, decoded, policy = {}, [], {}, {}, None
        allowed = ARMS if workspace["direction"] == "both" else (workspace["direction"],)
        if mode in ("mediated", "orchestrated", "hybrid"):
            if mode == "mediated":
                for sender in allowed:
                    output = self.llm.generate(
                        [
                            {
                                "role": "system",
                                "content": "Translate this robot's observed status into a brief factual peer message. Preserve uncertainty. Do not invent success or coordinates. Return only the message, at most 45 words. /no_think",
                            },
                            {"role": "user", "content": reports[sender]},
                        ]
                    )
                    contexts[sender] = bounded(output.text, "Qwen message", 1800)
                    generated.append(output.to_dict())
            else:
                # Both reports are intentionally visible to the central planner.
                output = self.llm.generate(
                    [
                        {
                            "role": "system",
                            "content": 'You coordinate two robots working on one shared task. Decide their immediate next actions from CURRENT reports. Return ONLY a JSON object with left and right instruction strings. An obstacle or blocked path reported by EITHER robot requires BOTH robots to HOLD NOW. Do not make holding conditional on each robot seeing its own obstacle. Give direct instructions, never if-statements. Preserve uncertainty; do not invent coordinates or claim success. Example: left reports blocked, right reports ready -> {"left":"Hold position; your path is blocked.","right":"Hold position; your teammate\'s path is blocked."}. /no_think',
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"objective": workspace["objective"], "reports": reports}
                            ),
                        },
                    ]
                )
                text = output.text.strip()
                if text.startswith("```"):
                    text = "\n".join(text.splitlines()[1:-1])
                instructions = json.loads(text)
                if not isinstance(instructions, dict) or set(instructions) != set(ARMS):
                    raise ValueError("Qwen did not produce both robot instructions; no fallback")
                contexts = {
                    sender: bounded(
                        instructions[ARMS[1 - ARMS.index(sender)]], "Qwen instruction", 1800
                    )
                    for sender in allowed
                }
                generated.append(output.to_dict())
        if mode in ("latent", "hybrid"):
            import torch

            policy = self.artifacts.policy(mode)
            for sender in allowed:
                if not isinstance(features, dict) or not isinstance(features.get(sender), dict):
                    raise TypeError(
                        "A latent exchange requires native features and extractor_id from each sender"
                    )
                f = features[sender]
                vectors[sender] = policy.encode(sender, f.get("features"), f.get("extractor_id"))
                recipient = ARMS[1 - ARMS.index(sender)]
                with torch.inference_mode():
                    decoded[sender] = policy.network.receivers[recipient](
                        torch.tensor(vectors[sender])[None]
                    )[0].tolist()
        tick = session["tick"] + 1
        messages = []
        for sender in allowed:
            recipient = ARMS[1 - ARMS.index(sender)]
            messages.append(
                {
                    "message_id": f"{sender}-{tick}",
                    "sender": sender,
                    "recipient": recipient,
                    "sequence": session["sequences"][sender],
                    "expires_tick": tick + 8,
                    "priority": 5,
                    "confidence": 1.0,
                    "kind": "belief",
                    "correlation_id": None,
                    "payload": {
                        "exchange": tick,
                        "mode": mode,
                        "context": contexts.get(sender),
                        "vector": vectors.get(sender),
                        "receiver_context": decoded.get(sender),
                        "safety_facts": self.safety_facts(sample[sender]["observation"])
                        if sample and mode == "hybrid"
                        else None,
                        "provenance": "captured foundation features"
                        if sample
                        else "user / external worker report",
                        "adapter": policy.model_info if policy else None,
                        "extractor_id": features[sender].get("extractor_id") if policy else None,
                        "captured_row_id": sample[sender].get("row_id") if sample else None,
                        "rgb_sha256": sample[sender].get("rgb_sha256") if sample else None,
                        "created_at": time.time(),
                        "objective": workspace["objective"],
                        "context_scope": "both local reports"
                        if mode in ("orchestrated", "hybrid")
                        else "sender report only",
                        "generated_by": "Qwen" if generated else "learned adapter",
                    },
                }
            )
        # Persist request identity and generation evidence in the same transaction
        # as the messages. Timings/previews after ACK cannot be part of that
        # transaction; recovery reports them as unavailable rather than rerunning.
        result = {
            "exchange": tick,
            "mode": mode,
            "messages": messages,
            "decisions": {},
            "prediction_error": None,
            "elapsed_ms": None,
            "database_ms": None,
            "llm_ms": sum(g["latency_seconds"] * 1000 for g in generated),
            "bytes": len(json.dumps(messages, separators=(",", ":")).encode()),
            "generations": generated,
            "source": "captured" if sample else "manual",
            "condition": request.get("condition", "left_obstacle") if sample else None,
            "input_reports": reports if generated else {},
            "actions_executed": False,
            "readback_verified": False,
            "recovered": False,
            "preview_status": "not_persisted",
            "adapter": policy.model_info if policy else None,
            "hybrid_scope": "Qwen instructions, latent messages and structured safety facts travel through Spacetime. The learned action preview consumes latents/safety facts; Qwen text is delivered for your runtime adapter."
            if mode == "hybrid"
            else None,
        }
        durable_result = {k: v for k, v in result.items() if k != "messages"}
        state = {
            **session["state"],
            "tick": tick,
            "updated_at": time.time(),
            "studio_exchange": {
                "request_id": request_id,
                "signature": signature,
                "result": durable_result,
            },
        }
        coordinator = session["coordinator"]
        from relay_lab.factory.coordination import canonical

        # Reject an oversized payload before prepare changes the run phase.
        canonical({"state": state, "outcomes": [], "messages": messages})
        db_started = time.perf_counter()
        try:
            coordinator.prepare(session["tick"], [])
            coordinator.commit(session["tick"], state, [], messages=messages)
            delivered = self.read_messages(coordinator, session_id, tick=tick)
            if delivered != sorted(messages, key=lambda m: m["sender"]):
                raise RuntimeError("Message readback differs from the submitted batch")
        except Exception:
            session["failed"] = True
            raise
        db_ms = (time.perf_counter() - db_started) * 1000
        decisions, prediction_error = {}, None
        if sample and policy:
            try:
                decisions = self.predict(policy, sample, delivered, mode, tick)
            except Exception as error:  # noqa: BLE001 - the message commit already succeeded
                # An optional action preview cannot undo an acknowledged commit or
                # leave the local revision behind the authoritative database.
                prediction_error = f"{type(error).__name__}: {error}"
        result.update(
            messages=delivered,
            decisions=decisions,
            prediction_error=prediction_error,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            database_ms=db_ms,
            readback_verified=True,
            preview_status="failed"
            if prediction_error
            else "computed"
            if decisions
            else "not_applicable",
        )
        session["tick"], session["state"] = tick, state
        for sender in allowed:
            session["sequences"][sender] += 1
        session["results"].append(result)
        session["requests"][request_id] = (signature, result)
        return result

    @staticmethod
    def safety_facts(observation):
        from relay_lab.cloth.communication import SAFETY_FIELDS, observation_features

        facts = observation_features(observation)
        return {k: facts[k] for k in SAFETY_FIELDS}

    @staticmethod
    def predict(policy, sample, delivered, mode, current_tick=None):
        # Predict only after durable readback; no physical actions are executed.
        decisions = {}
        if current_tick is None:
            current_tick = max((m["payload"].get("exchange", 0) for m in delivered), default=1)
        if type(current_tick) is not int or current_tick < 1:
            raise ValueError("Invalid current preview clock")
        for recipient in ARMS:
            received = next((m for m in delivered if m["recipient"] == recipient), None)
            belief = None
            if received:
                adapter = received["payload"].get("adapter")
                if (
                    not isinstance(adapter, dict)
                    or adapter.get("checkpoint_sha256") != policy.sha256
                ):
                    raise ValueError("Delivered packet belongs to another trained adapter")
                tick = received["payload"].get("exchange")
                if (
                    type(tick) is not int
                    or not 1 <= tick <= current_tick
                    or received.get("expires_tick", 0) <= current_tick
                ):
                    raise ValueError("Delivered packet is expired or has an invalid clock")
                expected_mode = "hybrid" if mode == "hybrid" else "latent"
                if received["payload"].get("mode") != expected_mode:
                    raise ValueError("Delivered packet belongs to another communication mode")
                belief = {
                    "recipient": recipient,
                    "checkpoint_sha256": policy.sha256,
                    "vector": received["payload"]["vector"],
                    "age_ticks": current_tick - tick,
                    "features": received["payload"]["safety_facts"] if mode == "hybrid" else {},
                }
            local = sample[recipient]
            action, prediction = policy.act(
                local["observation"],
                local["phase"],
                local["features"],
                belief,
                extractor_id=local["extractor_id"],
            )
            decisions[recipient] = {
                "action": action,
                "prediction": prediction,
                "qwen_instruction_executed": False,
            }
        return decisions

    def close(self):
        self.llm.close()
