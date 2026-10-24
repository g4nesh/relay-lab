"""Private visual beliefs cross only the committed, job-scoped semantic channel.

The sender sees a private camera crop once at receiving. The receiver gets
either its learned latent or an exact text rendering of the same prediction.
The scheduler never sees either. A missing packet uses the declared A prior;
it never asks the simulator which destination is correct.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy


class RoutingExchange:
    def __init__(self, world, coordinator, run_id, *, mode, codec=None):
        if mode not in {"embeddings", "text", "none"}:
            raise ValueError("Communication must be embeddings, text, or none")
        if codec is None:
            from .routing_codec import RoutingCodec

            codec = RoutingCodec()
        self.world, self.coordinator, self.run_id = world, coordinator, run_id
        self.mode, self.codec = mode, codec
        self.period, self.ttl = 5, 15
        self.memories, self.decisions = {}, []
        self.published = 0
        self.encode_ms, self.decode_ms = [], []
        self._initialized = False
        self._completed = set()

    def initialize(self):
        info = self.codec.model_info
        self.coordinator.configure_semantic_channel(
            self.mode,
            info["checkpoint_sha256"],
            space_id=info["space_id"],
            period_ticks=self.period,
            ttl_ticks=self.ttl,
        )
        self._initialized = True

    def observe(self, commands, tick):
        """Called after reservation authorization and before the physical step."""
        for command in commands:
            job = command.get("job_id")
            if command["robot_id"] != "humanoid-01" or not job or job in self.memories:
                continue
            rgb = self.world.inspect_private_cue("humanoid-01", job)
            start = time.perf_counter()
            encoded = self.codec.encode(rgb)
            self.encode_ms.append((time.perf_counter() - start) * 1000)
            info = self.codec.model_info
            if (
                encoded.get("space_id") != info["space_id"]
                or encoded.get("checkpoint_sha256") != info["checkpoint_sha256"]
            ):
                raise ValueError("Private encoder identity differs from the configured codec")
            # No-communication still has identical local perception. Its memory
            # stays private and never enters the receiver or central planner.
            content = {"vector": encoded["vector"]}
            if self.mode == "text":
                start = time.perf_counter()
                prediction = self.codec.decode(encoded["vector"])
                self.decode_ms.append((time.perf_counter() - start) * 1000)
                content = {
                    "text": json.dumps({"route": prediction["route"]}, separators=(",", ":"))
                }
            self.memories[job] = {
                "job_id": job,
                "observation_tick": tick,
                "source_command_id": command["command_id"],
                "source_frame_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                **content,
            }

    def collect(self, state):
        tick = state["tick"]
        if self.mode == "none":
            return []
        info = self.codec.model_info
        return [
            {
                "schema": "relay.factory.semantic.v1",
                "mode": self.mode,
                "run_id": self.run_id,
                "sender": "humanoid-01",
                "recipient": "gantry-01",
                "topic": "dispatch_route",
                "packet_id": f"humanoid-01-{job}-{tick}",
                "emitted_tick": tick,
                "expires_tick": tick + self.ttl,
                "space_id": info["space_id"],
                "checkpoint_sha256": info["checkpoint_sha256"],
                **deepcopy(memory),
            }
            for job, memory in self.memories.items()
            if state["jobs"][job]["stage"] != "completed"
            and (tick == memory["observation_tick"] + 1 or tick % self.period == 0)
        ]

    def confirmed(self, packets, state):
        # Reads the committed DB rows. Sender-local memory is never delivered.
        completed = {key for key, job in state["jobs"].items() if job["stage"] == "completed"}
        if packets or completed != self._completed:
            self.coordinator.refresh_semantic_packets(state["tick"])
        self._completed = completed
        self.published += len(packets)

    def decide(self, job_id, index, tick):
        packet = (
            self.coordinator.retrieve_semantic_packet(
                recipient="gantry-01", job_id=job_id, tick=tick
            )
            if self.mode != "none"
            else None
        )
        route, confidence = "A", None
        source = "fixed A prior; no peer evidence"
        if packet is not None:
            start = time.perf_counter()
            if self.mode == "embeddings":
                predicted = self.codec.decode(packet["vector"])
                route, confidence = predicted["route"], predicted["confidence"]
            elif self.mode == "text":
                value = json.loads(packet["text"])
                if set(value) != {"route"} or value["route"] not in {"A", "B"}:
                    raise ValueError("Invalid learned text route prediction")
                route = value["route"]
            else:
                raise ValueError("No-communication receiver obtained a packet")
            self.decode_ms.append((time.perf_counter() - start) * 1000)
            source = "committed peer latent" if self.mode == "embeddings" else "committed peer text"
        decision = {
            "job_id": job_id,
            "robot_id": "gantry-01",
            "tick": tick,
            "route": route,
            "pallet_slot": 2 * index + int(route == "B"),
            "packet_id": packet["packet_id"] if packet else None,
            "observation_tick": packet["observation_tick"] if packet else None,
            "emitted_tick": packet["emitted_tick"] if packet else None,
            "transport_age_ticks": tick - packet["emitted_tick"] if packet else None,
            "observation_age_ticks": tick - packet["observation_tick"] if packet else None,
            "source_frame_sha256": packet["source_frame_sha256"] if packet else None,
            "confidence": confidence,
            "source": source,
        }
        self.decisions.append(decision)
        return deepcopy(decision)

    def view(self):
        return {
            "task": "dispatch-routing",
            "mode": self.mode,
            "model": self.codec.model_info,
            "period_ticks": self.period,
            "ttl_ticks": self.ttl,
            "published": self.published,
            "local_observations": len(self.memories),
            "observation_provenance": [
                {
                    key: memory[key]
                    for key in (
                        "job_id",
                        "observation_tick",
                        "source_command_id",
                        "source_frame_sha256",
                    )
                }
                for memory in self.memories.values()
            ],
            "decisions": deepcopy(self.decisions),
            "encode_ms": list(self.encode_ms),
            "decode_ms": list(self.decode_ms),
            "metrics": self.coordinator.semantic_metrics if self._initialized else {},
            "control_plane": "Public stage, custody, safety leases and handoff ACKs in every mode",
            "decision_boundary": "Qwen schedules public skills; only the gantry's local route adapter reads peer packets",
            "blind_prior": "A for every job; independent of hidden labels and seed",
        }
