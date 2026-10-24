"""Move actual, versioned VLA latents through the authoritative database.

Only confirmed post-step embeddings enter peer policy inputs. The bounded local
retrieval cache is refreshed in batches; a failed refresh stops the worker.
"""

from __future__ import annotations

from copy import deepcopy

from .embeddings import SCHEMA


class LatentExchange:
    def __init__(self, world, coordinator, run_id, checkpoint_sha256, *, enabled=True):
        self.world, self.coordinator, self.run_id = world, coordinator, run_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.enabled = enabled
        self.space_id = "factory-vla-v1"
        self.period, self.ttl = 5, 15
        self.latest = {}
        self.deliveries = {}
        self.published = 0
        self._initialized = False

    def initialize(self):
        if self.enabled:
            self.coordinator.configure_embedding_space(self.space_id, self.checkpoint_sha256)
        self._initialized = True

    def collect(self, state, *, terminal=False):
        tick = state["tick"]
        if not self.enabled or not (tick == 1 or tick % self.period == 0 or terminal):
            return []
        observations = self.world.embedding_observations()
        records = []
        for observation in observations:
            if observation.get("tick") != tick:
                raise ValueError("Embedding observation is not from the current physical tick")
            if (
                observation.get("space_id") != self.space_id
                or observation.get("checkpoint_sha256") != self.checkpoint_sha256
            ):
                raise ValueError("Embedding encoder identity differs from the configured space")
            robot = observation["robot_id"]
            records.append(
                {
                    "schema": SCHEMA,
                    "space_id": self.space_id,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "dimension": 32,
                    "modality": "vla-hidden",
                    "embedding_id": f"{robot}-{tick}",
                    "run_id": self.run_id,
                    "robot_id": robot,
                    "tick": tick,
                    "expires_tick": tick + self.ttl,
                    "command_id": observation.get("command_id"),
                    "job_id": observation.get("job_id"),
                    "vector": observation["vector"],
                    "decoder": observation["decoder"],
                }
            )
        return records

    def confirmed(self, records, tick):
        if records:
            self.coordinator.refresh_embeddings(tick=tick)
            self.latest.update({record["robot_id"]: deepcopy(record) for record in records})
            self.published += len(records)

    def deliver(self, tick):
        delivered = {}
        if self.enabled:
            for robot, record in self.latest.items():
                if record["expires_tick"] <= tick:
                    continue
                delivered[robot] = self.coordinator.retrieve_embeddings(
                    robot, record["vector"], tick=tick, limit=4
                )
        self.deliveries = delivered
        self.world.set_peer_embeddings(delivered)
        return delivered

    def semantics(self, tick):
        if not self.enabled:
            return []
        return [
            {
                "robot": robot,
                "tick": record["tick"],
                "prediction": record["decoder"]["text"],
                "confidence": record["decoder"]["confidence"],
                "source": "learned latent decoder; not verified physical evidence",
            }
            for robot, record in self.latest.items()
            if record["expires_tick"] > tick
        ]

    def view(self):
        return {
            "enabled": self.enabled,
            "space_id": self.space_id,
            "dimension": 32,
            "checkpoint_sha256": self.checkpoint_sha256,
            "period_ticks": self.period,
            "ttl_ticks": self.ttl,
            "published": self.published,
            "latest": list(deepcopy(self.latest).values()),
            "deliveries": {
                robot: [
                    {
                        "embedding_id": item["embedding_id"],
                        "robot_id": item["robot_id"],
                        "age_ticks": item["age_ticks"],
                        "cosine_similarity": item["cosine_similarity"],
                    }
                    for item in records
                ]
                for robot, records in self.deliveries.items()
            },
            "metrics": self.coordinator.embedding_metrics
            if self.enabled and self._initialized
            else {},
        }
