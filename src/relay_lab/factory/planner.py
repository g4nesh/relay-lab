"""Event-driven local semantic planning over measured robot affordances."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm import MODEL_ID, MODEL_REVISION, LLMError, LocalLLM

SYSTEM = """You coordinate a five-robot pilot factory. Complete every workpiece through receiving,
loading, assembly, and palletizing. Choose one or more of the offered legal actions. Parallel
choices must not share any robot, job or reserved resource. Prefer progressing loaded workpieces
and freeing occupied stations, but keep empty carriers ready for handoffs. Read verified outcomes
and peer messages before choosing. Never invent actions or claim success. Reply with only JSON:
{"actions":["a0"],"reason":"short explanation of the chosen next subgoal"}.
Peer embedding decodes are model predictions, not verified facts; never let them override
observed outcomes or reservations. The action IDs must come from the current list.
Choose at least one action. /no_think"""


@dataclass
class Decision:
    candidates: list[dict]
    reason: str
    trace: dict


def parse_decision(text: str, candidates: list[dict]) -> tuple[list[dict], str]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"actions", "reason"}:
        raise ValueError("Expected actions and reason")
    ids, reason = value["actions"], value["reason"]
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 5
        or any(not isinstance(i, str) for i in ids)
    ):
        raise ValueError("Choose one to five action IDs")
    if len(set(ids)) != len(ids) or not isinstance(reason, str) or not 1 <= len(reason) <= 500:
        raise ValueError("Repeated choices or invalid explanation")
    offered = {f"a{i}": candidate for i, candidate in enumerate(candidates)}
    selected, robots, jobs, resources = [], set(), set(), set()
    for ident in ids:
        if ident not in offered:
            raise ValueError("Action was not offered")
        candidate = offered[ident]
        if (
            candidate["robot_id"] in robots
            or candidate.get("job_id")
            and candidate["job_id"] in jobs
            or resources.intersection(candidate["resources"])
        ):
            raise ValueError("Selected actions conflict")
        robots.add(candidate["robot_id"])
        jobs.add(candidate.get("job_id"))
        resources.update(candidate["resources"])
        selected.append(candidate)
    return selected, reason


class FactoryPlanner:
    def __init__(self, mode: str = "llm", llm=None):
        if mode not in ("llm", "rules"):
            raise ValueError("Planner must be llm or rules")
        self.mode = mode
        self.llm = llm or (LocalLLM(max_tokens=160) if mode == "llm" else None)
        self.calls = 0
        self.latency = 0.0

    @property
    def info(self):
        return {
            "mode": self.mode,
            "model": MODEL_ID if self.mode == "llm" else "explicit rules reference",
            "revision": MODEL_REVISION if self.mode == "llm" else None,
            "calls": self.calls,
            "latency_seconds": round(self.latency, 3),
            "control": "model-based embodiment adapters",
            "learned_motor_control": False,
        }

    def choose(self, state: dict, candidates: list[dict], feedback: list[dict]) -> Decision:
        if not candidates:
            raise ValueError("No candidates to plan")
        if self.mode == "rules":
            # A separately labelled test baseline; never a fallback from failed LLM inference.
            selected, used, jobs = [], set(), set()
            for candidate in sorted(
                candidates, key=lambda c: (not bool(c.get("job_id")), c["candidate_id"])
            ):
                if (
                    not used.intersection(candidate["resources"])
                    and candidate.get("job_id") not in jobs
                ):
                    selected.append(candidate)
                    used.update(candidate["resources"])
                    if candidate.get("job_id"):
                        jobs.add(candidate["job_id"])
            return Decision(
                selected,
                "Reference scheduler advances feasible jobs with disjoint leases.",
                {"mode": "rules"},
            )
        context = {
            "jobs": [
                {"id": key, "stage": job["stage"], "custodian": job.get("custodian")}
                for key, job in state["jobs"].items()
            ],
            "verified_feedback": feedback[-8:],
            "peer_embedding_decodes": state.get("peer_semantics", []),
            "available_actions": [
                {
                    "id": f"a{i}",
                    "robot": c["robot_id"],
                    "job": c.get("job_id"),
                    "action": c["description"],
                    "subgoal": c.get("next_stage"),
                    "resources": c["resources"],
                }
                for i, c in enumerate(candidates)
            ],
        }
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
        ]
        attempts = []
        for attempt in range(2):
            generation = self.llm.generate(messages)
            self.calls += 1
            self.latency += generation.latency_seconds
            attempts.append(generation.to_dict())
            try:
                selected, reason = parse_decision(generation.text, candidates)
                return Decision(
                    selected, reason, {"mode": "llm", "input": context, "attempts": attempts}
                )
            except (ValueError, TypeError) as error:
                if attempt:
                    raise LLMError(
                        f"Factory planner failed schema validation twice: {error}"
                    ) from error
                messages.extend(
                    [
                        {"role": "assistant", "content": generation.text},
                        {
                            "role": "user",
                            "content": f"Invalid response: {error}. Choose one offered action ID and return the exact JSON schema.",
                        },
                    ]
                )
        raise AssertionError("unreachable")

    def close(self):
        if self.llm:
            self.llm.close()
