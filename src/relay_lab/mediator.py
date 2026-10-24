"""Event-driven LLM coordination over grounded robot reports.

The LLM selects semantic subgoals. Deterministic code checks affordances and
turns a selected subgoal into the existing typed policy instruction. These
checks never select a replacement plan: invalid model output fails closed.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict
from math import dist, hypot
from typing import Protocol

from .llm import Generation
from .types import AGENTS, COLORS, AgentId, Decision, Instruction, Observation


class LanguageModel(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> Generation: ...


SYSTEM_PROMPT = """Select the next subgoal for each robot using CURRENT observed facts.
Picker transfers source cubes to relay; packer transfers relay cubes to packing slots.
A robot holding a cube should PLACE it using a legal place action, not wait or pick.
An empty robot should PICK a needed available cube, choosing the earliest needed color in packing_order.
Packer must place each color in the slot matching its zero-based index in packing_order.
Choose wait only when no useful pick or place is available. Wait retracts to a safe home.
Legal actions already enforce relay release, settling, clearance, and robot reachability.
Return ONLY JSON: {"picker":"exact legal action ID","packer":"exact legal action ID","reason":"short plan rationale"}.
Use exactly these three keys. Reason must be under 140 characters. No markdown. /no_think"""


class MediatorError(RuntimeError):
    """No validated LLM plan is available; simulation must not continue."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class LLMMediator:
    def __init__(self, llm: LanguageModel, *, stall_ticks: int = 60, failure_cooldown: int = 15):
        for name, value in (("stall_ticks", stall_ticks), ("failure_cooldown", failure_cooldown)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.llm, self.stall_ticks, self.failure_cooldown = llm, stall_ticks, failure_cooldown
        self.decisions: dict[AgentId, Decision] = {}
        self.last_record: dict | None = None
        self.calls = 0
        self.generation_calls = 0
        self.latency_seconds = 0.0
        self._ready: set[str] = set()
        self._seen_events: deque[str] = deque(maxlen=512)
        self._signature = None
        self._last_tick = -1
        self._last_call_tick = -stall_ticks
        self._progress_tick = 0
        self._distances: dict[str, float] = {}
        self._recent_events: deque[dict] = deque(maxlen=8)
        self._pending_failure = False

    @property
    def stats(self) -> dict:
        return {
            "kind": "local_llm_mediator",
            "decisions": self.calls,
            "generation_calls": self.generation_calls,
            "latency_seconds": round(self.latency_seconds, 6),
            "mean_generation_seconds": round(self.latency_seconds / self.generation_calls, 6)
            if self.generation_calls
            else 0.0,
            "stall_ticks": self.stall_ticks,
            "failure_cooldown": self.failure_cooldown,
            "fallback": False,
        }

    def _events(self, events: dict[AgentId, list[dict]], tick: int) -> bool:
        failed = False
        for agent in AGENTS:
            for event in events.get(agent, []):
                if not isinstance(event, dict) or event.get("agent_id") not in AGENTS:
                    raise ValueError("Mediator requires verified structured execution events")
                event_tick = event.get("tick")
                if (
                    isinstance(event_tick, bool)
                    or not isinstance(event_tick, int)
                    or event_tick > tick
                ):
                    raise ValueError("Mediator received future or invalid execution evidence")
                event_id = event.get("event_id")
                if not isinstance(event_id, str) or not event_id or len(event_id) > 200:
                    raise ValueError("Mediator event is missing a bounded event_id")
                if event_id in self._seen_events:
                    continue
                self._seen_events.append(event_id)
                kind, color = event.get("kind"), event.get("object_id")
                if kind == "holding" and color in COLORS:
                    self._ready.discard(color)
                if (
                    event["agent_id"] == "picker"
                    and kind == "transfer_ready"
                    and color in COLORS
                    and event.get("target_id") == "relay"
                    and event.get("evidence", {}).get("picker_clear") is True
                ):
                    self._ready.add(color)
                if kind in ("holding", "transfer_ready", "completed", "blocked"):
                    self._recent_events.append(
                        {
                            "robot": event["agent_id"],
                            "kind": kind,
                            "object": color,
                            "target": event.get("target_id"),
                            "reason": str(event.get("reason", ""))[:120],
                            "tick": event_tick,
                        }
                    )
                failed |= kind == "blocked"
        return failed

    def _legal_actions(self, observations: dict[AgentId, Observation]) -> dict:
        legal = {
            agent: {"wait": Instruction("wait", "Return home and keep the relay clear.")}
            for agent in AGENTS
        }
        picker, packer = observations["picker"], observations["packer"]
        held = {obs.holding for obs in observations.values() if obs.holding is not None}
        relay_objects = {
            color
            for obs in observations.values()
            for color, position in obs.objects.items()
            if hypot(position[0], position[1]) < 0.085 and position[2] < 0.12 and color not in held
        }
        picker_clear = (
            hypot(picker.ee_position[0], picker.ee_position[1]) >= 0.10
            or picker.ee_position[2] >= 0.18
        )
        if picker.holding:
            if not relay_objects:
                color = picker.holding
                legal["picker"][f"place:{color}:relay"] = Instruction(
                    "place", f"Place {color} on the shared relay pad.", color, "relay"
                )
        elif not relay_objects:
            for color, position in picker.objects.items():
                if color in COLORS and color not in held and position[0] < -0.12:
                    legal["picker"][f"pick:{color}"] = Instruction(
                        "pick", f"Pick the {color} cube from the source.", color
                    )
        if packer.holding:
            color = packer.holding
            for index in range(3):
                if packer.order[index] == color and (
                    index >= len(packer.packed) or packer.packed[index] is None
                ):
                    target = f"slot_{index}"
                    legal["packer"][f"place:{color}:{target}"] = Instruction(
                        "place", f"Place {color} in packing slot {index + 1}.", color, target
                    )
        elif picker_clear:
            for color in sorted(relay_objects & self._ready):
                if color in packer.objects:
                    legal["packer"][f"pick:{color}"] = Instruction(
                        "pick", f"Collect {color} from the relay pad.", color
                    )
        return legal

    def _progress(self, observations: dict[AgentId, Observation], tick: int) -> None:
        for agent, decision in self.decisions.items():
            instruction, obs = decision.instruction, observations[agent]
            target = (
                obs.objects.get(instruction.object_id)
                if instruction.kind == "pick"
                else obs.targets.get(instruction.target_id)
                if instruction.kind == "place"
                else None
            )
            if target is not None:
                distance = dist(obs.ee_position, target)
                if distance < self._distances.get(agent, float("inf")) - 0.005:
                    self._progress_tick = tick
                    self._distances[agent] = distance

    @staticmethod
    def _parse(text: str, legal: dict) -> tuple[dict[AgentId, Decision], str]:
        if not isinstance(text, str) or len(text) > 4000:
            raise MediatorError("Mediator output exceeded its text bound")
        try:
            payload = json.loads(text, object_pairs_hook=_unique_json_object)
        except ValueError as exc:
            raise MediatorError(
                "Mediator must return one valid JSON object with unique keys"
            ) from exc
        if not isinstance(payload, dict) or set(payload) != {"picker", "packer", "reason"}:
            raise MediatorError("Mediator JSON must contain exactly picker, packer, and reason")
        reason = payload["reason"]
        if not isinstance(reason, str) or not 1 <= len(reason) <= 140:
            raise MediatorError("Mediator reason must contain 1 to 140 characters")
        decisions = {}
        for agent in AGENTS:
            action_id = payload[agent]
            if not isinstance(action_id, str) or action_id not in legal[agent]:
                raise MediatorError(f"{agent} selected an unavailable action; use legal_actions")
            decisions[agent] = Decision(legal[agent][action_id])
        return decisions, reason

    def decide(
        self,
        observations: dict[AgentId, Observation],
        execution_events: dict[AgentId, list[dict]] | None = None,
        contexts: dict[AgentId, dict] | None = None,
    ) -> dict[AgentId, Decision]:
        if set(observations) != set(AGENTS):
            raise ValueError("Mediator requires one observation for each robot")
        ticks = {obs.tick for obs in observations.values()}
        if len(ticks) != 1:
            raise ValueError("Mediator observations must share a simulation tick")
        tick = ticks.pop()
        if tick < self._last_tick:
            raise ValueError("Mediator observations must not go backwards")
        for agent, obs in observations.items():
            if obs.agent_id != agent:
                raise ValueError("Mediator observation identity mismatch")
        order = observations["packer"].order
        if order is None or len(order) != 3 or set(order) != set(COLORS):
            raise ValueError("Mediator requires the actual three-color packing order")
        failed = self._events(execution_events or {}, tick)
        self._pending_failure |= failed
        for obs in observations.values():
            if obs.holding:
                self._ready.discard(obs.holding)
        self._last_tick = tick
        signature = (
            *(observations[agent].holding for agent in AGENTS),
            observations["packer"].packed,
            tuple(sorted(self._ready)),
        )
        changed = signature != self._signature
        if changed:
            self._progress_tick = tick
        self._progress(observations, tick)
        trigger = (
            "start"
            if not self.decisions
            else "observed_transition"
            if changed
            else "verified_failure"
            if self._pending_failure and tick - self._last_call_tick >= self.failure_cooldown
            else "sustained_stall"
            if tick - max(self._progress_tick, self._last_call_tick) >= self.stall_ticks
            else None
        )
        if trigger is None:
            return self.decisions.copy()
        legal = self._legal_actions(observations)
        state = {
            "tick": tick,
            "trigger": trigger,
            "packing_order": list(order),
            "packed": list(observations["packer"].packed),
            "relay_ready": sorted(self._ready),
            "robots": {
                agent: {
                    "holding": obs.holding,
                    "position": [round(value, 3) for value in obs.ee_position],
                    "visible_objects": {
                        color: [round(value, 3) for value in position]
                        for color, position in obs.objects.items()
                    },
                    "previous_subgoal": asdict(self.decisions[agent].instruction)
                    if agent in self.decisions
                    else None,
                    "observed_feedback": str((contexts or {}).get(agent, {}).get("text", ""))[:850],
                }
                for agent, obs in observations.items()
            },
            "verified_events": list(self._recent_events),
            "legal_actions": {agent: list(options) for agent, options in legal.items()},
        }
        # Small local models follow explicit observed facts more reliably than
        # nested JSON containing stale instruction prose. The full structured
        # evidence is retained in the record; this projection invents no state.
        facts = [
            "Packing goal: " + ", ".join(f"{color} in slot_{i}" for i, color in enumerate(order)),
            "Observed packed slots: " + json.dumps(state["packed"]),
            "Verified relay cubes ready for PACKER collection: "
            + (", ".join(sorted(self._ready)) or "none"),
        ]
        for agent in AGENTS:
            obs = observations[agent]
            facts.append(f"{agent.upper()} currently holds: {obs.holding or 'nothing'}.")
            visible = []
            for color, position in obs.objects.items():
                if color == obs.holding:
                    continue
                location = (
                    "source"
                    if position[0] < -0.12
                    else "relay"
                    if abs(position[0]) < 0.085
                    else "packing area"
                )
                visible.append(f"{color} at {location}")
            facts.append(f"{agent.upper()} sees: {', '.join(visible) or 'no unheld cubes'}.")
            facts.append(f"{agent.upper()} legal action IDs: " + ", ".join(legal[agent]))
            feedback = state["robots"][agent]["observed_feedback"]
            if feedback:
                facts.append(f"{agent.upper()} observed outcome report: {feedback}")
        facts.append("Select an action for each robot from its own legal action IDs. Return JSON.")
        model_input = "\n".join(facts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": model_input},
        ]
        attempts = []
        self.last_record = {
            "schema": "relay-lab.mediator.v1",
            "call_id": self.calls + 1,
            "tick": tick,
            "trigger": trigger,
            "state": state,
            "model_input": model_input,
            "attempts": attempts,
            "status": "pending",
            "instructions": {},
            "fallback": False,
        }
        for attempt in range(2):
            try:
                generated = self.llm.generate(messages)
            except Exception as exc:
                attempts.append({"error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
                self.last_record.update(status="failed", error="Local LLM unavailable")
                raise MediatorError("Local LLM unavailable; no fallback plan was executed") from exc
            self.generation_calls += 1
            self.latency_seconds += generated.latency_seconds
            attempts.append(generated.to_dict())
            try:
                decisions, reason = self._parse(generated.text, legal)
                # Slot identity is an explicit task invariant, not a proposed plan.
                chosen = decisions["packer"].instruction
                if chosen.kind == "place" and order[int(chosen.target_id[-1])] != chosen.object_id:
                    raise MediatorError("Packer destination violates the specified packing order")
                break
            except MediatorError as exc:
                attempts[-1]["validation_error"] = str(exc)
                if attempt:
                    self.last_record.update(status="failed", error=str(exc))
                    raise MediatorError(
                        "Two LLM plans failed validation; execution stopped"
                    ) from exc
                messages.extend(
                    [
                        {"role": "assistant", "content": generated.text[:4000]},
                        {
                            "role": "user",
                            "content": f"Invalid plan: {exc}. Correct it using the legal actions and exact JSON schema.",
                        },
                    ]
                )
        self.decisions = decisions
        self.calls += 1
        self._last_call_tick = tick
        self._signature = signature
        self._distances.clear()
        self._pending_failure = False
        self.last_record = {
            "schema": "relay-lab.mediator.v1",
            "call_id": self.calls,
            "tick": tick,
            "trigger": trigger,
            "state": state,
            "model_input": model_input,
            "attempts": attempts,
            "reason": reason,
            "instructions": {
                agent: asdict(decision.instruction) for agent, decision in decisions.items()
            },
            "fallback": False,
            "status": "accepted",
        }
        return self.decisions.copy()
