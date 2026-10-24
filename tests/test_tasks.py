"""Task intent remains constrained and preserves each robot's observation boundary."""

import pytest

from relay_lab.tasks import TaskSpec, get_task_adapter, parse_task, task_capabilities


@pytest.mark.parametrize(
    ("text", "order"),
    [
        ("run relay", None),
        ("run relay seed 7", None),
        ("run relay seed 2147483647", None),
        ("pack coral, mint, gold", ("coral", "mint", "gold")),
        ("pack gold then coral then mint", ("gold", "coral", "mint")),
        ("Please pack mint, gold, coral.", ("mint", "gold", "coral")),
        ("task: pack mint then coral then gold", ("mint", "coral", "gold")),
        ("  PACK GOLD,CORAL,MINT  ", ("gold", "coral", "mint")),
    ],
)
def test_supported_intents_generate_only_canonical_task_goals(text, order):
    spec = parse_task(text)
    assert spec is not None
    assert spec.kind == "relay"
    assert spec.order == order
    assert spec.goal == TaskSpec(order=order).goal
    assert "shared relay" in spec.goal


@pytest.mark.parametrize(
    "text",
    [
        "pack coral, mint",
        "pack coral, mint, mint",
        "pack coral, mint, blue",
        "pack coral, mint, gold, coral",
        "pack coral, mint, gold; rm -rf /",
        "pack coral, mint, gold --policy scripted",
        "pack coral, mint, gold and clean the table",
        "pack coral, mint, gold seed 42",
        "pack coral mint gold",
        "clean the table",
        "task: fly a drone",
        "please run relay",
        "task: task: pack coral, mint, gold",
        "@agent1unknown pack coral, mint, gold",
        "run relay seed 2147483648",
        "run relay seed -1",
        "run relay seed 999999999999999",
        " " * 256 + "pack coral, mint, gold",
        "",
    ],
)
def test_unsupported_or_ambiguous_intent_is_not_guessed(text):
    assert parse_task(text) is None


@pytest.mark.parametrize(
    "order",
    [
        ("coral", "mint"),
        ("coral", "mint", "mint"),
        ("coral", "mint", "blue"),
        ("coral", "mint", None),
        ["coral", "mint", "gold"],
    ],
)
def test_task_parameters_are_validated_outside_the_chat_parser(order):
    with pytest.raises(ValueError, match="exactly once"):
        TaskSpec(order=order)


def test_adapter_keeps_private_order_out_of_picker_goal_and_builds_separate_planners():
    adapter = get_task_adapter(TaskSpec(order=("gold", "coral", "mint")))
    assert adapter.spec.order == ("gold", "coral", "mint")
    assert adapter.goal_for("packer") == adapter.spec.goal
    assert "gold" not in adapter.goal_for("picker")
    assert "requested by the packer" in adapter.goal_for("picker")
    planners = adapter.create_planners(require_transfer_ready=True)
    assert set(planners) == {"picker", "packer"}
    assert planners["picker"] is not planners["packer"]
    assert planners["packer"].require_transfer_ready is True
    with pytest.raises(ValueError, match="Unknown agent"):
        adapter.goal_for("other")


def test_capabilities_expose_the_actual_available_task_family():
    description = task_capabilities()
    assert "relay:" in description
    assert "arbitrary tasks" in description.lower()
    assert "local LLM" in description
    assert "SpacetimeDB" in description


def test_unknown_task_kind_fails_early():
    with pytest.raises(ValueError, match="Only the relay"):
        TaskSpec(kind="drone")
