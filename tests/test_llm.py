import io
import json
import subprocess
import sys

import pytest

from relay_lab.llm import MODEL_ID, MODEL_REVISION, Generation, LLMError, LocalLLM


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"startup_timeout": -1},
        {"max_tokens": 1},
        {"max_tokens": True},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LocalLLM(**kwargs)


def test_missing_runtime_has_actionable_error(tmp_path):
    with pytest.raises(LLMError, match="Install mlx-lm"):
        LocalLLM(tmp_path / "missing").start()


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "hi"}],
        [{"role": "user", "content": 3}],
        [{"role": "user", "content": "x" * 25000}],
    ],
)
def test_invalid_or_oversized_prompt_rejected_before_start(messages):
    with pytest.raises(ValueError):
        LocalLLM().generate(messages)


def test_partial_output_obeys_deadline():
    llm = LocalLLM(timeout=0.1)
    llm._process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(10)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    with pytest.raises(LLMError, match="deadline"):
        llm._read(0.1)
    assert llm._process is None


def test_worker_early_exit_reported_without_hanging():
    llm = LocalLLM()
    llm._process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    with pytest.raises(LLMError, match="exited"):
        llm._read(3)
    assert llm._process is None


def test_generation_checks_worker_identity_and_correlation(monkeypatch):
    class Process:
        stdin = io.BytesIO()

    llm = LocalLLM()
    llm._process = Process()
    monkeypatch.setattr(llm, "start", lambda: None)
    data = {"id": 1, **Generation('{"ok":true}', 0.1, 20, 7).to_dict()}
    monkeypatch.setattr(llm, "_read", lambda timeout: data.copy())
    result = llm.generate([{"role": "user", "content": "Return JSON"}])
    assert result.model_id == MODEL_ID and result.revision == MODEL_REVISION
    assert json.loads(llm._process.stdin.getvalue())["max_tokens"] == 160
    llm._process = None


def test_mismatched_model_metadata_rejected(monkeypatch):
    class Process:
        stdin = io.BytesIO()

    llm = LocalLLM()
    llm._process = Process()
    monkeypatch.setattr(llm, "start", lambda: None)
    data = {"id": 1, **Generation("answer", 0.1, 20, 7, model_id="other").to_dict()}
    monkeypatch.setattr(llm, "_read", lambda timeout: data)
    with pytest.raises(LLMError, match="metadata"):
        llm.generate([{"role": "user", "content": "test"}])
    llm._process = None
