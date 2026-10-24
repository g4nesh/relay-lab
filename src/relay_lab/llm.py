"""A pinned local language model in an isolated, persistent MLX process.

The parent needs only the standard library. MLX and its transformers dependency
live in .venv-llm, separately from the VLA runtime. No network inference service,
API key, fallback planner, or arbitrary tool execution is involved.
"""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

MODEL_ID = "Qwen/Qwen3-1.7B-MLX-4bit"
MODEL_REVISION = "21457c6f51ed54a7c16e988c0844db973815c137"
MAX_INPUT_BYTES = 24000
MAX_RESPONSE_BYTES = 32000


class LLMError(RuntimeError):
    """The actual model failed; callers must stop rather than invent a plan."""


@dataclass(frozen=True)
class Generation:
    text: str
    latency_seconds: float
    prompt_tokens: int
    generation_tokens: int
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION

    def to_dict(self) -> dict:
        return asdict(self)


class LocalLLM:
    def __init__(
        self,
        python: str | Path = ".venv-llm/bin/python",
        *,
        timeout: float = 45,
        startup_timeout: float = 180,
        max_tokens: int = 160,
    ) -> None:
        for name, value in (("timeout", timeout), ("startup_timeout", startup_timeout)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 16 <= max_tokens <= 512
        ):
            raise ValueError("max_tokens must be an integer between 16 and 512")
        self.python = Path(python).absolute()
        self.timeout, self.startup_timeout, self.max_tokens = timeout, startup_timeout, max_tokens
        self._process: subprocess.Popen | None = None
        self._stderr = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._buffer = b""

    def _read(self, timeout: float) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise LLMError("Local LLM is not running")
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer and len(self._buffer) <= MAX_RESPONSE_BYTES:
            remaining = max(0, deadline - time.monotonic())
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                self.close()
                raise LLMError(f"Local LLM exceeded its {timeout:g}-second deadline")
            part = os.read(process.stdout.fileno(), 4096)
            if not part:
                break
            self._buffer += part
        if b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
        else:
            line, self._buffer = b"", b""
        if not line or len(line) > MAX_RESPONSE_BYTES:
            detail = ""
            if self._stderr:
                self._stderr.seek(0, os.SEEK_END)
                length = self._stderr.tell()
                self._stderr.seek(max(0, length - 1500))
                detail = self._stderr.read().decode("utf-8", errors="replace")
            self.close()
            raise LLMError(f"Local LLM exited or exceeded its response bound: {detail}")
        try:
            response = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            self.close()
            raise LLMError("Local LLM returned invalid IPC data") from exc
        if not isinstance(response, dict):
            raise LLMError("Local LLM returned a non-object response")
        if "error" in response:
            raise LLMError(f"Local LLM failed: {str(response['error'])[:1000]}")
        return response

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.python.is_file():
            raise LLMError(
                f"Missing local LLM interpreter {self.python}. Install mlx-lm==0.32.2 in .venv-llm."
            )
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(Path(".cache/huggingface").absolute()))
        env["TOKENIZERS_PARALLELISM"] = "false"
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - lifetime matches the worker
        try:
            self._process = subprocess.Popen(
                [str(self.python), "-u", str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=env,
                bufsize=0,
            )
            ready = self._read(self.startup_timeout)
            if ready != {"ready": True, "model_id": MODEL_ID, "revision": MODEL_REVISION}:
                raise LLMError("Local LLM startup identity mismatch")
        except BaseException:
            self.close()
            raise

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
            raise ValueError("messages must contain between one and eight messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str)
            ):
                raise ValueError("Invalid local LLM chat message")
        encoded = json.dumps(messages, ensure_ascii=False).encode()
        if len(encoded) > MAX_INPUT_BYTES:
            raise ValueError("Local LLM prompt exceeds the byte limit")
        with self._lock:
            self.start()
            self._next_id += 1
            request_id = self._next_id
            request = {
                "id": request_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
            }
            try:
                self._process.stdin.write((json.dumps(request) + "\n").encode())
                self._process.stdin.flush()
            except (OSError, BrokenPipeError) as exc:
                self.close()
                raise LLMError("Local LLM worker disconnected") from exc
            response = self._read(self.timeout + 1)
            if response.pop("id", None) != request_id:
                self.close()
                raise LLMError("Local LLM response correlation mismatch")
            try:
                result = Generation(**response)
                if (
                    result.model_id != MODEL_ID
                    or result.revision != MODEL_REVISION
                    or not isinstance(result.text, str)
                    or not result.text.strip()
                    or not math.isfinite(result.latency_seconds)
                    or result.latency_seconds < 0
                    or not 0 < result.prompt_tokens <= 4096
                    or not 0 < result.generation_tokens <= self.max_tokens
                ):
                    raise ValueError("Invalid generation metadata")
            except (TypeError, ValueError) as exc:
                raise LLMError("Local LLM returned invalid generation metadata") from exc
            return result

    def close(self) -> None:
        process, self._process = self._process, None
        self._buffer = b""
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if self._stderr:
            self._stderr.close()
            self._stderr = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _worker() -> None:
    # This file runs directly with the isolated Python; do not import relay_lab.
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    path = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
    )
    model, tokenizer = load(path)
    print(json.dumps({"ready": True, "model_id": MODEL_ID, "revision": MODEL_REVISION}), flush=True)
    for line in sys.stdin:
        request_id = None
        try:
            if len(line.encode()) > MAX_INPUT_BYTES + 1024:
                raise ValueError("Worker request exceeds input limit")
            request = json.loads(line)
            request_id = request["id"]
            prompt = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # The pinned official MLX conversion predates the template's
            # enable_thinking switch. Complete an empty thought prefix, exactly
            # as Qwen's newer non-thinking template does, before generation.
            if not prompt.rstrip().endswith("</think>"):
                prompt += "<think>\n\n</think>\n\n"
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens > 4096:
                raise ValueError("Prompt exceeds 4096 tokens; no truncation is permitted")
            started = time.monotonic()
            pieces, count = [], 0
            for chunk in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=request["max_tokens"],
                sampler=make_sampler(temp=0),
            ):
                pieces.append(chunk.text)
                count += 1
                if time.monotonic() - started > request["timeout"]:
                    raise TimeoutError("Local generation deadline exceeded")
            response = Generation(
                "".join(pieces), time.monotonic() - started, prompt_tokens, count
            ).to_dict()
            print(json.dumps({"id": request_id, **response}), flush=True)
        except Exception as exc:  # noqa: BLE001 - serialize worker failures; never execute a fallback
            print(
                json.dumps({"id": request_id, "error": f"{type(exc).__name__}: {exc}"}), flush=True
            )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module is an internal local LLM worker")
    _worker()
