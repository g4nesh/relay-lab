# Connect your own agent

Studio delivers goals and current peer reports to an outbound worker on your
runtime host. The worker can call your own model, simulator or observation adapter.
It does not infer robot hardware or load an arbitrary registered checkpoint.

1. Register your two model references under **Models & releases** and save.
2. A manager releases both exact configurations.
3. Under **Tasks → Runtime connections**, create a connection for one model slot.
   Save its one-time token on that model's host, outside source control, mode 0600.
4. Install this package and run a locally installed handler:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install /path/to/relay-studio
chmod 600 /private/path/worker-token
.venv/bin/relay-agent --url https://YOUR-WORKSPACE-DOMAIN \
  --token-file /private/path/worker-token \
  --handler my_agent:handle --state-dir /private/path/agent-state
```

Your host needs Python 3.12+ on macOS or Linux plus your model's own dependencies.
Studio has none of the research checkout's ML dependencies. A connected badge
means a recent authenticated heartbeat, not verified camera/model readiness.

## Handler contract

`handle(context)` receives `task`, `worker`, `step`, and a `should_stop` callback.
Read the exact goal from `task['objective']`, criteria from `success_criteria`, and
peer reports from `task['assignments']`. Your own slot is `worker['agent_id']`.
Before executing and during long steps, check `context['should_stop']()`; hold on
true. Generate an action using your model, validate it locally, execute it in your
simulator, and report what you observed:

```python
def handle(context):
    # Replace this deliberate hold with your own model/simulator integration.
    return {
        "status": "blocked",
        "progress": 0,
        "summary": "No robot adapter is configured on this host yet.",
        "evidence": [],
    }
```

Statuses are `running`, `blocked`, `succeeded`, or `failed`. Progress is 0–100
(success requires 100); summaries are 1–1,000 characters; evidence is at most eight
strings of 1–500 characters. `running` invokes another step on the next poll;
`blocked` waits for a task revision change. A completed dashboard task means all
assigned agents reported success, not independent verification of physical work.

## Durable execution

Use one credential and one persistent journal on one agent host. Never share that
credential across hosts. Live events wake workers when assignments change;
polling provides a 30-second fallback, and heartbeats run every ten seconds.
Tokens expire after 30 days and are pinned to the approved model
revision. Retirement, changed approval, issuer demotion and explicit revocation
stop access. New releases require new credentials and tasks.

The SDK retains request identities and pending reports across retries. It records
an execution entry before calling your handler. If it crashes mid-action, it holds
on restart rather than guessing whether to execute again. Do not erase the journal
to force a retry. Cancellation is cooperative; the host owns hardware interlocks
and emergency stops. A network timeout does not prove an action did not happen.

The current workspace permits 32 active worker credentials, 100 stored tasks,
120 update revisions per task and 50 handler steps per agent/task. Large reports
can hit the 48 KiB active task budget sooner. Keep reports compact.

## API integration

The SDK uses `/api/agent/me`, `/api/agent/heartbeat`, `/api/agent/events`, `/api/agent/tasks`, and
`/api/agent/tasks/{goal-id}/{claim,report}` with a scoped bearer credential.
Claims/reports carry `request_id` and `expected_revision`; stale writes conflict.
Browser task APIs use authenticated sessions and CSRF checks. Remote origins
require HTTPS. Credentials never belong in URLs, command arguments or logs.

The generic loop exchanges structured text. Integrate Qwen or latent adapters
inside your runtime handler if you need them; saving a communication mode in the
dashboard does not activate a model or translator on your agent host.
