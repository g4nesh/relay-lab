# Relay Studio UI handoff

The shared starting branch is `codex/relay-studio`. It includes the current
Studio UI, its Python API, and the underlying cloth/VLA/Spacetime implementation.
Create your own branch from it for UI work:

```bash
git clone --branch codex/relay-studio https://github.com/g4nesh/relay-lab.git
cd relay-lab
git switch -c ui/team-workspace
uv run relay-lab studio --port 8770
```

Open <http://127.0.0.1:8770/>. Python 3.12+ and `uv` are required. Use a GitHub
account with repository write access when pushing your UI branch.

## Files to work in

| File | Responsibility |
|---|---|
| `src/relay_lab/studio/web/index.html` | Page structure, forms and dialogs |
| `src/relay_lab/studio/web/studio.css` | Layout, colors, typography and responsive behavior |
| `src/relay_lab/studio/web/studio.js` | UI state, model registration and API calls |
| `docs/studio.md` | Backend API and supported behavior |

This is plain HTML/CSS/JavaScript: no frontend bundler or Node dependencies are
required. Edit the files and refresh the browser; the server reads static assets
from disk on every request. `node --check src/relay_lab/studio/web/studio.js`
checks JavaScript syntax if Node is installed.

The registration and layout flow can be developed without loading foundation
models or running Spacetime. Opening a live communication session requires the
existing Spacetime setup. Text inference additionally requires the local Qwen
runtime on Apple Silicon; learned adapter inference requires the `learned`
dependency extra. Missing services must appear as unavailable, not as simulated
successful exchanges. See the main README and Studio guide for full setup.

## Work in parallel

- Keep UI changes in `studio/web/` where possible. Coordinate API changes through
  `docs/studio.md` and separate commits.
- Backend implementation lives in `studio/server.py`, `studio/service.py`,
  `studio/artifacts.py`, and the existing coordination modules. Avoid moving or
  renaming those files as part of a visual redesign.
- Preserve API field names and distinguish registration, communication, model
  loading and actual robot execution. Captured camera inputs are not live feeds.
- Commit only your changed source files. Local tokens, uploaded weights and
  workspace configuration belong under ignored `.relay-lab/`; do not commit them.
- Open a PR from the UI branch into `codex/relay-studio` when ready. The existing
  backend checkout can continue independently and bring updates into this branch.

The API already exposes health, compatibility diagnostics, session listing and
recovery, and captured camera images. The minimal UI does not yet use every API
capability; these are available for the next UI iteration.
