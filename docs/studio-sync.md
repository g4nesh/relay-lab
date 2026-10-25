# Keep Relay Lab and Relay Studio synchronized

[Relay Lab](https://github.com/g4nesh/relay-lab) owns the shared dashboard,
coordination services, worker protocol, Spacetime module and subscription bridge.
[Relay Studio](https://github.com/g4nesh/relay-studio) is the private, independently
deployable export of those components. Simulation and training remain in Relay
Lab. Both applications use the same shared source; experiment availability is
determined by installed modules and the deployment profile.

`relay-lab studio`, `relay-studio`, and the local research website launcher all
use the same ASGI application lifecycle, subscription bridge and scheduled
coordination services. The research website adds its static pages and cloth
viewer around that application; those pages are outside the standalone export.
The older synchronous `serve_studio` entry point delegates to the same ASGI
launcher, so an existing research launch command also starts the current live
coordination services. Hosted deployments and the stripped export disable
experiment upload/execution routes even when a client requests them directly.

## Export boundary

`scripts/export_studio.py` in Relay Lab defines an explicit file allowlist.
It copies shared files byte-for-byte and uses `deploy/studio/` templates for the
standalone README, Python package metadata, ignore rules and agent guide. Only
the Dockerfile's package-metadata location and the corresponding Docker ignore
entry are rewritten. The standalone package has no ML or simulator dependencies.

The export includes the complete native Spacetime schema, its scheduled reducers,
the Node subscription bridge, generated bindings, dashboard fonts/logo,
Spacetime checkpoint storage, Render deployment files and control-plane tests.
Existing schema and generated bindings retain research-named
tables for compatibility; this does not include experiment executables, models,
observations, recordings or customer records. Exporting never reads Git history,
credentials, state directories, checkpoints or datasets.

Every export writes `export-manifest.json` with the original source path, byte
length and SHA-256 hash of each delivered file. Build products and dependencies
are installed from pinned package/lock files after export.

## Updating the repositories

1. Implement shared changes in a clean Relay Lab branch. For urgent fixes first
   made in Relay Studio, port and review them in Relay Lab before the next export.
   Keep standalone-specific packaging changes in `deploy/studio/`.
2. Run the relevant research checks and the export tests. Export to a **new**
   directory; the script refuses existing destinations and source symlinks:

   ```bash
   # Run in the Relay Lab checkout.
   .venv/bin/python -m pytest tests/test_studio_export.py
   python3 scripts/export_studio.py --output /absolute/path/to/new-relay-studio-export
   ```

3. Compare the export with a clean Relay Studio branch. Review additions, changes
   and deletions explicitly, then copy only the reviewed source changes and
   manifest. Never replace the checkout wholesale or copy state between repos.
   Run the exported Python tests, Node checks and deployment build.
4. Verify synchronization from Relay Lab before committing the paired updates:

   ```bash
   python3 scripts/export_studio.py --check-against /absolute/path/to/relay-studio
   ```

   This command is read-only. It checks the bytes of every expected file and the
   manifest. For a Git checkout, unexpected **tracked** files and expected files
   omitted from Git's index also fail the check;
   untracked local state and build outputs are neither read nor copied. For a
   plain export directory, the full file inventory is checked. It exits nonzero
   on drift and names the missing, changed, unexpected or untracked files without printing
   their contents. Stage newly added files before the final check so Git's
   tracked inventory includes them.
5. Commit and push the paired repository updates, recording both revisions in
   the release or pull request. Merge after applicable checks. This workflow
   does not automatically push, deploy, publish a schema, reset a database or
   change a running customer's workspace.

An isolated dashboard build can be deployed directly from Relay Studio without
Relay Lab. Preserve one private workspace per customer and one intended replica.
Disk mode requires a persistent app volume and consistent volume/database backups
as described in [deployment](studio-deployment.md). Ephemeral hosts can instead
use [Spacetime checkpoint storage](render-free.md), including its required fenced
reducers and compatible rollback builds. Code synchronization does not synchronize
customer accounts, tokens, task records or robot journals.
