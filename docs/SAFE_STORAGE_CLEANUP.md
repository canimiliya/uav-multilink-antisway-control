# Safe storage cleanup plan

This plan is advisory only. S6T6 performed a read-only size audit and deleted nothing. The current inventory is in `artifacts/s6_taskspace/final/storage_inventory.json`.

## SAFE_DELETE_NOW

Only after an independent preflight confirms the path is ignored or reproducible cache data:

- `__pycache__/`
- `.pytest_cache/`
- `*.pyc`
- temporary local solver/compiler caches that are not tracked and are not needed to reproduce a recorded artifact

Do not use a broad recursive delete. Resolve each target path, confirm it is ignored, and re-run `git status --porcelain` afterward.

## PRESERVE

Preserve all source, configs, frozen method/result files, final summaries, metric definitions, gates, raw metrics, hash manifests, and the complete `artifacts/s6_taskspace/final/` directory. Preserve `artifacts/s5b/` because it contains the formal LS-PMPC holdout evidence. Preserve the full original clone until the lightweight copy has been independently verified.

## REMOTE_ARCHIVED_BULKY

The largest local evidence is in retained raw CSV runs under `artifacts/s6_taskspace/t1`, `t2`, `t3`, `t5`, and `t5r1`. They may be candidates for a remote-archived/local-lightweight layout only after the final tag and remote contents are verified. Do not remove any tracked raw CSV in this task.

## Later lightweight-local-copy procedure

1. Confirm `s6-taskspace-final-2026-08-08` exists on the remote and points to the final freeze commit.
2. Create a new `git clone --depth 1` of the remote repository in a separate directory.
3. Verify the shallow clone HEAD and tag SHA.
4. Compare source, final summaries, manifests, and directory sizes against the original clone.
5. Ask the user for explicit approval before deleting or archiving the old full clone.

This task does not delete the original repository or any tracked evidence.
