#!/usr/bin/env bash
#
# Server-side deploy receiver — the *only* thing the CI deploy key may run.
#
# The deploy key's authorized_keys entry pins this script as a forced command
# (command="...",no-pty,no-*-forwarding), so a holder of that key cannot open a
# shell, forward ports, or run anything else — it can only stream in a release
# and have it deployed. The CI runner does:
#
#     git archive --format=tar HEAD | ssh -i deploy_key ltsar@treasury.liam.cool
#
# SSH ignores whatever command the client requests and runs THIS script with the
# `git archive` tarball on stdin. We validate it, snapshot the current release,
# sync the new files in, reinstall deps, restart the service, and health-gate the
# result — rolling back automatically if the service does not come up healthy.
#
# This file is also part of the deployed tree (scripts/deploy-receive.sh), so on
# every successful deploy it reinstalls the newest copy of itself as the forced
# command for next time (see step 4) — no drift between repo and server.
set -euo pipefail

APP="$HOME/treasury"
BACKUP="$HOME/treasury.prev.tar"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

log() { echo "[deploy] $*" >&2; }

health_ok() {
  # /health is a pure status endpoint (no model call) -> instant when up.
  curl -fsS --max-time 5 http://127.0.0.1:8000/health 2>/dev/null \
    | grep -q '"status": *"ok"'
}

# 1. Receive the release tarball on stdin. Drain ALL of stdin into a file (cat
#    reads to EOF) BEFORE extracting: `tar -xf -` stops at the archive's
#    end-of-archive marker and leaves trailing record padding unread, so the SSH
#    client is left writing into a closed pipe — it sees a broken pipe, exits
#    255, and tears the session down mid-deploy. Reading to EOF first avoids that
#    entirely. Then sanity-check the payload — refuse to deploy an empty/corrupt
#    tarball rather than wiping a working release.
TARBALL="$STAGE/release.tar"
cat > "$TARBALL"
tar -xf "$TARBALL" -C "$STAGE"
rm -f "$TARBALL"
if [ ! -f "$STAGE/app.py" ] || [ ! -f "$STAGE/requirements.txt" ]; then
  log "FATAL: payload missing app.py/requirements.txt — refusing to deploy"
  exit 1
fi
log "received $(find "$STAGE" -type f | wc -l | tr -d ' ') files"

# 2. Snapshot the current release (minus the venv/caches) so we can roll back.
tar -cf "$BACKUP" -C "$APP" \
  --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' . 2>/dev/null || \
  log "WARN: could not snapshot current release (first deploy?) — no rollback"

# 3. Sync the new files into place. No rsync on the host, so cp -a; this
#    overwrites tracked files and leaves untracked state (.venv, .env, logs)
#    intact, since those are never in a `git archive` tarball.
cp -a "$STAGE"/. "$APP"/
log "files synced into $APP"

# Roll back to the pre-deploy snapshot and exit non-zero. Invoked for ANY failure
# after the files are synced — a broken dep install, a failed restart, or an app
# that never becomes healthy — so a bad deploy never leaves the box running
# new-but-broken code. Defensive (|| log) so a fault mid-rollback still reports.
rollback() {
  log "ERROR: $1 — rolling back to previous release"
  if [ -f "$BACKUP" ]; then
    tar -xf "$BACKUP" -C "$APP" || log "WARN: rollback extract reported errors"
    sudo -n systemctl restart treasury || log "WARN: restart during rollback failed"
    if health_ok; then
      log "rolled back to previous release (healthy)"
    else
      log "CRITICAL: rollback did not restore health — manual intervention needed"
    fi
  else
    log "CRITICAL: no snapshot to roll back to — manual intervention needed"
  fi
  exit 1
}

# 4. Reinstall deps (pinned, fast no-op when unchanged); a dependency failure
#    rolls back rather than leaving new code with missing/half deps. Then
#    self-update this receiver so the next deploy uses the newest committed
#    version (non-fatal — a stale receiver still deploys correctly next time).
if ! "$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"; then
  rollback "pip install failed"
fi
if [ -f "$APP/scripts/deploy-receive.sh" ]; then
  install -m 700 "$APP/scripts/deploy-receive.sh" "$HOME/deploy-receive.sh" \
    || log "WARN: could not self-update receiver (non-fatal)"
fi

# 5. Restart (just uvicorn — Ollama keeps the model resident, so it comes back in
#    seconds). A restart failure (broken unit, systemd error) also rolls back —
#    without this, `set -e` would abort here and skip both health-gate and rollback.
if ! sudo -n systemctl restart treasury; then
  rollback "systemctl restart failed"
fi

# 6. Health-gate the new release (retry ~30s). If it never comes up healthy,
#    roll back to the snapshot.
for _ in $(seq 1 15); do
  if health_ok; then
    log "deploy OK — service healthy after restart"
    exit 0
  fi
  sleep 2
done
rollback "service unhealthy after deploy"
