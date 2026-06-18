#!/usr/bin/env bash
#
# Client-side deploy — runs on the CI runner (or a laptop) to ship the current
# commit to the server. It streams a `git archive` tarball of HEAD over SSH to
# the deploy key, whose forced command (scripts/deploy-receive.sh, installed at
# ~/deploy-receive.sh on the server) extracts, installs, restarts and
# health-gates it. The SSH exit code IS the deploy verdict: the receiver exits
# non-zero (after rolling back) if the service is unhealthy, which fails the run.
#
# Required env:
#   DEPLOY_HOST  - SSH host (e.g. treasury.liam.cool)
#   DEPLOY_USER  - SSH user (e.g. ltsar)
#   SSH_KEY      - path to the private deploy key
#   KNOWN_HOSTS  - path to a known_hosts file pinning the server's host key
set -euo pipefail

: "${DEPLOY_HOST:?set DEPLOY_HOST}"
: "${DEPLOY_USER:?set DEPLOY_USER}"
: "${SSH_KEY:?set SSH_KEY}"
: "${KNOWN_HOSTS:?set KNOWN_HOSTS}"

REF="$(git rev-parse --short HEAD)"
echo ">> Deploying ${REF} to ${DEPLOY_USER}@${DEPLOY_HOST}"

# Stream the committed tree to the server (no .venv/.env/.git, no runner junk).
# Exclude test_images/ — ~25 MB of generated test bitmaps the server never runs;
# shipping them would bloat every deploy and widen the stdin-stream window. The
# forced command ignores any requested command and reads the tarball on stdin.
git archive --format=tar HEAD -- ':(exclude)test_images' '.' | ssh \
  -i "$SSH_KEY" \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$KNOWN_HOSTS" \
  -o ConnectTimeout=15 \
  -o BatchMode=yes \
  "${DEPLOY_USER}@${DEPLOY_HOST}"

echo ">> Deploy of ${REF} succeeded (receiver reported healthy)."
