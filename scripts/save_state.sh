#!/usr/bin/env bash
# Publish state/seen.json to the `state` branch as a single orphan commit (force-pushed), so
# 5-minute polling doesn't bury main's history and the branch never grows.
set -euo pipefail
REMOTE="${STATE_REMOTE:-https://x-access-token:${GITHUB_TOKEN:?}@github.com/${GITHUB_REPOSITORY:?}.git}"
[ -f state/seen.json ] || { echo "no state file to save"; exit 0; }
cd state
rm -rf .git
git init -q -b state
git config user.name "news-bot"
git config user.email "actions@github.com"
git add seen.json
git commit -q -m "state $(date -u +'%Y-%m-%d %H:%M')"
git push -q --force "$REMOTE" state
echo "saved state"
