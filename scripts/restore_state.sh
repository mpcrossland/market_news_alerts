#!/usr/bin/env bash
# Pull state/seen.json from the `state` branch (if it exists) into ./state/.
# STATE_REMOTE defaults to this repo over HTTPS using the Actions token.
set -euo pipefail
REMOTE="${STATE_REMOTE:-https://x-access-token:${GITHUB_TOKEN:?}@github.com/${GITHUB_REPOSITORY:?}.git}"
mkdir -p state
if git ls-remote --exit-code --heads "$REMOTE" state >/dev/null 2>&1; then
  tmp="$(mktemp -d)"
  git -C "$tmp" init -q
  git -C "$tmp" fetch -q --depth=1 "$REMOTE" state
  git -C "$tmp" show FETCH_HEAD:seen.json > state/seen.json
  rm -rf "$tmp"
  echo "restored state ($(wc -c < state/seen.json) bytes)"
else
  echo "no state branch yet; the bot will seed on this run"
fi
