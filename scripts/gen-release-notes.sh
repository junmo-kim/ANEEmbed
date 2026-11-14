#!/usr/bin/env bash
#
# GitHub release notes: the CHANGELOG.md section for the tag (if any) as the keynote, then the commit
# subjects since the previous tag. No install instructions (those live in the README).
#
# Usage: scripts/gen-release-notes.sh <tag>      e.g. scripts/gen-release-notes.sh v0.1.0
# Needs full git history and tags (actions/checkout with fetch-depth: 0).
# Written for bash 3.2 (macOS default).
set -euo pipefail

TAG="${1:?usage: scripts/gen-release-notes.sh <tag>}"
PREV_TAG="$(git describe --abbrev=0 --tags "${TAG}^" 2>/dev/null || true)"
RANGE="${TAG}"
[ -n "$PREV_TAG" ] && RANGE="${PREV_TAG}..${TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANGELOG="${REPO_ROOT}/CHANGELOG.md"

echo "## aneembed ${TAG}"
echo

if [ -f "$CHANGELOG" ]; then
  # Literal comparison, not a regex: "." in a tag must not match any character.
  keynote="$(awk -v tag="$TAG" '
    $0 == "## " tag { p=1; next }
    /^## / { if (p) exit }
    p { buf[++n]=$0 }
    END {
      first=1; last=n
      while (first<=n && buf[first]=="") first++
      while (last>=first && buf[last]=="") last--
      for (i=first;i<=last;i++) print buf[i]
    }
  ' "$CHANGELOG")"
  if [ -n "$keynote" ]; then
    printf '%s\n\n' "$keynote"
  fi
fi

if [ -n "$PREV_TAG" ]; then
  echo "## What's Changed since ${PREV_TAG}"
else
  echo "## What's Changed"
fi
echo

log="$(git log --no-merges --format='%h %s' "$RANGE" | grep -vE '^[0-9a-f]+ (chore\(release\)|Release v)' || true)"
if [ -n "$log" ]; then
  printf '%s\n' "$log" | sed -E 's/^([0-9a-f]+) (.*)$/- \2 (\1)/'
else
  echo "_no notable changes_"
fi
