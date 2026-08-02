#!/bin/bash
# Pull a cohort's curves into its own directory, so a read is one command.
#
# Every selection error in this project came from assembling curve files by hand:
# the longest-run rule living in how I copied files rather than in the analysis,
# the drift tags colliding with the level tags, and a fetch that wiped the level
# experiment's committed record because it shared a directory. The read that
# decides the drift replication is the worst possible moment to do that again.
#
# Usage: fetch_curves.sh <tag-prefix> <dest-dir>
#   fetch_curves.sh 'r[03]s4' report/driftrep_csv
set -euo pipefail

PREFIX="${1:?tag prefix, e.g. 'r[03]s4'}"
DEST="${2:?destination directory}"
REMOTE="srijit@sc.stanford.edu"
JUMP="srijit@scdt.stanford.edu"

case "$DEST" in
  */drift_csv|*/seeded_csv|*/beta_csv)
    echo "refusing: $DEST holds a committed result. Use a fresh directory." >&2
    exit 1 ;;
esac

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

ssh -o ConnectTimeout=45 -J "$JUMP" "$REMOTE" \
  "cd /atlas/u/srijit/repro_csv && for f in ${PREFIX}*_eval*.csv; do \
     [ -s \"\$f\" ] || continue; echo \"===\$f\"; cat \"\$f\"; done" > "$tmp"

rm -rf "$DEST"; mkdir -p "$DEST"
awk -v dest="$DEST" '
  /^===/ { name = substr($0, 4); path = dest "/" name; next }
  path   { print > path }
' "$tmp"

n=$(ls "$DEST" | wc -l)
echo "$n curve files -> $DEST"
[ "$n" -gt 0 ] || { echo "nothing fetched - check the prefix" >&2; exit 1; }
