#!/bin/sh
# prune-receipt.sh <host-label> <dir> [<dir>...]
# Writes a pre-deletion receipt (bytes, file count, sha256 of all *.json identity files, full file listing)
# to /tmp/prune-removal-<host>-pre.json + /tmp/prune-removal-<host>-listing.txt. READ-ONLY otherwise.
set -u
HOST=$1; shift
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OUT=/tmp/prune-removal-$HOST-pre.json
LIST=/tmp/prune-removal-$HOST-listing.txt
: > "$LIST"
{
  printf '{"schema":"prune-removal-pre-deletion-v1","utc":"%s","host":"%s","user":"%s","dirs":[' "$TS" "$HOST" "$(id -un)"
  first=1
  for D in "$@"; do
    [ $first -eq 1 ] || printf ','
    first=0
    BYTES=$(du -sb "$D" | cut -f1)
    NFILES=$(find "$D" -type f | wc -l)
    JSONSHA=$(for f in "$D"/*.json; do [ -f "$f" ] && sha256sum "$f"; done | tr '\n' ';')
    printf '{"path":"%s","bytes":%s,"files":%s,"json_sha256":"%s"}' "$D" "$BYTES" "$NFILES" "$JSONSHA"
    echo "### $D ($BYTES bytes, $NFILES files, $(date -u +%Y-%m-%dT%H:%M:%SZ))" >> "$LIST"
    find "$D" -type f -printf '%s\t%P\n' 2>/dev/null | sort -k2 >> "$LIST"
  done
  printf ']}'
} > "$OUT"
echo "RECEIPT_WRITTEN $OUT"; cat "$OUT"; echo; echo "LISTING $LIST lines=$(wc -l < "$LIST")"
