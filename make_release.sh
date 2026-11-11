#!/usr/bin/env bash
# Builds the Zenodo deposit zip.  Refuses while the metadata is incomplete, and
# refuses to package a deposit that does not reproduce *as packaged*: the file
# list is staged into a temporary directory and repro.sh runs there, not here.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if grep -q 'FILL:' .zenodo.json; then
  echo "REFUSING: .zenodo.json still contains FILL placeholders."
  echo "Fill creators (name, ORCID, affiliation) and the related identifier first."
  echo "Uploading with placeholders would publish an artifact that cannot be cited."
  exit 1
fi
if grep -q 'Author One\|example.org' ../authors.tex 2>/dev/null; then
  echo "WARNING: ../authors.tex still has placeholder authors."
fi
if [ -f ../sections/01_intro.tex ] && grep -q 'zenodo.XXXXXXX' ../sections/01_intro.tex; then
  echo "REFUSING: the paper still prints a placeholder artifact DOI (zenodo.XXXXXXX)."
  echo "Reserve the Zenodo DOI, paste it into \\artifactdoi in sections/01_intro.tex, recompile."
  exit 1
fi

VERSION="${1:-v1}"
OUT="$HERE/../paper_icde_artifact_${VERSION}.zip"

# The file list is the deposit: no caches, no run output, no repository furniture.
find . -type f \
  ! -name '*.pyc' ! -name 'make_release*.sh' ! -name 'SHA256SUMS.txt' \
  ! -path '*/__pycache__/*' ! -path './outputs/*' \
  ! -path './.git/*' ! -path './.github/*' \
  | LC_ALL=C sort > /tmp/_art_files.txt

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE" 2>/dev/null || true' EXIT
while IFS= read -r f; do
  mkdir -p "$STAGE/$(dirname "$f")"
  cp -p "$f" "$STAGE/$f"
done < /tmp/_art_files.txt

echo "== running repro.sh on the staged deposit =="
if ! ( cd "$STAGE" && bash repro.sh > /tmp/_release_repro.log 2>&1 ); then
  echo "REFUSING: repro.sh failed on the staged deposit. Last lines:"
  tail -20 /tmp/_release_repro.log
  exit 1
fi
grep -E "checks, all pass|supplement checks agree" /tmp/_release_repro.log

rm -f "$OUT"
( cd "$STAGE" && zip -q -X "$OUT" -@ < /tmp/_art_files.txt )
( cd "$STAGE" && sha256sum $(cat /tmp/_art_files.txt) ) > "$HERE/SHA256SUMS.txt"

wc -l < /tmp/_art_files.txt | xargs echo "files in deposit:"
du -h "$OUT" | awk '{print "zip size:", $1}'

cat <<'MSG'

Next steps (Zenodo web UI, or `zenodo_get`/API with a token):
  1. New upload -> select the zip
  2. Paste .zenodo.json into "Metadata" (or upload it alongside; Zenodo reads it)
  3. Reserve DOI -> copy the DOI into sections/01_intro.tex and README.md
  4. Publish, then put the *concept* DOI in the paper so later versions resolve
MSG
