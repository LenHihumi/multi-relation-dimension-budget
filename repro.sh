#!/usr/bin/env bash
# Rebuilds tables/ and figs/ from data/.
#
# This is a build, not a test.  Nothing here compares a result against a number
# typed into a script: the tables are a function of the released JSON, and the
# evidence that they are is that running this leaves the working tree unchanged
# -- which CI checks on every push (.github/workflows/repro.yml).  The assertion
# suites that hold expected values live with the manuscript, not here.
set -eu
cd "$(dirname "$0")"

python3 scripts/make_tables.py
python3 scripts/arms_by_branch.py    # per branch, plus the bootstrap intervals; reads the
                                     # two files just written
python3 scripts/make_proj_table.py   # tables/proj_budget.csv, from data/proj/*.json

# The figures are a view of the tables, not a computation over the data, so they
# are rebuilt only when matplotlib is present.
python3 -c "import matplotlib" 2>/dev/null && python3 scripts/fig_response_law.py \
  || echo '[skip] matplotlib absent: FIG2 not rebuilt'
python3 -c "import matplotlib" 2>/dev/null && python3 scripts/fig_motivation.py \
  || echo '[skip] matplotlib absent: FIG1 not rebuilt'

cat <<'MSG'

tables/ regenerated. Nine of the twenty shipped tables are measurements that need a
corpus to regenerate (see README section 3); this script does not touch them.
MSG
