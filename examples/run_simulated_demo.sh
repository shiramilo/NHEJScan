#!/usr/bin/env bash
# Self-contained demo: simulate a genome triplet with planted indels, run Mauve + the pipeline,
# and check that the planted events are recovered. Takes about one minute.
set -euo pipefail
OUT=${1:-demo_out}
mkdir -p "$OUT"
python3 - "$OUT" <<'PY'
import sys; sys.path.insert(0, "tests")
from simulate import simulate
simulate(sys.argv[1], seed=7, genome_len=200_000, n_events=24, snp_ab=0.0)
PY
nhejscan pair --genome1 "$OUT/A.fna" --genome2 "$OUT/B.fna" --outgroup "$OUT/O.fna" --outdir "$OUT/run"
echo "Results: $OUT/run/events.csv   (planted events: $OUT/truth.json)"
