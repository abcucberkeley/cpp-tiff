#!/usr/bin/env bash
# Build the read benchmark against the current sources and run it on benchData.
#
# Usage:
#   benchmarks/run_bench.sh [--label <name>] [--iters N] [--warmup N] [extra args...]
#
# Any extra arguments are passed through to benchRead. Results are appended to
# benchResults.csv at the repo root. Compare two runs with benchmarks/compare.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD="$ROOT/build"

echo ">> Configuring (Release, BUILD_BENCH=ON)..."
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCH=ON >/dev/null

echo ">> Building benchRead..."
cmake --build "$BUILD" --target benchRead --parallel

echo ">> Running benchmark..."
"$BUILD/benchRead" --dir "$ROOT/benchData" --csv "$ROOT/benchResults.csv" "$@"
