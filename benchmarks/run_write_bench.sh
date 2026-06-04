#!/usr/bin/env bash
# Build the write benchmark against the current sources and run it.
#
# Usage:
#   benchmarks/run_write_bench.sh [--label <name>] [--compression lzw|none]
#                            [--transpose] [--fsync] [extra args...]
#
# Loads each volume in --dir (default benchData) via the reader, then times
# writing it back out to --outdir. Results append to benchWriteResults.csv at
# the repo root. Compare two runs with benchmarks/compare.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD="$ROOT/build"

echo ">> Configuring (Release, BUILD_BENCH=ON)..."
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCH=ON >/dev/null

echo ">> Building benchWrite..."
cmake --build "$BUILD" --target benchWrite --parallel

echo ">> Running write benchmark..."
"$BUILD/benchWrite" --dir "$ROOT/benchData" --outdir "$ROOT/benchWriteOut" \
    --csv "$ROOT/benchWriteResults.csv" "$@"
