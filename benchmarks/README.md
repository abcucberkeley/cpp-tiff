# Read benchmark

Tooling to benchmark the parallel TIFF reader and compare different versions of
it against each other. By default it times `readTiffParallelWrapperNoXYFlip`
(the Python / C++ path); pass `--flip` to time `readTiffParallelWrapper` (the
XY-flipped MATLAB path) instead.

## Quick start

```bash
# Build + run on benchData, tagging this run as "baseline"
benchmarks/run_bench.sh --label baseline

# ...make an optimization in src/, then:
benchmarks/run_bench.sh --label opt1

# Diff the two runs
python3 benchmarks/compare.py baseline opt1
```

`run_bench.sh` configures a Release build with `-DBUILD_BENCH=ON`, rebuilds the
library and `benchRead` from the current sources, and runs the benchmark. The
first build compiles the vendored dependencies (libtiff, zstd, zlib, libdeflate)
and is slow; later builds are incremental.

## What it measures

For each `.tif` in the target directory (default `benchData`, sorted small to
large) it does `--warmup` untimed reads then `--iters` timed reads of the full
volume via `readTiffParallelWrapperNoXYFlip`, freeing the buffer each time. It
reports min / median / mean / stddev in ms and throughput in GB/s, where bytes
is the **logical** (uncompressed) volume size `x*y*z*(bits/8)` — a stable
normalizer that does not depend on the on-disk compression.

Timings are **warm-cache** by design (warmup populates the page cache): we are
comparing reader code, not disk speed. With enough RAM to cache the data set
this is reproducible across runs. To measure cold reads instead, drop caches
between runs (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`) and set
`--warmup 0`.

## benchRead options

```
--dir <path>     Directory of .tif files (default: benchData)
--iters <n>      Timed iterations per file (default: 3)
--warmup <n>     Untimed warmup iterations per file (default: 1)
--label <str>    Label for CSV rows (default: git short hash)
--csv <path>     CSV to append to (default: benchResults.csv)
--range <a,b>    Optional zRange passed to the reader (default: full volume)
--no-csv         Console output only
--verify         Print an FNV-1a hash of each read buffer (untimed) to confirm an
                 optimization produces identical output
--flip           Benchmark readTiffParallelWrapper (the XY-flipped MATLAB path)
                 instead of the no-flip wrapper. Reads the full volume.
```

Use `--verify` to A/B-check correctness: the hash must be unchanged between a
baseline and an optimized run of the *same* mode. Flip and no-flip produce
different hashes (different memory layout), so only compare like with like.

Results accumulate in `benchResults.csv` (one row per file per run, keyed by
`label`). `compare.py` uses the most recent row for each `(label, file)`.

## Write benchmark (`benchWrite`)

Times `writeTiffParallelWrapper` on the **same volumes** as the reader. For each
`.tif` in `--dir` it loads the volume into memory once (untimed, via the reader),
then times writing that buffer back out as a TIFF, reporting throughput against
the same logical-byte normalizer so read and write GB/s are directly comparable.

```bash
# native (row-major) path, LZW
benchmarks/run_write_bench.sh --label w-lzw --compression lzw
# MATLAB (column-major) path is --transpose; uncompressed ceiling is --compression none
benchmarks/run_write_bench.sh --label w-none-matlab --compression none --transpose
```

Two layout modes mirror the reader's flip/no-flip:
- default: `transpose=false`, row-major input (loaded with the no-flip reader) —
  the Python / C++ path.
- `--transpose`: `transpose=true`, column-major MATLAB input (loaded with the
  flip reader) — the path the mex writer uses.

The writer supports `--compression lzw` (the default), `zstd`, or `none`. zstd
is available as an option and encodes far faster than LZW (so the write is I/O-
rather than compress-bound) and compresses 16-bit data better. `--verify` does a round-trip (write, read the
file back, compare its hash to the in-memory input) to confirm correctness.
By default writes are buffered (no fsync), so the number reflects the library's
compress+format throughput; pass `--fsync` to fold the flush-to-disk cost into
the timed region. Results go to `benchWriteResults.csv` (same core columns as
the read CSV, so `compare.py` works on it, plus `out_bytes`/`ratio`/
`compression`/`transpose`/`fsync`).
