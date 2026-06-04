// Benchmark harness for readTiffParallelWrapperNoXYFlip.
//
// Reads every *.tif in a directory (default: benchData) one or more times and
// reports per-file timing statistics plus throughput, so that different
// versions of the reader can be compared against each other.
//
// Results are appended to a CSV keyed by a --label (default: the current git
// short hash) so that two runs can be diffed with benchmarks/compare.py.
//
// Build via the top-level CMake with -DBUILD_BENCH=ON (see benchmarks/run_bench.sh).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include <omp.h>

#include "parallelreadtiff.h"
#include "helperfunctions.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct VolInfo {
    uint64_t x = 1, y = 1, z = 1, bits = 1, bytes = 0;
};

// Compute the logical (uncompressed) size of the volume the reader produces.
// Mirrors the dispatch logic in readTiffParallelWrapperHelper so the byte count
// matches what readTiffParallelWrapperNoXYFlip actually allocates and fills.
VolInfo logicalSize(const std::string& path) {
    VolInfo v;
    uint64_t* dims = getImageSize(path.c_str());  // {y, x, z}
    if (dims) {
        v.y = dims[0];
        v.x = dims[1];
        v.z = dims[2];
        free(dims);
    }
    if (isImageJIm(path.c_str())) {
        uint64_t tempZ = imageJImGetZ(path.c_str());
        if (tempZ) v.z = tempZ;
    }
    v.bits = getDataType(path.c_str());
    v.bytes = v.x * v.y * v.z * (v.bits / 8);
    return v;
}

struct Stats {
    double min_ms = 0, med_ms = 0, mean_ms = 0, std_ms = 0;
};

// FNV-1a over the buffer (word-wise for speed). Used only by --verify, outside
// the timed region, to confirm an optimization produces identical output.
uint64_t hashBuffer(const void* buf, uint64_t bytes) {
    uint64_t h = 1469598103934665603ULL;
    const uint64_t* w = static_cast<const uint64_t*>(buf);
    uint64_t nw = bytes / 8;
    for (uint64_t i = 0; i < nw; i++) { h ^= w[i]; h *= 1099511628211ULL; }
    const uint8_t* tail = static_cast<const uint8_t*>(buf) + nw * 8;
    for (uint64_t i = 0; i < (bytes & 7); i++) { h ^= tail[i]; h *= 1099511628211ULL; }
    return h;
}

Stats computeStats(std::vector<double> ms) {
    Stats s;
    if (ms.empty()) return s;
    std::sort(ms.begin(), ms.end());
    const size_t n = ms.size();
    s.min_ms = ms.front();
    s.med_ms = (n % 2) ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
    double sum = 0;
    for (double v : ms) sum += v;
    s.mean_ms = sum / n;
    if (n > 1) {
        double acc = 0;
        for (double v : ms) acc += (v - s.mean_ms) * (v - s.mean_ms);
        s.std_ms = std::sqrt(acc / (n - 1));
    } else {
        s.std_ms = 0;
    }
    return s;
}

std::string gitLabel() {
    FILE* p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (!p) return "unlabeled";
    char buf[64];
    std::string out;
    if (fgets(buf, sizeof(buf), p)) out = buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.empty() ? "unlabeled" : out;
}

std::string shortName(const std::string& name, size_t width) {
    if (name.size() <= width) return name;
    // Keep the tail (the distinguishing part, e.g. ..._nz_50000.tif).
    return "..." + name.substr(name.size() - (width - 3));
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --dir <path>     Directory of .tif files to benchmark (default: benchData)\n"
        "  --iters <n>      Timed iterations per file (default: 3)\n"
        "  --warmup <n>     Untimed warmup iterations per file (default: 1)\n"
        "  --label <str>    Label for CSV rows (default: git short hash)\n"
        "  --csv <path>     CSV file to append results to (default: benchResults.csv)\n"
        "  --range <a,b>    Optional zRange passed to the reader (default: full volume)\n"
        "  --no-csv         Do not write a CSV, console output only\n"
        "  --verify         Print an FNV-1a hash of each read buffer (untimed) to\n"
        "                   confirm an optimization produces identical output\n"
        "  --flip           Benchmark readTiffParallelWrapper (the XY-flipped\n"
        "                   MATLAB path) instead of the no-flip wrapper. Reads the\n"
        "                   full volume (ignores --range).\n"
        "  -h, --help       Show this help\n",
        prog);
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string dir = "benchData";
    std::string label;
    std::string csvPath = "benchResults.csv";
    int iters = 3;
    int warmup = 1;
    bool writeCsv = true;
    bool verify = false;
    bool flip = false;
    std::vector<uint64_t> zRange;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--dir") dir = next("--dir");
        else if (a == "--iters") iters = std::atoi(next("--iters").c_str());
        else if (a == "--warmup") warmup = std::atoi(next("--warmup").c_str());
        else if (a == "--label") label = next("--label");
        else if (a == "--csv") csvPath = next("--csv");
        else if (a == "--no-csv") writeCsv = false;
        else if (a == "--verify") verify = true;
        else if (a == "--flip") flip = true;
        else if (a == "--range") {
            std::string r = next("--range");
            size_t comma = r.find(',');
            if (comma == std::string::npos) {
                zRange = {(uint64_t)std::strtoull(r.c_str(), nullptr, 10)};
            } else {
                zRange = {(uint64_t)std::strtoull(r.substr(0, comma).c_str(), nullptr, 10),
                          (uint64_t)std::strtoull(r.substr(comma + 1).c_str(), nullptr, 10)};
            }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (iters < 1) iters = 1;
    if (warmup < 0) warmup = 0;
    if (label.empty()) label = gitLabel();

    if (!fs::is_directory(dir)) {
        std::fprintf(stderr, "Error: '%s' is not a directory\n", dir.c_str());
        return 1;
    }

    // Collect .tif files, sorted by on-disk size ascending (small -> large).
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".tif" || ext == ".tiff") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    if (files.empty()) {
        std::fprintf(stderr, "No .tif files found in '%s'\n", dir.c_str());
        return 1;
    }

    const int threads = omp_get_max_threads();
    char tsbuf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    std::printf("cpp-tiff read benchmark (%s)\n",
                flip ? "readTiffParallelWrapper [flip / MATLAB]"
                     : "readTiffParallelWrapperNoXYFlip");
    std::printf("  label=%s  threads=%d  warmup=%d  iters=%d  dir=%s\n",
                label.c_str(), threads, warmup, iters, dir.c_str());
    if (flip && !zRange.empty())
        std::printf("  note: --flip reads the full volume; --range is ignored\n");
    if (!flip && !zRange.empty()) {
        std::printf("  zRange=[");
        for (size_t i = 0; i < zRange.size(); i++)
            std::printf("%s%llu", i ? "," : "", (unsigned long long)zRange[i]);
        std::printf("]\n");
    }
    std::printf("\n");
    std::printf("%-34s %-16s %4s %8s %10s %10s %10s %8s %7s\n",
                "file", "dims(x,y,z)", "bits", "size", "min_ms", "med_ms",
                "mean_ms", "std_ms", "GB/s");
    std::printf("%s\n", std::string(118, '-').c_str());
    std::fflush(stdout);

    FILE* csv = nullptr;
    if (writeCsv) {
        bool existed = fs::exists(csvPath);
        csv = std::fopen(csvPath.c_str(), "a");
        if (!csv) {
            std::fprintf(stderr, "Warning: could not open CSV '%s' for writing\n", csvPath.c_str());
        } else if (!existed) {
            std::fprintf(csv,
                "label,timestamp,file,x,y,z,bits,logical_bytes,file_bytes,"
                "threads,warmup,iters,min_ms,median_ms,mean_ms,std_ms,gbps_median\n");
        }
    }

    // The flip wrapper has no zRange parameter and always reads the full volume.
    auto doRead = [&](const std::string& p) -> void* {
        return flip ? readTiffParallelWrapper(p.c_str())
                    : readTiffParallelWrapperNoXYFlip(p.c_str(), zRange);
    };

    double totalBytes = 0, totalMedSec = 0;
    for (const auto& path : files) {
        VolInfo v = logicalSize(path.string());
        uint64_t fileBytes = fs::file_size(path);

        std::vector<double> times;
        bool failed = false;
        uint64_t hash = 0;
        bool haveHash = false;

        for (int w = 0; w < warmup; w++) {
            void* buf = doRead(path.string());
            if (!buf) { failed = true; break; }
            if (verify && !haveHash) { hash = hashBuffer(buf, v.bytes); haveHash = true; }
            free(buf);
        }
        // If no warmup happened, do one untimed read just for the hash.
        if (verify && !haveHash && !failed) {
            void* buf = doRead(path.string());
            if (!buf) failed = true;
            else { hash = hashBuffer(buf, v.bytes); haveHash = true; free(buf); }
        }
        for (int it = 0; it < iters && !failed; it++) {
            auto t0 = Clock::now();
            void* buf = doRead(path.string());
            auto t1 = Clock::now();
            if (!buf) { failed = true; break; }
            free(buf);
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::string dims =
            std::to_string(v.x) + "x" + std::to_string(v.y) + "x" + std::to_string(v.z);

        if (failed) {
            std::printf("%-34s %-16s %4llu %8s %s\n",
                        shortName(path.filename().string(), 34).c_str(), dims.c_str(),
                        (unsigned long long)v.bits, "-", "FAILED to read");
            std::fflush(stdout);
            continue;
        }

        Stats s = computeStats(times);
        double medSec = s.med_ms / 1000.0;
        double gbps = (medSec > 0) ? (double)v.bytes / 1e9 / medSec : 0.0;

        std::printf("%-34s %-16s %4llu %7.2fG %10.2f %10.2f %10.2f %8.2f %7.2f\n",
                    shortName(path.filename().string(), 34).c_str(), dims.c_str(),
                    (unsigned long long)v.bits, (double)v.bytes / 1e9,
                    s.min_ms, s.med_ms, s.mean_ms, s.std_ms, gbps);
        if (verify && haveHash)
            std::printf("    verify: hash=%016llx\n", (unsigned long long)hash);
        std::fflush(stdout);

        if (csv) {
            std::fprintf(csv,
                "%s,%s,%s,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%d,"
                "%.3f,%.3f,%.3f,%.3f,%.4f\n",
                label.c_str(), tsbuf, path.filename().string().c_str(),
                (unsigned long long)v.x, (unsigned long long)v.y, (unsigned long long)v.z,
                (unsigned long long)v.bits, (unsigned long long)v.bytes,
                (unsigned long long)fileBytes, threads, warmup, iters,
                s.min_ms, s.med_ms, s.mean_ms, s.std_ms, gbps);
            std::fflush(csv);
        }

        totalBytes += (double)v.bytes;
        totalMedSec += medSec;
    }

    std::printf("%s\n", std::string(118, '-').c_str());
    if (totalMedSec > 0) {
        std::printf("total: %.2f GB across %zu files in %.2f s (median sum) -> %.2f GB/s aggregate\n",
                    totalBytes / 1e9, files.size(), totalMedSec, totalBytes / 1e9 / totalMedSec);
    }
    if (csv) std::fclose(csv);
    if (writeCsv) std::printf("results appended to %s\n", csvPath.c_str());
    return 0;
}
