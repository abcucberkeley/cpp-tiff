// Benchmark harness for writeTiffParallelWrapper (the parallel TIFF writer).
//
// For each *.tif in a directory it loads the volume into memory ONCE (untimed,
// via the reader), then times writing that buffer back out as a TIFF, reporting
// throughput where bytes is the logical (uncompressed) volume x*y*z*(bits/8) --
// the same normalizer benchRead uses, so read and write GB/s are comparable.
//
// Two layout modes mirror the reader's flip/no-flip:
//   default        : transpose=false, row-major input (loaded with the no-flip
//                    reader) -- the Python/C++ path.
//   --transpose    : transpose=true, column-major MATLAB input (loaded with the
//                    flip reader) -- the MATLAB path the mex writer uses.
//
// The writer currently supports only --compression lzw (default) or none.
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

#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct VolInfo {
    uint64_t x = 1, y = 1, z = 1, bits = 1, bytes = 0;
};

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

struct Stats { double min_ms = 0, med_ms = 0, mean_ms = 0, std_ms = 0; };

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
    return "..." + name.substr(name.size() - (width - 3));
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --dir <path>         Directory of source .tif volumes to load (default: benchData)\n"
        "  --outdir <path>      Directory to write output TIFFs to (default: benchWriteOut)\n"
        "  --iters <n>          Timed iterations per file (default: 3)\n"
        "  --warmup <n>         Untimed warmup iterations per file (default: 1)\n"
        "  --label <str>        Label for CSV rows (default: git short hash)\n"
        "  --csv <path>         CSV file to append results to (default: benchWriteResults.csv)\n"
        "  --compression <s>    lzw (default), zstd, or none\n"
        "  --transpose          Write the MATLAB column-major path (transpose=true)\n"
        "  --fsync              fsync the output to disk inside the timed region\n"
        "  --no-csv             Console output only\n"
        "  --verify             Round-trip: read the written file back and check its\n"
        "                       hash matches the in-memory input buffer\n"
        "  -h, --help           Show this help\n",
        prog);
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string dir = "benchData";
    std::string outdir = "benchWriteOut";
    std::string label;
    std::string csvPath = "benchWriteResults.csv";
    std::string compression = "lzw";
    int iters = 3, warmup = 1;
    uint64_t stripSize = 512;
    bool writeCsv = true, verify = false, transpose = false, doFsync = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", opt); std::exit(2); }
            return argv[++i];
        };
        if (a == "--dir") dir = next("--dir");
        else if (a == "--outdir") outdir = next("--outdir");
        else if (a == "--iters") iters = std::atoi(next("--iters").c_str());
        else if (a == "--warmup") warmup = std::atoi(next("--warmup").c_str());
        else if (a == "--label") label = next("--label");
        else if (a == "--csv") csvPath = next("--csv");
        else if (a == "--compression") compression = next("--compression");
        else if (a == "--stripsize") stripSize = std::strtoull(next("--stripsize").c_str(), nullptr, 10);
        else if (a == "--transpose") transpose = true;
        else if (a == "--fsync") doFsync = true;
        else if (a == "--no-csv") writeCsv = false;
        else if (a == "--verify") verify = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (iters < 1) iters = 1;
    if (warmup < 0) warmup = 0;
    if (label.empty()) label = gitLabel();
    if (compression != "lzw" && compression != "none" && compression != "zstd" && !compression.empty()) {
        std::fprintf(stderr, "Error: --compression must be 'lzw', 'zstd', 'none', or '' (none)\n");
        return 2;
    }
    if (!fs::is_directory(dir)) {
        std::fprintf(stderr, "Error: '%s' is not a directory\n", dir.c_str());
        return 1;
    }
    std::error_code ec;
    fs::create_directories(outdir, ec);
    if (fs::weakly_canonical(outdir) == fs::weakly_canonical(dir)) {
        std::fprintf(stderr, "Error: --outdir must differ from --dir (would overwrite sources)\n");
        return 1;
    }

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
    if (files.empty()) { std::fprintf(stderr, "No .tif files in '%s'\n", dir.c_str()); return 1; }

    const int threads = omp_get_max_threads();
    char tsbuf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    std::printf("cpp-tiff WRITE benchmark (writeTiffParallelWrapper, transpose=%s)\n",
                transpose ? "true [MATLAB]" : "false [native]");
    std::printf("  label=%s  compression=%s  fsync=%s  threads=%d  warmup=%d  iters=%d\n",
                label.c_str(), compression.c_str(), doFsync ? "yes" : "no", threads, warmup, iters);
    std::printf("  src=%s  out=%s\n\n", dir.c_str(), outdir.c_str());
    std::printf("%-34s %-16s %4s %8s %10s %10s %10s %8s %7s %7s\n",
                "file", "dims(x,y,z)", "bits", "size", "min_ms", "med_ms",
                "mean_ms", "std_ms", "GB/s", "ratio");
    std::printf("%s\n", std::string(126, '-').c_str());
    std::fflush(stdout);

    FILE* csv = nullptr;
    if (writeCsv) {
        bool existed = fs::exists(csvPath);
        csv = std::fopen(csvPath.c_str(), "a");
        if (csv && !existed) {
            std::fprintf(csv,
                "label,timestamp,file,x,y,z,bits,logical_bytes,out_bytes,ratio,"
                "compression,transpose,fsync,threads,warmup,iters,"
                "min_ms,median_ms,mean_ms,std_ms,gbps_median\n");
        }
    }

    auto loadInput = [&](const std::string& p) -> void* {
        return transpose ? readTiffParallelWrapper(p.c_str())
                         : readTiffParallelWrapperNoXYFlip(p.c_str(), {});
    };

    auto writeOut = [&](const std::string& outPath, const VolInfo& v, void* buf) -> bool {
        uint8_t err = writeTiffParallelWrapper(v.x, v.y, v.z, outPath.c_str(), buf, v.bits,
                                               0, stripSize, "w", transpose, compression);
        if (err) return false;
        if (doFsync) {
            int fd = ::open(outPath.c_str(), O_RDONLY);
            if (fd >= 0) { ::fsync(fd); ::close(fd); }
        }
        return true;
    };

    double totalBytes = 0, totalMedSec = 0;
    for (const auto& path : files) {
        VolInfo v = logicalSize(path.string());
        std::string base = path.filename().string();
        std::string outPath = (fs::path(outdir) / base).string();
        std::string dims = std::to_string(v.x) + "x" + std::to_string(v.y) + "x" + std::to_string(v.z);

        void* buf = loadInput(path.string());
        if (!buf) {
            std::printf("%-34s %-16s %4llu %8s %s\n", shortName(base, 34).c_str(), dims.c_str(),
                        (unsigned long long)v.bits, "-", "FAILED to load source");
            std::fflush(stdout);
            continue;
        }
        uint64_t inHash = verify ? hashBuffer(buf, v.bytes) : 0;

        bool failed = false;
        uint64_t outBytes = 0;
        for (int w = 0; w < warmup && !failed; w++) {
            if (!writeOut(outPath, v, buf)) failed = true;
            else if (!outBytes) outBytes = fs::file_size(outPath, ec);
            fs::remove(outPath, ec);
        }
        std::vector<double> times;
        for (int it = 0; it < iters && !failed; it++) {
            auto t0 = Clock::now();
            bool ok = writeOut(outPath, v, buf);
            auto t1 = Clock::now();
            if (!ok) { failed = true; break; }
            if (!outBytes) outBytes = fs::file_size(outPath, ec);
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            fs::remove(outPath, ec);
        }

        bool verifyOk = false;
        if (verify && !failed) {
            if (writeOut(outPath, v, buf)) {
                void* rb = transpose ? readTiffParallelWrapper(outPath.c_str())
                                     : readTiffParallelWrapperNoXYFlip(outPath.c_str(), {});
                if (rb) { verifyOk = (hashBuffer(rb, v.bytes) == inHash); free(rb); }
                fs::remove(outPath, ec);
            }
        }
        free(buf);

        if (failed) {
            std::printf("%-34s %-16s %4llu %8s %s\n", shortName(base, 34).c_str(), dims.c_str(),
                        (unsigned long long)v.bits, "-", "FAILED to write");
            std::fflush(stdout);
            continue;
        }

        Stats s = computeStats(times);
        double medSec = s.med_ms / 1000.0;
        double gbps = (medSec > 0) ? (double)v.bytes / 1e9 / medSec : 0.0;
        double ratio = outBytes ? (double)v.bytes / (double)outBytes : 0.0;

        std::printf("%-34s %-16s %4llu %7.2fG %10.2f %10.2f %10.2f %8.2f %7.2f %6.2fx\n",
                    shortName(base, 34).c_str(), dims.c_str(), (unsigned long long)v.bits,
                    (double)v.bytes / 1e9, s.min_ms, s.med_ms, s.mean_ms, s.std_ms, gbps, ratio);
        if (verify) std::printf("    verify: round-trip %s (in-hash=%016llx)\n",
                                verifyOk ? "OK" : "*** MISMATCH ***", (unsigned long long)inHash);
        std::fflush(stdout);

        if (csv) {
            std::fprintf(csv,
                "%s,%s,%s,%llu,%llu,%llu,%llu,%llu,%llu,%.4f,%s,%d,%d,%d,%d,%d,"
                "%.3f,%.3f,%.3f,%.3f,%.4f\n",
                label.c_str(), tsbuf, base.c_str(),
                (unsigned long long)v.x, (unsigned long long)v.y, (unsigned long long)v.z,
                (unsigned long long)v.bits, (unsigned long long)v.bytes,
                (unsigned long long)outBytes, ratio, compression.c_str(), transpose ? 1 : 0,
                doFsync ? 1 : 0, threads, warmup, iters,
                s.min_ms, s.med_ms, s.mean_ms, s.std_ms, gbps);
            std::fflush(csv);
        }
        totalBytes += (double)v.bytes;
        totalMedSec += medSec;
    }

    std::printf("%s\n", std::string(126, '-').c_str());
    if (totalMedSec > 0)
        std::printf("total: %.2f GB across %zu files in %.2f s (median sum) -> %.2f GB/s aggregate\n",
                    totalBytes / 1e9, files.size(), totalMedSec, totalBytes / 1e9 / totalMedSec);
    if (csv) std::fclose(csv);
    if (writeCsv) std::printf("results appended to %s\n", csvPath.c_str());
    return 0;
}
