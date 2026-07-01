// Minimal smoke test for the cpp-tiff library: write small random volumes, read
// them back, and check the bytes survive the round trip. Covers every bit depth /
// sample format and each compression, plus the XY-flip reader (MATLAB layout).
//
// Exits 0 if every case passes, 1 otherwise, so CTest reports pass/fail.
// Usage: roundtripTest [output_dir]   (defaults to the current directory)
//
// Progress is written (flushed) to stderr before each operation so that if a
// build segfaults, the CTest log shows exactly which step/dtype/compression died.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"

struct Case { const char* name; uint64_t bits; uint16_t sampleFormat; };

static void step(const char* name, const char* comp, const char* what){
    std::fprintf(stderr, "  [%s/%s] %s\n", name, comp, what);
    std::fflush(stderr);
}

int main(int argc, char** argv){
    const std::string dir = (argc > 1) ? argv[1] : ".";

    const uint64_t x = 40, y = 24, z = 3;
    const uint64_t stripSize = 8;
    const uint64_t n = x * y * z;

    const Case cases[] = {
        {"uint8",  8,  1}, {"uint16", 16, 1}, {"int16",  16, 2},
        {"int32",  32, 2}, {"float",  32, 3}, {"double", 64, 3},
    };
    const char* comps[] = {"none", "lzw", "zstd"};

    std::mt19937 rng(1234567u);
    std::uniform_int_distribution<int> byteDist(0, 255);

    int total = 0, failures = 0;
    for (const Case& c : cases){
        const uint64_t es = c.bits / 8;
        const uint64_t nbytes = n * es;

        std::vector<uint8_t> orig(nbytes);
        for (uint64_t i = 0; i < nbytes; i++) orig[i] = static_cast<uint8_t>(byteDist(rng));

        for (const char* comp : comps){
            total++;
            const std::string path = dir + "/rt_" + c.name + "_" + comp + ".tif";

            step(c.name, comp, "write");
            uint8_t werr = writeTiffParallelWrapper(x, y, z, path.c_str(), orig.data(), c.bits,
                                                    0, stripSize, "w", false,
                                                    std::string(comp), c.sampleFormat);
            if (werr){ std::printf("FAIL  %-6s %-4s  (write error)\n", c.name, comp); failures++; continue; }

            // Metadata should round trip.
            step(c.name, comp, "read metadata");
            const uint64_t rbits = getDataType(path.c_str());
            const uint64_t rfmt  = getSampleFormat(path.c_str());
            uint64_t* dims = getImageSize(path.c_str());   // {y, x, z}
            const bool metaOK = rbits == c.bits && rfmt == c.sampleFormat &&
                                dims[0] == y && dims[1] == x && dims[2] == z;
            step(c.name, comp, "free metadata buffer");
            free(dims);

            // No-flip read (the Python/C layout) must be byte-identical to what we wrote.
            step(c.name, comp, "read no-flip");
            void* rbuf = readTiffParallelWrapperNoXYFlip(path.c_str(), {});
            const bool dataOK = rbuf && std::memcmp(rbuf, orig.data(), nbytes) == 0;
            step(c.name, comp, "free no-flip buffer");
            free(rbuf);

            // Flip read (the MATLAB (y,x,z) column-major layout): compare against the
            // transpose of the original, exercising the flipTransposeStrip path.
            step(c.name, comp, "read flip");
            std::vector<uint8_t> expFlip(nbytes);
            for (uint64_t zi = 0; zi < z; zi++)
                for (uint64_t yi = 0; yi < y; yi++)
                    for (uint64_t xi = 0; xi < x; xi++)
                        std::memcpy(&expFlip[(yi + xi*y + zi*x*y) * es],
                                    &orig[(zi*y*x + yi*x + xi) * es], es);
            void* fbuf = readTiffParallelWrapper(path.c_str());
            const bool flipOK = fbuf && std::memcmp(fbuf, expFlip.data(), nbytes) == 0;
            step(c.name, comp, "free flip buffer");
            free(fbuf);

            std::remove(path.c_str());

            if (metaOK && dataOK && flipOK){
                std::printf("PASS  %-6s %-4s\n", c.name, comp);
            } else {
                std::printf("FAIL  %-6s %-4s  (meta=%d noflip=%d flip=%d)\n",
                            c.name, comp, (int)metaOK, (int)dataOK, (int)flipOK);
                failures++;
            }
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
