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

// ---- ImageJ-format fixture writer -----------------------------------------
// A single-IFD TIFF describing slice 0, every slice's raw data contiguous after
// it, and an ImageJ description carrying images=Z -- the layout ImageJ's fast
// save produces, which the library reads through its ImageJ path. The library's
// own writer cannot produce this layout, so the test writes it by hand. Fields
// are emitted byte-by-byte so both byte orders can be produced on any host.
static void ijPut16(std::vector<uint8_t>& v, uint16_t s, bool be){
    if(be){ v.push_back((uint8_t)(s >> 8)); v.push_back((uint8_t)s); }
    else  { v.push_back((uint8_t)s); v.push_back((uint8_t)(s >> 8)); }
}
static void ijPut32(std::vector<uint8_t>& v, uint32_t l, bool be){
    if(be) for(int i = 3; i >= 0; i--) v.push_back((uint8_t)(l >> (8*i)));
    else   for(int i = 0; i <  4; i++) v.push_back((uint8_t)(l >> (8*i)));
}
static void ijEntry(std::vector<uint8_t>& v, uint16_t tag, uint16_t typ, uint32_t cnt, bool be){
    ijPut16(v, tag, be); ijPut16(v, typ, be); ijPut32(v, cnt, be);
}
static bool writeImageJTiff(const std::string& path, const uint8_t* data,
                            uint64_t x, uint64_t y, uint64_t z,
                            uint64_t spp, uint64_t bits, bool bigEndian){
    const uint64_t es = bits/8;
    std::string desc = "ImageJ=1.11a\nimages=" + std::to_string(z) +
                       "\nslices=" + std::to_string(z) + "\n";
    desc.push_back('\0');
    if(desc.size() % 2) desc.push_back('\0');

    const uint32_t nEntries = 11;
    const uint32_t auxOff = 8 + 2 + nEntries*12 + 4;
    const uint32_t bpsLen = (spp > 1) ? (uint32_t)(spp*2) : 0;
    const uint32_t descOff = auxOff + bpsLen;
    uint32_t pixOff = descOff + (uint32_t)desc.size();
    pixOff += pixOff % 2;

    std::vector<uint8_t> v;
    v.push_back(bigEndian ? 'M' : 'I'); v.push_back(bigEndian ? 'M' : 'I');
    ijPut16(v, 42, bigEndian); ijPut32(v, 8, bigEndian);
    ijPut16(v, nEntries, bigEndian);
    // A SHORT value sits in the first two bytes of the 4-byte value field.
    auto shortVal = [&](uint16_t s){ ijPut16(v, s, bigEndian); v.push_back(0); v.push_back(0); };
    auto longVal  = [&](uint32_t l){ ijPut32(v, l, bigEndian); };
    ijEntry(v,256,3,1,bigEndian); shortVal((uint16_t)x);            // ImageWidth
    ijEntry(v,257,3,1,bigEndian); shortVal((uint16_t)y);            // ImageLength
    ijEntry(v,258,3,(uint32_t)spp,bigEndian);                       // BitsPerSample
    if(spp == 1) shortVal((uint16_t)bits); else longVal(auxOff);
    ijEntry(v,259,3,1,bigEndian); shortVal(1);                      // Compression: none
    ijEntry(v,262,3,1,bigEndian); shortVal(spp == 3 ? 2 : 1);       // Photometric
    ijEntry(v,270,2,(uint32_t)desc.size(),bigEndian); longVal(descOff);
    ijEntry(v,273,4,1,bigEndian); longVal(pixOff);                  // StripOffsets
    ijEntry(v,277,3,1,bigEndian); shortVal((uint16_t)spp);          // SamplesPerPixel
    ijEntry(v,278,3,1,bigEndian); shortVal((uint16_t)y);            // RowsPerStrip
    ijEntry(v,279,4,1,bigEndian); longVal((uint32_t)(x*y*spp*es));  // StripByteCounts
    ijEntry(v,284,3,1,bigEndian); shortVal(1);                      // PlanarConfig: chunky
    ijPut32(v, 0, bigEndian);                                       // next IFD
    if(spp > 1) for(uint64_t i = 0; i < spp; i++) ijPut16(v, (uint16_t)bits, bigEndian);
    v.insert(v.end(), desc.begin(), desc.end());
    while(v.size() < pixOff) v.push_back(0);

    const uint64_t nElems = x*y*z*spp;
    if(es == 1 || !bigEndian){
        v.insert(v.end(), data, data + nElems*es);
    } else {
        for(uint64_t i = 0; i < nElems; i++)
            for(int b = (int)es - 1; b >= 0; b--) v.push_back(data[i*es + b]);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t w = std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return w == v.size();
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
    // 0 = striped. 16 gives a 3x2 grid over the 40x24 image, ragged in both
    // directions, so the padded edge tiles are exercised too.
    const uint64_t tiles[] = {0, 16};

    std::mt19937 rng(1234567u);
    std::uniform_int_distribution<int> byteDist(0, 255);

    int total = 0, failures = 0;
    for (const Case& c : cases){
        const uint64_t es = c.bits / 8;
        const uint64_t nbytes = n * es;

        std::vector<uint8_t> orig(nbytes);
        for (uint64_t i = 0; i < nbytes; i++) orig[i] = static_cast<uint8_t>(byteDist(rng));

        for (const char* comp : comps)
        for (const uint64_t tile : tiles){
            total++;
            char lbl[24];
            std::snprintf(lbl, sizeof lbl, "%s/%s", comp, tile ? "tile" : "strip");
            const std::string path = dir + "/rt_" + c.name + "_" + comp + "_t" + std::to_string(tile) + ".tif";

            step(c.name, lbl, "write");
            uint8_t werr = writeTiffParallelWrapper(x, y, z, path.c_str(), orig.data(), c.bits,
                                                    0, stripSize, "w", false,
                                                    std::string(comp), c.sampleFormat, tile);
            if (werr){ std::printf("FAIL  %-6s %-10s  (write error)\n", c.name, lbl); failures++; continue; }

            // Metadata should round trip.
            step(c.name, lbl, "read metadata");
            const uint64_t rbits = getDataType(path.c_str());
            const uint64_t rfmt  = getSampleFormat(path.c_str());
            uint64_t* dims = getImageSize(path.c_str());   // {y, x, z}
            const bool metaOK = rbits == c.bits && rfmt == c.sampleFormat &&
                                dims[0] == y && dims[1] == x && dims[2] == z;
            step(c.name, lbl, "free metadata buffer");
            free(dims);

            // No-flip read (the Python/C layout) must be byte-identical to what we wrote.
            step(c.name, lbl, "read no-flip");
            void* rbuf = readTiffParallelWrapperNoXYFlip(path.c_str(), {});
            const bool dataOK = rbuf && std::memcmp(rbuf, orig.data(), nbytes) == 0;
            step(c.name, lbl, "free no-flip buffer");
            free(rbuf);

            // Flip read (the MATLAB (y,x,z) column-major layout): compare against the
            // transpose of the original, exercising the flipTransposeStrip path.
            step(c.name, lbl, "read flip");
            std::vector<uint8_t> expFlip(nbytes);
            for (uint64_t zi = 0; zi < z; zi++)
                for (uint64_t yi = 0; yi < y; yi++)
                    for (uint64_t xi = 0; xi < x; xi++)
                        std::memcpy(&expFlip[(yi + xi*y + zi*x*y) * es],
                                    &orig[(zi*y*x + yi*x + xi) * es], es);
            void* fbuf = readTiffParallelWrapper(path.c_str());
            const bool flipOK = fbuf && std::memcmp(fbuf, expFlip.data(), nbytes) == 0;
            step(c.name, lbl, "free flip buffer");
            free(fbuf);

            std::remove(path.c_str());

            if (metaOK && dataOK && flipOK){
                std::printf("PASS  %-6s %-10s\n", c.name, lbl);
            } else {
                std::printf("FAIL  %-6s %-10s  (meta=%d noflip=%d flip=%d)\n",
                            c.name, lbl, (int)metaOK, (int)dataOK, (int)flipOK);
                failures++;
            }
        }
    }

    // ---- ImageJ-format files: RGB sample scaling, z-range offset, byte order.
    {
        const uint64_t ix = 40, iy = 24, iz = 3;

        // Chunky RGB uint8 with distinct per-channel patterns.
        std::vector<uint8_t> rgb(ix*iy*iz*3);
        for(uint64_t k = 0; k < iz; k++)
            for(uint64_t i = 0; i < iy; i++)
                for(uint64_t j = 0; j < ix; j++){
                    uint8_t* p = &rgb[((k*iy + i)*ix + j)*3];
                    p[0] = (uint8_t)((j*3 + k*7) % 250);
                    p[1] = (uint8_t)((i*5 + k*11) % 250);
                    p[2] = (uint8_t)((j + i*2 + k*13) % 250);
                }
        std::vector<uint16_t> g16(ix*iy*iz);
        for(uint64_t i = 0; i < g16.size(); i++) g16[i] = (uint16_t)((i*7 + 13) % 6000);

        struct IJCase { const char* name; const uint8_t* data; uint64_t spp, bits; bool be; };
        const IJCase ijCases[] = {
            {"ij-rgb",    rgb.data(),                 3, 8,  false},
            {"ij-g16-le", (const uint8_t*)g16.data(), 1, 16, false},
            {"ij-g16-be", (const uint8_t*)g16.data(), 1, 16, true},
        };
        for(const IJCase& c : ijCases){
            total++;
            const std::string path = dir + "/" + c.name + ".tif";
            const uint64_t es = c.bits/8;
            const uint64_t sliceBytes = ix*iy*c.spp*es;

            step(c.name, "imagej", "write fixture");
            if(!writeImageJTiff(path, c.data, ix, iy, iz, c.spp, c.bits, c.be)){
                std::printf("FAIL  %-9s (fixture write)\n", c.name); failures++; continue;
            }

            // No-flip read must reproduce the host-order source exactly.
            step(c.name, "imagej", "read no-flip");
            void* nf = readTiffParallelWrapperNoXYFlip(path.c_str(), {});
            const bool nfOK = nf && std::memcmp(nf, c.data, iz*sliceBytes) == 0;
            free(nf);

            // z-range [1,3) must start at slice 1, not slice 0.
            step(c.name, "imagej", "read z-range");
            void* zr = readTiffParallelWrapperNoXYFlip(path.c_str(), {1, 3});
            const bool zrOK = zr && std::memcmp(zr, c.data + sliceBytes, 2*sliceBytes) == 0;
            free(zr);

            // Flip read: column-major (y, x[, c], z), the layout the mex returns.
            step(c.name, "imagej", "read flip");
            std::vector<uint8_t> expFlip(iz*sliceBytes);
            for(uint64_t k = 0; k < iz; k++)
                for(uint64_t i = 0; i < iy; i++)
                    for(uint64_t j = 0; j < ix; j++)
                        for(uint64_t ch = 0; ch < c.spp; ch++)
                            std::memcpy(&expFlip[(i + j*iy + ch*ix*iy + k*ix*iy*c.spp)*es],
                                        c.data + (((k*iy + i)*ix + j)*c.spp + ch)*es, es);
            void* fl = readTiffParallelWrapper(path.c_str());
            const bool flOK = fl && std::memcmp(fl, expFlip.data(), expFlip.size()) == 0;
            free(fl);

            std::remove(path.c_str());
            if(nfOK && zrOK && flOK){
                std::printf("PASS  %-9s imagej\n", c.name);
            } else {
                std::printf("FAIL  %-9s imagej  (noflip=%d zrange=%d flip=%d)\n",
                            c.name, (int)nfOK, (int)zrOK, (int)flOK);
                failures++;
            }
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
