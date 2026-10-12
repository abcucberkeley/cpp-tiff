// Minimal smoke test for the cpp-tiff library: write small random volumes, read
// them back, and check the bytes survive the round trip. Covers every bit depth /
// sample format and each compression, plus the XY-flip reader (MATLAB layout).
//
// Exits 0 if every case passes, 1 otherwise, so CTest reports pass/fail.
// Usage: roundtripTest [output_dir]   (defaults to the current directory)
//
// Progress is written (flushed) to stderr before each operation so that if a
// build segfaults, the CTest log shows exactly which step/dtype/compression died.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <omp.h>
#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#endif

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

// ---- Plain multi-page fixture writer -----------------------------------------
// Classic TIFF, uncompressed, one strip per page, chunky samples; little-endian
// unless bigEndian (data is in host order and swapped as written). The library's
// writer only writes grayscale little-endian, so these fixtures are written here.
static bool writePlainTiff(const std::string& path, const uint8_t* data,
                           uint64_t x, uint64_t y, uint64_t z, uint64_t spp, uint64_t bits,
                           bool bigEndian = false){
    const bool be = bigEndian;
    const uint64_t es = bits/8;
    const uint64_t pageBytes = x*y*spp*(bits/8);
    const uint32_t nEntries = 10;
    const uint32_t ifdSize = 2 + nEntries*12 + 4;
    const uint32_t bpsLen = (spp > 1) ? (uint32_t)(spp*2) : 0;
    std::vector<uint8_t> v;
    v.push_back(be ? 'M' : 'I'); v.push_back(be ? 'M' : 'I'); ijPut16(v, 42, be); ijPut32(v, 8, be);
    for(uint64_t k = 0; k < z; k++){
        const uint32_t ifdOff = (uint32_t)v.size();
        const uint32_t bpsOff = ifdOff + ifdSize;
        const uint32_t pixOff = bpsOff + bpsLen;
        const uint32_t next = (k + 1 < z) ? (uint32_t)(pixOff + pageBytes) : 0;
        auto shortVal = [&](uint16_t sv){ ijPut16(v, sv, be); v.push_back(0); v.push_back(0); };
        auto longVal  = [&](uint32_t lv){ ijPut32(v, lv, be); };
        ijPut16(v, nEntries, be);
        ijEntry(v,256,3,1,be); shortVal((uint16_t)x);                  // ImageWidth
        ijEntry(v,257,3,1,be); shortVal((uint16_t)y);                  // ImageLength
        ijEntry(v,258,3,(uint32_t)spp,be);                             // BitsPerSample
        if(spp == 1) shortVal((uint16_t)bits); else longVal(bpsOff);
        ijEntry(v,259,3,1,be); shortVal(1);                            // Compression: none
        ijEntry(v,262,3,1,be); shortVal(spp == 3 ? 2 : 1);             // Photometric
        ijEntry(v,273,4,1,be); longVal(pixOff);                        // StripOffsets
        ijEntry(v,277,3,1,be); shortVal((uint16_t)spp);                // SamplesPerPixel
        ijEntry(v,278,3,1,be); shortVal((uint16_t)y);                  // RowsPerStrip
        ijEntry(v,279,4,1,be); longVal((uint32_t)pageBytes);           // StripByteCounts
        ijEntry(v,284,3,1,be); shortVal(1);                            // PlanarConfig: chunky
        ijPut32(v, next, be);                                          // next IFD
        if(spp > 1) for(uint64_t i = 0; i < spp; i++) ijPut16(v, (uint16_t)bits, be);
        if(!be || es == 1) v.insert(v.end(), data + k*pageBytes, data + (k + 1)*pageBytes);
        else for(uint64_t i = 0; i < pageBytes; i += es)
            for(int b = (int)es - 1; b >= 0; b--) v.push_back(data[k*pageBytes + i + b]);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t w = std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return w == v.size();
}

// Little-endian uncompressed grayscale with several strips per page stored in
// reverse order with gaps between them, a legal layout that cannot be read as
// one contiguous block, so the readers must fall back to libtiff per strip.
static bool writeReversedStripsTiff(const std::string& path, const uint8_t* data,
                                    uint64_t x, uint64_t y, uint64_t z, uint64_t bits, uint64_t rps){
    const uint64_t es = bits/8, rowBytes = x*es, nStrips = (y + rps - 1)/rps, gap = 16;
    const uint32_t nEntries = 10;
    std::vector<uint8_t> v;
    v.push_back('I'); v.push_back('I'); ijPut16(v, 42, false); ijPut32(v, 8, false);
    for(uint64_t k = 0; k < z; k++){
        const uint32_t ifdOff = (uint32_t)v.size();
        const uint32_t offsOff = ifdOff + 2 + nEntries*12 + 4, cntsOff = offsOff + 4*(uint32_t)nStrips;
        std::vector<uint32_t> offs(nStrips), cnts(nStrips);
        uint32_t pos = cntsOff + 4*(uint32_t)nStrips;
        for(uint64_t i = nStrips; i-- > 0;){
            cnts[i] = (uint32_t)(std::min<uint64_t>(rps, y - i*rps)*rowBytes);
            offs[i] = pos; pos += cnts[i] + gap;
        }
        auto shortVal = [&](uint16_t sv){ ijPut16(v, sv, false); v.push_back(0); v.push_back(0); };
        auto longVal  = [&](uint32_t lv){ ijPut32(v, lv, false); };
        ijPut16(v, nEntries, false);
        ijEntry(v,256,3,1,false); shortVal((uint16_t)x);                  // ImageWidth
        ijEntry(v,257,3,1,false); shortVal((uint16_t)y);                  // ImageLength
        ijEntry(v,258,3,1,false); shortVal((uint16_t)bits);               // BitsPerSample
        ijEntry(v,259,3,1,false); shortVal(1);                            // Compression: none
        ijEntry(v,262,3,1,false); shortVal(1);                            // Photometric
        ijEntry(v,273,4,(uint32_t)nStrips,false); longVal(offsOff);       // StripOffsets
        ijEntry(v,277,3,1,false); shortVal(1);                            // SamplesPerPixel
        ijEntry(v,278,3,1,false); shortVal((uint16_t)rps);                // RowsPerStrip
        ijEntry(v,279,4,(uint32_t)nStrips,false); longVal(cntsOff);       // StripByteCounts
        ijEntry(v,284,3,1,false); shortVal(1);                            // PlanarConfig: chunky
        ijPut32(v, (k + 1 < z) ? pos : 0, false);                         // next IFD
        for(uint32_t o : offs) ijPut32(v, o, false);
        for(uint32_t c : cnts) ijPut32(v, c, false);
        for(uint64_t i = nStrips; i-- > 0;){
            const uint8_t* src = data + k*y*rowBytes + i*rps*rowBytes;
            v.insert(v.end(), src, src + cnts[i]);
            v.insert(v.end(), (size_t)gap, (uint8_t)0);
        }
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t w = std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return w == v.size();
}

// Offset and byte count of one strip, for damaging it on purpose. Handles the
// little-endian classic and BigTIFF files this test writes.
static bool stripLocation(const std::string& path, uint64_t page, uint64_t strip,
                          uint64_t* off, uint64_t* cnt){
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) return false;
    auto rd = [&](uint64_t pos, uint8_t* dst, size_t n){
        return std::fseek(f, (long)pos, SEEK_SET) == 0 && std::fread(dst, 1, n, f) == n; };
    auto le = [](const uint8_t* b, int n){ uint64_t val = 0; for(int i = n - 1; i >= 0; i--) val = (val << 8) | b[i]; return val; };
    uint8_t h[16];
    bool ok = rd(0, h, 16) && h[0] == 'I' && h[1] == 'I';
    const bool big = ok && h[2] == 43;
    const int cntSz = big ? 8 : 2, entSz = big ? 20 : 12, offSz = big ? 8 : 4;
    uint64_t ifd = big ? le(h + 8, 8) : le(h + 4, 4);
    uint8_t b[20];
    for(uint64_t pg = 0; ok && pg < page; pg++){
        ok = rd(ifd, b, cntSz) && rd(ifd + cntSz + le(b, cntSz)*entSz, b, offSz);
        ifd = le(b, offSz);
    }
    const uint64_t n = (ok && rd(ifd, b, cntSz)) ? le(b, cntSz) : 0;
    bool haveOff = false, haveCnt = false;
    for(uint64_t e = 0; ok && e < n; e++){
        ok = rd(ifd + cntSz + e*entSz, b, entSz);
        const uint64_t tag = le(b, 2), typ = le(b + 2, 2), count = big ? le(b + 4, 8) : le(b + 4, 4);
        if(!ok || (tag != 273 && tag != 279)) continue;
        const int sz = typ == 16 ? 8 : typ == 4 ? 4 : typ == 3 ? 2 : 0;
        if(!sz || strip >= count){ ok = false; break; }
        const uint8_t* field = b + (big ? 12 : 8);
        uint8_t val[8];
        if(count*sz <= (uint64_t)offSz) std::memcpy(val, field + strip*sz, sz);
        else ok = rd(le(field, offSz) + strip*sz, val, sz);
        if(tag == 273){ *off = le(val, sz); haveOff = true; } else { *cnt = le(val, sz); haveCnt = true; }
    }
    std::fclose(f);
    return ok && haveOff && haveCnt;
}

// Drop the last `cut` bytes of a file, as an interrupted copy would.
static bool truncateFile(const std::string& path, uint64_t cut){
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) return false;
    std::vector<uint8_t> v;
    uint8_t buf[65536];
    size_t n;
    while((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    if(cut >= v.size()) return false;
    f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t keep = v.size() - cut;
    const size_t w = std::fwrite(v.data(), 1, keep, f);
    std::fclose(f);
    return w == keep;
}

static std::vector<uint8_t> readFileBytes(const std::string& path){
    std::vector<uint8_t> v;
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) return v;
    uint8_t buf[65536];
    size_t n;
    while((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    return v;
}

// RowsPerStrip of a file's first directory, read from the bytes (0 if absent),
// so the test needs no libtiff headers. Little-endian TIFF or BigTIFF.
static uint64_t firstRowsPerStrip(const std::vector<uint8_t>& f){
    auto u = [&](uint64_t off, int n){
        uint64_t v = 0;
        for(int i = n - 1; i >= 0; i--) v = (v << 8) | f[off + i];
        return v;
    };
    if(f.size() < 16 || f[0] != 'I' || f[1] != 'I') return 0;
    const bool big = u(2, 2) == 43;
    const uint64_t ifd = big ? u(8, 8) : u(4, 4);
    if(ifd + 8 > f.size()) return 0;
    const uint64_t n = big ? u(ifd, 8) : u(ifd, 2), size = big ? 20 : 12, first = ifd + (big ? 8 : 2);
    for(uint64_t e = 0; e < n && first + (e + 1)*size <= f.size(); e++){
        const uint64_t p = first + e*size;
        if(u(p, 2) != 278) continue;
        const uint64_t type = u(p + 2, 2), val = p + (big ? 12 : 8);
        return type == 3 ? u(val, 2) : type == 4 ? u(val, 4) : type == 16 ? u(val, 8) : 0;
    }
    return 0;
}

// Column-major MATLAB layout (y, x[, c], z) of a chunky row-major source.
static std::vector<uint8_t> flipped(const uint8_t* src, uint64_t x, uint64_t y, uint64_t z,
                                    uint64_t spp, uint64_t es){
    std::vector<uint8_t> out(x*y*z*spp*es);
    for(uint64_t k = 0; k < z; k++)
        for(uint64_t i = 0; i < y; i++)
            for(uint64_t j = 0; j < x; j++)
                for(uint64_t c = 0; c < spp; c++)
                    std::memcpy(&out[(i + j*y + c*x*y + k*x*y*spp)*es],
                                src + (((k*y + i)*x + j)*spp + c)*es, es);
    return out;
}

// The scanline fallbacks are internal: the readers only use them when a strip
// fails to decode, so valid files never reach them. Call them directly.
uint8_t readTiffParallelBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY);
uint8_t readTiffParallel2DBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY);

// Run one fallback read into a poisoned, guarded buffer: unwritten bytes keep
// the poison and any write past the end changes the guard.
static bool fallbackMatches(bool twoD, const std::string& path, uint64_t x, uint64_t y, uint64_t z,
                            uint64_t spp, uint64_t bits, uint64_t startSlice, bool flip,
                            const std::vector<uint8_t>& expected){
    const size_t guard = 4096;
    std::vector<uint8_t> out(expected.size() + guard, 0xA5);
    const uint8_t err = twoD ? readTiffParallel2DBak(x, y, z, path.c_str(), out.data(), bits, startSlice, flip)
                             : readTiffParallelBak(x, y, z, path.c_str(), out.data(), bits, startSlice, flip);
    bool guardOK = true;
    for(size_t i = expected.size(); i < out.size(); i++) guardOK = guardOK && out[i] == 0xA5;
    (void)spp;
    return !err && guardOK && std::memcmp(out.data(), expected.data(), expected.size()) == 0;
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

    // ---- Scanline fallbacks, called directly on valid files. Wide and tall
    // images (the old row offset only ran past the buffer when x < y), 3D, 2D
    // and an offset z-range, both layouts, grayscale from the library's writer
    // and hand-written RGB.
    {
        struct FB { const char* name; uint64_t x, y, spp, bits; uint16_t fmt; const char* comp; };
        const FB fbs[] = {
            {"u16-lzw-wide",  40, 24, 1, 16, 1, "lzw"},
            {"u16-lzw-tall",  24, 40, 1, 16, 1, "lzw"},
            {"u8-none-tall",  24, 40, 1,  8, 1, "none"},
            {"f32-zstd-wide", 40, 24, 1, 32, 3, "zstd"},
            {"rgb8-wide",     40, 24, 3,  8, 1, nullptr},
            {"rgb8-tall",     24, 40, 3,  8, 1, nullptr},
        };
        std::mt19937 frng(7654321u);
        std::uniform_int_distribution<int> fbyte(0, 255);
        for(const FB& c : fbs){
            const uint64_t fz = 3, es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*fz);
            for(auto& v8 : src) v8 = (uint8_t)fbyte(frng);
            if(c.bits == 32) for(uint64_t i = 0; i < src.size(); i += 4) src[i + 3] &= 0x3F;   // keep floats finite
            const std::string p3 = dir + "/fb3_" + c.name + ".tif", p2 = dir + "/fb2_" + c.name + ".tif";
            step(c.name, "fallback", "write fixtures");
            bool wrote;
            if(c.comp) wrote = !writeTiffParallelWrapper(c.x, c.y, fz, p3.c_str(), src.data(), c.bits, 0, 8, "w", false, std::string(c.comp), c.fmt, 0)
                            && !writeTiffParallelWrapper(c.x, c.y, 1, p2.c_str(), src.data(), c.bits, 0, 8, "w", false, std::string(c.comp), c.fmt, 0);
            else       wrote = writePlainTiff(p3, src.data(), c.x, c.y, fz, c.spp, c.bits)
                            && writePlainTiff(p2, src.data(), c.x, c.y, 1, c.spp, c.bits);
            total++;
            if(!wrote){ std::printf("FAIL  %-14s fallback (fixture write)\n", c.name); failures++; continue; }

            const std::vector<uint8_t> all(src.begin(), src.end());
            const std::vector<uint8_t> tail(src.begin() + slice, src.end());          // slices 1..2
            const std::vector<uint8_t> first(src.begin(), src.begin() + slice);       // slice 0
            step(c.name, "fallback", "3D no-flip / flip / z-range");
            const bool b3  = fallbackMatches(false, p3, c.x, c.y, fz, c.spp, c.bits, 0, false, all)
                          && fallbackMatches(false, p3, c.x, c.y, fz, c.spp, c.bits, 0, true, flipped(src.data(), c.x, c.y, fz, c.spp, es))
                          && fallbackMatches(false, p3, c.x, c.y, 2, c.spp, c.bits, 1, false, tail)
                          && fallbackMatches(false, p3, c.x, c.y, 2, c.spp, c.bits, 1, true, flipped(src.data() + slice, c.x, c.y, 2, c.spp, es));
            step(c.name, "fallback", "2D no-flip / flip");
            const bool b2  = fallbackMatches(true, p2, c.x, c.y, 1, c.spp, c.bits, 0, false, first)
                          && fallbackMatches(true, p2, c.x, c.y, 1, c.spp, c.bits, 0, true, flipped(src.data(), c.x, c.y, 1, c.spp, es));
            // The normal readers must agree on the same files (RGB fixtures included).
            step(c.name, "fallback", "normal readers");
            void* nf = readTiffParallelWrapperNoXYFlip(p3.c_str(), {});
            void* fl = readTiffParallelWrapper(p3.c_str());
            const std::vector<uint8_t> f3 = flipped(src.data(), c.x, c.y, fz, c.spp, es);
            const bool rd = nf && fl && std::memcmp(nf, all.data(), all.size()) == 0 && std::memcmp(fl, f3.data(), f3.size()) == 0;
            free(nf); free(fl);
            std::remove(p3.c_str()); std::remove(p2.c_str());
            if(b3 && b2 && rd) std::printf("PASS  %-14s fallback\n", c.name);
            else { std::printf("FAIL  %-14s fallback  (3D=%d 2D=%d readers=%d)\n", c.name, (int)b3, (int)b2, (int)rd); failures++; }
        }
    }

    // ---- Short strips. Flip reads decode runs of short strips and transpose
    // each run at once (stripsPerTranspose). Odd heights leave a partial last
    // strip and a partial last run. Wide uncompressed single-strip images are
    // split by libtiff into one-row strips on read, including RGB.
    {
        struct SS { const char* name; uint64_t x, y, z, spp, bits; uint16_t fmt; const char* comp; };
        const SS sss[] = {
            {"u8-none-rps3",    40,   41, 3, 1,  8, 1, "none"},
            {"u16-lzw-rps3",    40,   41, 3, 1, 16, 1, "lzw"},
            {"f32-none-rps3",   40,   41, 3, 1, 32, 3, "none"},
            {"f64-zstd-rps3",   40,   41, 3, 1, 64, 3, "zstd"},
            {"u16-onerow",    4100,   37, 2, 1, 16, 1, nullptr},
            {"rgb8-onerow",   1500,   37, 2, 3,  8, 1, nullptr},
        };
        std::mt19937 srng(24680u);
        std::uniform_int_distribution<int> sbyte(0, 255);
        for(const SS& c : sss){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*c.z);
            for(auto& v8 : src) v8 = (uint8_t)sbyte(srng);
            if(c.bits == 32) for(uint64_t i = 0; i < src.size(); i += 4) src[i + 3] &= 0x3F;   // keep floats finite
            if(c.bits == 64) for(uint64_t i = 0; i < src.size(); i += 8) src[i + 7] &= 0x3F;
            const std::string p3 = dir + "/ss3_" + c.name + ".tif", p2 = dir + "/ss2_" + c.name + ".tif";
            step(c.name, "short strips", "write fixtures");
            bool ok = c.comp ? !writeTiffParallelWrapper(c.x, c.y, c.z, p3.c_str(), src.data(), c.bits, 0, 3, "w", false, std::string(c.comp), c.fmt, 0)
                               && !writeTiffParallelWrapper(c.x, c.y, 1, p2.c_str(), src.data(), c.bits, 0, 3, "w", false, std::string(c.comp), c.fmt, 0)
                             : writePlainTiff(p3, src.data(), c.x, c.y, c.z, c.spp, c.bits)
                               && writePlainTiff(p2, src.data(), c.x, c.y, 1, c.spp, c.bits);
            step(c.name, "short strips", "3D, 2D and z-range reads, both layouts");
            struct Read { const std::string* path; std::vector<uint64_t> range; uint64_t first, count; };
            const Read reads[] = {{&p3, {}, 0, c.z}, {&p3, {1, c.z}, 1, c.z - 1}, {&p3, {1}, 1, 1}, {&p2, {}, 0, 1}};
            for(const Read& r : reads){
                if(!ok) break;
                const uint8_t* want = src.data() + r.first*slice;
                const std::vector<uint8_t> wantFlip = flipped(want, c.x, c.y, r.count, c.spp, es);
                void* nf = readTiffParallelWrapperNoXYFlip(r.path->c_str(), r.range);
                void* fl = readTiffParallelWrapper(r.path->c_str(), r.range);
                ok = nf && fl && std::memcmp(nf, want, r.count*slice) == 0
                     && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
                free(nf); free(fl);
            }
            std::remove(p3.c_str()); std::remove(p2.c_str());
            if(ok) std::printf("PASS  %-14s short strips\n", c.name);
            else { std::printf("FAIL  %-14s short strips\n", c.name); failures++; }
        }
    }

    // ---- Many slices, few threads. With more slices than threads each thread
    // reads its own block and then takes slices from the back of the others
    // (Linux), so every slice must still be read exactly once, in its place.
    {
        struct MS { const char* name; const char* comp; };
        const MS mss[] = {{"u16-none-37z", "none"}, {"u16-lzw-37z", "lzw"}};
        const uint64_t mx = 24, my = 17, mz = 37, slice = mx*my;
        std::vector<uint16_t> ms(slice*mz);
        for(uint64_t i = 0; i < ms.size(); i++) ms[i] = (uint16_t)((i*31 + 7) % 65000);
        const std::vector<uint8_t> wantFlip = flipped((const uint8_t*)ms.data(), mx, my, mz, 1, 2);
        const std::vector<uint8_t> wantRangeFlip = flipped((const uint8_t*)(ms.data() + 3*slice), mx, my, 27, 1, 2);
        const int prevThreads = omp_get_max_threads();
        for(const MS& c : mss){
            total++;
            const std::string mp = dir + "/many_" + c.name + ".tif";
            step(c.name, "many slices", "write fixture");
            bool ok = !writeTiffParallelWrapper(mx, my, mz, mp.c_str(), ms.data(), 16, 0, 5, "w", false, std::string(c.comp), 1, 0);
            const int threads[] = {2, 3, 5, prevThreads};
            for(int t : threads){
                if(!ok) break;
                omp_set_num_threads(t);
                step(c.name, "many slices", "full and z-range reads, both layouts");
                void* nf = readTiffParallelWrapperNoXYFlip(mp.c_str(), {});
                void* fl = readTiffParallelWrapper(mp.c_str());
                void* rg = readTiffParallelWrapperNoXYFlip(mp.c_str(), {3, 30});
                void* rf = readTiffParallelWrapper(mp.c_str(), {3, 30});
                ok = nf && fl && rg && rf
                     && std::memcmp(nf, ms.data(), ms.size()*2) == 0 && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0
                     && std::memcmp(rg, ms.data() + 3*slice, 27*slice*2) == 0 && std::memcmp(rf, wantRangeFlip.data(), wantRangeFlip.size()) == 0;
                free(nf); free(fl); free(rg); free(rf);
            }
            omp_set_num_threads(prevThreads);
            std::remove(mp.c_str());
            if(ok) std::printf("PASS  %-14s many slices\n", c.name);
            else { std::printf("FAIL  %-14s many slices\n", c.name); failures++; }
        }
    }

    // ---- Big-endian uncompressed stacks. Each page is larger than libtiff's
    // ~8 KB strip size, so libtiff splits it into short strips and the readers
    // copy them straight from the file (readRawStrips), swapping bytes the way
    // libtiff would.
    {
        struct BE { const char* name; uint64_t spp, bits; };
        const BE bes[] = {{"be-u8", 1, 8}, {"be-u16", 1, 16}, {"be-32bit", 1, 32}, {"be-64bit", 1, 64}, {"be-rgb16", 3, 16}};
        const uint64_t bx = 300, by = 41, bz = 3;
        std::mt19937 brng(13579u);
        std::uniform_int_distribution<int> bbyte(0, 255);
        for(const BE& c : bes){
            total++;
            const uint64_t es = c.bits/8, slice = bx*by*c.spp*es;
            std::vector<uint8_t> src(slice*bz);
            for(auto& v8 : src) v8 = (uint8_t)bbyte(brng);
            const std::string bp = dir + "/" + c.name + ".tif";
            step(c.name, "big-endian", "write fixture");
            bool ok = writePlainTiff(bp, src.data(), bx, by, bz, c.spp, c.bits, true);
            step(c.name, "big-endian", "full, z-range and single-slice reads, both layouts");
            struct Read { std::vector<uint64_t> range; uint64_t first, count; };
            const Read reads[] = {{{}, 0, bz}, {{1, bz}, 1, bz - 1}, {{2}, 2, 1}};
            for(const Read& r : reads){
                if(!ok) break;
                const uint8_t* want = src.data() + r.first*slice;
                const std::vector<uint8_t> wantFlip = flipped(want, bx, by, r.count, c.spp, es);
                void* nf = readTiffParallelWrapperNoXYFlip(bp.c_str(), r.range);
                void* fl = readTiffParallelWrapper(bp.c_str(), r.range);
                ok = nf && fl && std::memcmp(nf, want, r.count*slice) == 0
                     && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
                free(nf); free(fl);
            }
            std::remove(bp.c_str());
            if(ok) std::printf("PASS  %-14s big-endian\n", c.name);
            else { std::printf("FAIL  %-14s big-endian\n", c.name); failures++; }
        }

        // Strips stored out of order cannot be copied as one block.
        total++;
        const uint64_t rx = 300, ry = 37, rz = 3, rslice = rx*ry*2;
        std::vector<uint8_t> rsrc(rslice*rz);
        for(auto& v8 : rsrc) v8 = (uint8_t)bbyte(brng);
        const std::string rp = dir + "/reversed_strips.tif";
        step("reversed", "strip order", "write + reads");
        bool rok = writeReversedStripsTiff(rp, rsrc.data(), rx, ry, rz, 16, 4);
        for(int flip = 0; rok && flip < 2; flip++){
            const std::vector<uint8_t> want = flip ? flipped(rsrc.data(), rx, ry, rz, 1, 2) : rsrc;
            const std::vector<uint8_t> want1 = flip ? flipped(rsrc.data() + rslice, rx, ry, 1, 1, 2)
                                                    : std::vector<uint8_t>(rsrc.begin() + rslice, rsrc.begin() + 2*rslice);
            void* r = flip ? readTiffParallelWrapper(rp.c_str()) : readTiffParallelWrapperNoXYFlip(rp.c_str(), {});
            void* r1 = flip ? readTiffParallelWrapper(rp.c_str(), {1}) : readTiffParallelWrapperNoXYFlip(rp.c_str(), {1});
            rok = r && r1 && std::memcmp(r, want.data(), want.size()) == 0 && std::memcmp(r1, want1.data(), want1.size()) == 0;
            free(r); free(r1);
        }
        std::remove(rp.c_str());
        if(rok) std::printf("PASS  %-14s strip order\n", "reversed");
        else { std::printf("FAIL  %-14s strip order\n", "reversed"); failures++; }
    }

    // ---- One OpenMP thread. Single-image reads used to take a separate raw-read
    // path there, which always returned slice 0 of an uncompressed stack.
    {
        total++;
        const uint64_t sx = 40, sy = 24, sz = 4, slice = sx*sy;
        std::vector<uint16_t> st(slice*sz);
        for(uint64_t i = 0; i < st.size(); i++) st[i] = (uint16_t)((i*7 + 3) % 50000);
        const std::string sp = dir + "/one_thread.tif";
        step("one-thread", "none", "write fixture");
        bool ok = writePlainTiff(sp, (const uint8_t*)st.data(), sx, sy, sz, 1, 16);
        const int prevThreads = omp_get_max_threads();
        omp_set_num_threads(1);
        step("one-thread", "none", "single-slice reads, both layouts");
        for(uint64_t k = 0; ok && k < sz; k++){
            const uint16_t* want = st.data() + k*slice;
            const std::vector<uint8_t> wantFlip = flipped((const uint8_t*)want, sx, sy, 1, 1, 2);
            void* one = readTiffParallelWrapperNoXYFlip(sp.c_str(), {k});
            void* range = readTiffParallelWrapperNoXYFlip(sp.c_str(), {k, k + 1});
            void* flip = readTiffParallelWrapper(sp.c_str(), {k, k + 1});
            ok = one && range && flip
                 && std::memcmp(one, want, slice*2) == 0 && std::memcmp(range, want, slice*2) == 0
                 && std::memcmp(flip, wantFlip.data(), wantFlip.size()) == 0;
            free(one); free(range); free(flip);
        }
        step("one-thread", "none", "full read");
        void* all = ok ? readTiffParallelWrapperNoXYFlip(sp.c_str(), {}) : nullptr;
        ok = all && std::memcmp(all, st.data(), st.size()*2) == 0;
        free(all);
        omp_set_num_threads(prevThreads);
        std::remove(sp.c_str());
        if(ok) std::printf("PASS  %-14s one thread\n", "u16-none");
        else { std::printf("FAIL  %-14s one thread  (single-slice or full read wrong)\n", "u16-none"); failures++; }
    }

    // ---- Writer layouts. MATLAB-layout input (column-major, transposed strip by
    // strip as it is written) must give the same file as row-major input, a
    // single-threaded write must give the same file as a parallel one, and the
    // file must read back to the source.
    {
        struct WL { const char* name; uint64_t x, y, z, bits; uint16_t fmt; const char* comp; uint64_t strip, tile; };
        const WL wls[] = {
            {"u8-none-s7",    40, 41, 3,  8, 1, "none",   7,  0},
            {"u16-lzw-s512", 333, 77, 4, 16, 1, "lzw",  512,  0},
            {"f32-zstd-s5",   61, 29, 3, 32, 3, "zstd",   5,  0},
            {"f64-lzw-t16",   50, 37, 2, 64, 3, "lzw",  512, 16},
            {"u16-none-t32",  70, 45, 3, 16, 1, "none", 512, 32},
            {"u8-zstd-t16",   33, 70, 2,  8, 1, "zstd", 512, 16},
        };
        std::mt19937 wrng(424242u);
        std::uniform_int_distribution<int> wbyte(0, 255);
        const int prevThreads = omp_get_max_threads();
        for(const WL& c : wls){
            total++;
            const uint64_t es = c.bits/8;
            std::vector<uint8_t> rm(c.x*c.y*c.z*es);
            for(auto& v8 : rm) v8 = (uint8_t)(wbyte(wrng) & 0x3F);   // compressible, and floats stay finite
            const std::vector<uint8_t> cm = flipped(rm.data(), c.x, c.y, c.z, 1, es);
            const std::string pa = dir + "/wl_row_" + c.name + ".tif", pb = dir + "/wl_matlab_" + c.name + ".tif",
                              pc = dir + "/wl_one_" + c.name + ".tif";
            step(c.name, "writer layouts", "row-major, MATLAB-layout and one-thread writes");
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, pa.c_str(), rm.data(), c.bits, 0, c.strip, "w", false, std::string(c.comp), c.fmt, c.tile)
                      && !writeTiffParallelWrapper(c.x, c.y, c.z, pb.c_str(), cm.data(), c.bits, 0, c.strip, "w", true, std::string(c.comp), c.fmt, c.tile);
            omp_set_num_threads(1);
            ok = ok && !writeTiffParallelWrapper(c.x, c.y, c.z, pc.c_str(), cm.data(), c.bits, 0, c.strip, "w", true, std::string(c.comp), c.fmt, c.tile);
            omp_set_num_threads(prevThreads);
            const std::vector<uint8_t> fa = readFileBytes(pa), fb = readFileBytes(pb), fc = readFileBytes(pc);
            ok = ok && !fa.empty() && fa == fb && fa == fc;
            void* back = ok ? readTiffParallelWrapperNoXYFlip(pa.c_str(), {}) : nullptr;
            ok = ok && back && std::memcmp(back, rm.data(), rm.size()) == 0;
            free(back);
            std::remove(pa.c_str()); std::remove(pb.c_str()); std::remove(pc.c_str());
            if(ok) std::printf("PASS  %-14s writer layouts\n", c.name);
            else { std::printf("FAIL  %-14s writer layouts  (files differ or read back wrong)\n", c.name); failures++; }
        }
    }

    // ---- Automatic strip size (strip size 0, what the Python and MATLAB writers
    // pass). 2D images and short stacks get smaller strips so more threads can
    // compress them; tall stacks keep 512 rows. Only the image size decides, so
    // one thread and many must give the same file, and it must read back.
    {
        struct AS { const char* name; uint64_t x, y, z, bits; const char* comp; uint64_t rows; };
        const AS cases[] = {
            {"auto-2d",    2048, 2048,   1, 16, "lzw",   16},   // 8 rows would give 256 strips; 64 KB minimum
            {"auto-short", 2000, 1500,   3, 16, "zstd",  18},   // 86 strips per slice
            {"auto-tall",   256,  700, 260,  8, "none", 512},   // already enough strips
        };
        std::mt19937 arng(77u);
        const int prevThreads = omp_get_max_threads();
        for(const AS& c : cases){
            total++;
            const uint64_t es = c.bits/8;
            std::vector<uint8_t> src(c.x*c.y*c.z*es);
            for(auto& v8 : src) v8 = (uint8_t)(arng() & 0x3F);
            const std::string pa = dir + "/" + c.name + "_n.tif", pb = dir + "/" + c.name + "_1.tif";
            step(c.name, "automatic strips", "strip size 0 at full and one thread, read back both ways");
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, pa.c_str(), src.data(), c.bits, 0, 0, "w", false, std::string(c.comp), 1, 0);
            omp_set_num_threads(1);
            ok = ok && !writeTiffParallelWrapper(c.x, c.y, c.z, pb.c_str(), src.data(), c.bits, 0, 0, "w", false, std::string(c.comp), 1, 0);
            omp_set_num_threads(prevThreads);
            const std::vector<uint8_t> fa = readFileBytes(pa), fb = readFileBytes(pb);
            const uint64_t rows = firstRowsPerStrip(fa);
            ok = ok && !fa.empty() && fa == fb && rows == c.rows;
            void* back = ok ? readTiffParallelWrapperNoXYFlip(pa.c_str(), {}) : nullptr;
            ok = ok && back && std::memcmp(back, src.data(), src.size()) == 0;
            free(back);
            if(ok){
                const std::vector<uint8_t> cm = flipped(src.data(), c.x, c.y, c.z, 1, es);
                void* fl = readTiffParallelWrapper(pa.c_str());
                ok = fl && std::memcmp(fl, cm.data(), cm.size()) == 0;
                free(fl);
            }
            std::remove(pa.c_str()); std::remove(pb.c_str());
            if(ok) std::printf("PASS  %-14s automatic strips (%llu rows)\n", c.name, (unsigned long long)rows);
            else { std::printf("FAIL  %-14s automatic strips  (%llu rows per strip, expected %llu; or the files differ or read back wrong)\n",
                               c.name, (unsigned long long)rows, (unsigned long long)c.rows); failures++; }
        }
    }

    // ---- Short stacks (a few slices with many strips each) are read slice by
    // slice with every thread instead of one thread per slice. Grayscale stacks
    // written by the library, an RGB fixture, and a z-range must match the
    // source in both layouts.
    {
        const int prevThreads = omp_get_max_threads();
        omp_set_num_threads(8);   // 2 x slices <= threads, so the slice-by-slice path is taken
        struct SS { const char* name; uint64_t x, y, z, spp, bits; const char* comp; };
        // Slices must be at least 4 MB (16 MB uncompressed) to be read this way.
        const SS cases[] = {
            {"short-u16-lzw",  2048, 1024, 3, 1, 16, "lzw"},
            {"short-u8-none",  4096, 4096, 3, 1,  8, "none"},
            {"short-rgb",      2400, 2400, 2, 3,  8, nullptr},   // fixture, one strip per page split by libtiff
        };
        std::mt19937 srng(31u);
        for(const SS& c : cases){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*c.z);
            for(auto& v8 : src) v8 = (uint8_t)srng();
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "short stacks", "read slice by slice in both layouts and a z-range");
            bool ok = c.comp ? !writeTiffParallelWrapper(c.x, c.y, c.z, path.c_str(), src.data(), c.bits, 0, 64, "w", false, std::string(c.comp), 1, 0)
                             : writePlainTiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits);
            void* rm = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
            ok = ok && rm && std::memcmp(rm, src.data(), src.size()) == 0;
            free(rm);
            void* fl = ok ? readTiffParallelWrapper(path.c_str()) : nullptr;
            if(fl){
                const std::vector<uint8_t> cm = flipped(src.data(), c.x, c.y, c.z, c.spp, es);
                ok = std::memcmp(fl, cm.data(), cm.size()) == 0;
            }
            else ok = false;
            free(fl);
            void* part = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {1, c.z}) : nullptr;
            ok = ok && part && std::memcmp(part, src.data() + slice, slice*(c.z - 1)) == 0;
            free(part);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-14s short stacks\n", c.name);
            else { std::printf("FAIL  %-14s short stacks  (read back wrong)\n", c.name); failures++; }
        }
        omp_set_num_threads(prevThreads);
    }

    // ---- Write failures must be reported instead of leaving a broken file and
    // returning success.
    {
        const uint64_t fx = 128, fy = 128, fz = 8;
        std::vector<uint16_t> fv(fx*fy*fz);
        for(uint64_t i = 0; i < fv.size(); i++) fv[i] = (uint16_t)(i*2654435761u >> 7);
        // A folder that cannot exist: the path runs through a regular file.
        total++;
        const std::string blocker = dir + "/not_a_folder.txt";
        if(FILE* f = std::fopen(blocker.c_str(), "wb")){ std::fputs("x", f); std::fclose(f); }
        const std::string inside = blocker + "/sub/out.tif";
        step("bad-folder", "write failures", "write below a regular file");
        const bool badFolder = writeTiffParallelHelper(inside.c_str(), fv.data(), 16, "w", fx, fy, fz, 0, false, "lzw", 1, 0) != 0;
        std::remove(blocker.c_str());
        if(badFolder) std::printf("PASS  %-14s write failures\n", "bad-folder");
        else { std::printf("FAIL  %-14s write failures  (reported success)\n", "bad-folder"); failures++; }
#ifdef __linux__
        // A device that rejects every write.
        total++;
        step("dev-full", "write failures", "write to /dev/full");
        const bool devFull = writeTiffParallelWrapper(fx, fy, fz, "/dev/full", fv.data(), 16, 0, 16, "w", false, "none", 1, 0) != 0;
        if(devFull) std::printf("PASS  %-14s write failures\n", "dev-full");
        else { std::printf("FAIL  %-14s write failures  (reported success)\n", "dev-full"); failures++; }
#endif
#ifndef _WIN32
        // Running out of space partway through: lower the file-size limit so
        // writes past it fail (with SIGXFSZ ignored they return an error instead
        // of killing the process), in both the parallel and one-thread writers.
        const std::string partial = dir + "/partial_write.tif";
        struct rlimit oldLimit;
        getrlimit(RLIMIT_FSIZE, &oldLimit);
        void (*oldHandler)(int) = std::signal(SIGXFSZ, SIG_IGN);
        const int prevThreads = omp_get_max_threads();
        const int threadCounts[] = {prevThreads, 1};
        for(int t : threadCounts){
            for(const char* comp : {"none", "zstd"}){
                total++;
                struct rlimit lim = oldLimit;
                lim.rlim_cur = 64*1024;   // the volume is 256 KB uncompressed, well above this compressed
                step("partial", comp, "write past a 64 KB file-size limit");
                omp_set_num_threads(t);
                std::vector<uint16_t> noisy(fv.size());
                std::mt19937 nrng(7u);
                for(auto& v16 : noisy) v16 = (uint16_t)nrng();   // incompressible, so zstd output exceeds the limit too
                setrlimit(RLIMIT_FSIZE, &lim);
                const bool reported = writeTiffParallelWrapper(fx, fy, fz, partial.c_str(), noisy.data(), 16, 0, 16, "w", false, std::string(comp), 1, 0) != 0;
                setrlimit(RLIMIT_FSIZE, &oldLimit);
                omp_set_num_threads(prevThreads);
                std::remove(partial.c_str());
                if(reported) std::printf("PASS  partial-%-6s t%-3d write failures\n", comp, t);
                else { std::printf("FAIL  partial-%-6s t%-3d write failures  (reported success)\n", comp, t); failures++; }
            }
        }
        std::signal(SIGXFSZ, oldHandler);
#endif
    }

    // ---- Unreadable and damaged files: report failure, never crash.
    {
        const std::string missing = dir + "/does_not_exist.tif";
        const std::string notTiff = dir + "/not_a_tiff.tif";
        if(FILE* f = std::fopen(notTiff.c_str(), "wb")){
            const char line[] = "this is not a tiff file\n";
            for(int i = 0; i < 100; i++) std::fwrite(line, 1, sizeof line - 1, f);
            std::fclose(f);
        }
        const std::string bad[] = {missing, notTiff};
        for(const std::string& p : bad){
            total++;
            const char* nm = (&p == &bad[0]) ? "missing" : "not-a-tiff";
            step(nm, "damaged", "metadata + reads");
            uint64_t* d = getImageSize(p.c_str());
            const bool ok = !d && getDataType(p.c_str()) == 0 && getImageSizeZ(p.c_str()) == 0
                            && !readTiffParallelWrapperNoXYFlip(p.c_str(), {})
                            && !readTiffParallelWrapper(p.c_str());
            free(d);
            if(ok) std::printf("PASS  %-14s damaged\n", nm);
            else { std::printf("FAIL  %-14s damaged  (expected NULL/0 everywhere)\n", nm); failures++; }
        }
        std::remove(notTiff.c_str());

        // Damage one LZW strip of page 2: reading that page must fail cleanly in
        // both layouts and through the single-image path, and the undamaged
        // pages must still read correctly.
        total++;
        const uint64_t cx = 64, cy = 48, cz = 4;
        std::vector<uint16_t> vol(cx*cy*cz);
        std::mt19937 crng(97531u);
        for(auto& v16 : vol) v16 = (uint16_t)(crng() & 0xFFFF);
        const std::string cp = dir + "/corrupt_strip.tif";
        step("corrupt", "damaged", "write + damage page 2 strip 1");
        uint64_t off = 0, cnt = 0;
        bool ok = !writeTiffParallelWrapper(cx, cy, cz, cp.c_str(), vol.data(), 16, 0, 16, "w", false, "lzw", 1, 0)
                  && stripLocation(cp, 2, 1, &off, &cnt) && cnt > 64;
        if(ok){
            FILE* f = std::fopen(cp.c_str(), "r+b");
            std::vector<uint8_t> junk(std::min<uint64_t>(256, cnt/2), 0xFF);
            ok = f && std::fseek(f, (long)(off + cnt/4), SEEK_SET) == 0 && std::fwrite(junk.data(), 1, junk.size(), f) == junk.size();
            if(f) std::fclose(f);
        }
        step("corrupt", "damaged", "reads");
        void* r3 = ok ? readTiffParallelWrapperNoXYFlip(cp.c_str(), {}) : nullptr;
        void* rf = ok ? readTiffParallelWrapper(cp.c_str()) : nullptr;
        void* r2 = ok ? readTiffParallelWrapperNoXYFlip(cp.c_str(), {2, 3}) : nullptr;
        void* rg = ok ? readTiffParallelWrapperNoXYFlip(cp.c_str(), {0, 2}) : nullptr;
        const bool good = rg && std::memcmp(rg, vol.data(), 2*cx*cy*sizeof(uint16_t)) == 0;
        const bool pass = ok && !r3 && !rf && !r2 && good;
        free(r3); free(rf); free(r2); free(rg);
        std::remove(cp.c_str());
        if(pass) std::printf("PASS  %-14s damaged\n", "corrupt-strip");
        else { std::printf("FAIL  %-14s damaged  (setup=%d full=%d flip=%d page=%d undamaged=%d)\n", "corrupt-strip",
                           (int)ok, (int)!r3, (int)!rf, (int)!r2, (int)good); failures++; }

        // An ImageJ stack cut short, as an interrupted copy leaves it. Reading
        // past the end must fail instead of returning unread memory (or, past
        // 2 GB, retrying forever), while the complete slices still read.
        {
            total++;
            const uint64_t tx = 40, ty = 24, tz = 6, slice = tx*ty, sliceBytes = slice*2;
            std::vector<uint16_t> ij(slice*tz);
            for(uint64_t i = 0; i < ij.size(); i++) ij[i] = (uint16_t)((i*13 + 5) % 60000);
            const std::string tp = dir + "/truncated_imagej.tif";
            step("truncated-ij", "damaged", "write + cut half of the last slice");
            const bool setup = writeImageJTiff(tp, (const uint8_t*)ij.data(), tx, ty, tz, 1, 16, false)
                               && truncateFile(tp, sliceBytes/2);
            step("truncated-ij", "damaged", "reads");
            void* full  = setup ? readTiffParallelWrapperNoXYFlip(tp.c_str(), {}) : nullptr;
            void* fullF = setup ? readTiffParallelWrapper(tp.c_str()) : nullptr;
            void* tail  = setup ? readTiffParallelWrapperNoXYFlip(tp.c_str(), {4, 6}) : nullptr;
            void* head  = setup ? readTiffParallelWrapperNoXYFlip(tp.c_str(), {0, 5}) : nullptr;
            const bool complete = head && std::memcmp(head, ij.data(), 5*sliceBytes) == 0;
            const bool tpass = setup && !full && !fullF && !tail && complete;
            free(full); free(fullF); free(tail); free(head);
            std::remove(tp.c_str());
            if(tpass) std::printf("PASS  %-14s damaged\n", "truncated-ij");
            else { std::printf("FAIL  %-14s damaged  (setup=%d full=%d flip=%d tail=%d complete=%d)\n", "truncated-ij",
                               (int)setup, (int)!full, (int)!fullF, (int)!tail, (int)complete); failures++; }
        }

        // A single uncompressed image cut short, read with every thread and
        // with one (which used to take a raw-read path that ignored the error).
        {
            total++;
            const uint64_t px = 64, py = 48, imgBytes = px*py*2;
            std::vector<uint16_t> img(px*py);
            for(uint64_t i = 0; i < img.size(); i++) img[i] = (uint16_t)((i*29 + 11) % 65000);
            const std::string pp = dir + "/truncated_page.tif";
            step("truncated-2d", "damaged", "write + cut half of the image");
            const bool setup = writePlainTiff(pp, (const uint8_t*)img.data(), px, py, 1, 1, 16)
                               && truncateFile(pp, imgBytes/2);
            step("truncated-2d", "damaged", "reads, all threads and one");
            void* a  = setup ? readTiffParallelWrapperNoXYFlip(pp.c_str(), {}) : nullptr;
            void* af = setup ? readTiffParallelWrapper(pp.c_str()) : nullptr;
            const int prevThreads = omp_get_max_threads();
            omp_set_num_threads(1);
            void* b  = setup ? readTiffParallelWrapperNoXYFlip(pp.c_str(), {}) : nullptr;
            void* bf = setup ? readTiffParallelWrapper(pp.c_str()) : nullptr;
            omp_set_num_threads(prevThreads);
            const bool ppass = setup && !a && !af && !b && !bf;
            free(a); free(af); free(b); free(bf);
            std::remove(pp.c_str());
            if(ppass) std::printf("PASS  %-14s damaged\n", "truncated-2d");
            else { std::printf("FAIL  %-14s damaged  (setup=%d all=%d allFlip=%d one=%d oneFlip=%d)\n", "truncated-2d",
                               (int)setup, (int)!a, (int)!af, (int)!b, (int)!bf); failures++; }
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
