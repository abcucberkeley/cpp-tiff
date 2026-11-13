// GPU reads (gpureadtiff.h) against the CPU reader (readTiffParallelInfo), byte for byte: every sample size, in
// both byte orders; uncompressed, LZW, zstd and deflate strips, with the horizontal predictor and without; strips
// of one, seven and all rows; tiles, ragged ones too; chunky and planar color; ImageJ stacks; slice ranges and
// single slices; both layouts; batches of reads of several files in one call; and a damaged deflate strip, which
// must fail as the CPU read fails. Exits 0 with a message when this machine has no NVIDIA GPU, driver or nvCOMP.
// Usage: gpuTest <scratch folder>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "gpu/cudaapi.h"
#include "gpureadtiff.h"
#include "helperfunctions.h"
#include "parallelreadtiff.h"
#include "tiffio.h"

using namespace ctcuda;

static const Api* A = nullptr;
static int total = 0, failures = 0;

// An environment variable set, or unset with value nullptr (Windows has no setenv)
static void setEnv(const char* name, const char* value){
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if(value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static void ck(CUresult r, const char* what){
    if(r){ std::fprintf(stderr, "%s: %s\n", what, errorText(A, r).c_str()); std::exit(2); }
}

// A stack written with libtiff, its data row-major (z, y, x, c) in the host's byte order
struct Spec {
    uint64_t x, y, z, spp, bits;
    uint16_t fmt, comp, predictor;
    uint32_t rps, tile;
    bool planar, bigEndian;
};

static bool writeStack(const std::string &path, const Spec &s, const std::vector<uint8_t> &data){
    TIFF* tif = TIFFOpen(path.c_str(), s.bigEndian ? "w8b" : "w8l");
    if(!tif) return false;
    const uint64_t es = s.bits/8, slice = s.x*s.y*s.spp*es;
    std::vector<uint8_t> plane;
    for(uint64_t k = 0; k < s.z; k++){
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (uint32_t)s.x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (uint32_t)s.y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (uint16_t)s.bits);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)s.spp);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, s.fmt);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, s.comp);
        if(s.predictor != PREDICTOR_NONE) TIFFSetField(tif, TIFFTAG_PREDICTOR, s.predictor);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, s.spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        if(s.spp == 4){ const uint16_t extra = EXTRASAMPLE_UNASSALPHA; TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, &extra); }
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, s.planar ? PLANARCONFIG_SEPARATE : PLANARCONFIG_CONTIG);
        const uint8_t* src = data.data() + k*slice;
        // (planar: each color's plane on its own, from the chunky data)
        const uint64_t planes = s.planar ? s.spp : 1, ppx = s.planar ? 1 : s.spp;
        for(uint64_t p = 0; p < planes; p++){
            const uint8_t* img = src;
            if(s.planar){
                plane.resize(s.x*s.y*es);
                for(uint64_t i = 0; i < s.x*s.y; i++) std::memcpy(&plane[i*es], src + (i*s.spp + p)*es, es);
                img = plane.data();
            }
            if(s.tile){
                if(p == 0){ TIFFSetField(tif, TIFFTAG_TILEWIDTH, s.tile); TIFFSetField(tif, TIFFTAG_TILELENGTH, s.tile); }
                std::vector<uint8_t> t((uint64_t)s.tile*s.tile*ppx*es, 0);
                const uint64_t across = (s.x + s.tile - 1)/s.tile, down = (s.y + s.tile - 1)/s.tile;
                for(uint64_t ty = 0; ty < down; ty++)
                    for(uint64_t tx = 0; tx < across; tx++){
                        std::fill(t.begin(), t.end(), 0);
                        for(uint64_t r = 0; r < s.tile && ty*s.tile + r < s.y; r++){
                            const uint64_t cols = std::min<uint64_t>(s.tile, s.x - tx*s.tile);
                            std::memcpy(&t[r*s.tile*ppx*es], img + ((ty*s.tile + r)*s.x + tx*s.tile)*ppx*es, cols*ppx*es);
                        }
                        if(TIFFWriteEncodedTile(tif, (uint32_t)(p*across*down + ty*across + tx), t.data(), (tmsize_t)t.size()) < 0){
                            TIFFClose(tif);
                            return false;
                        }
                    }
            }
            else{
                if(p == 0) TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, s.rps);
                const uint64_t strips = (s.y + s.rps - 1)/s.rps;
                for(uint64_t i = 0; i < strips; i++){
                    const uint64_t rows = std::min<uint64_t>(s.rps, s.y - i*s.rps);
                    if(TIFFWriteEncodedStrip(tif, (uint32_t)(p*strips + i), (void*)(img + i*s.rps*s.x*ppx*es),
                                             (tmsize_t)(rows*s.x*ppx*es)) < 0){
                        TIFFClose(tif);
                        return false;
                    }
                }
            }
        }
        if(!TIFFWriteDirectory(tif)){ TIFFClose(tif); return false; }
    }
    TIFFClose(tif);
    return true;
}

// An ImageJ stack: one directory, the slices back to back after it (the layout of tests/roundtrip.cpp's
// writeImageJTiff)
static void put16(std::vector<uint8_t> &v, uint16_t s, bool be){
    if(be){ v.push_back((uint8_t)(s >> 8)); v.push_back((uint8_t)s); } else { v.push_back((uint8_t)s); v.push_back((uint8_t)(s >> 8)); }
}
static void put32(std::vector<uint8_t> &v, uint32_t l, bool be){
    if(be) for(int i = 3; i >= 0; i--) v.push_back((uint8_t)(l >> (8*i)));
    else for(int i = 0; i < 4; i++) v.push_back((uint8_t)(l >> (8*i)));
}
static bool writeImageJ(const std::string &path, const std::vector<uint8_t> &data, uint64_t x, uint64_t y, uint64_t z,
                        uint64_t spp, uint64_t bits, bool be){
    const uint64_t es = bits/8;
    std::string desc = "ImageJ=1.11a\nimages=" + std::to_string(z) + "\nslices=" + std::to_string(z) + "\n";
    desc.push_back('\0');
    if(desc.size() % 2) desc.push_back('\0');
    const uint32_t nEntries = 11, auxOff = 8 + 2 + nEntries*12 + 4, bpsLen = spp > 1 ? (uint32_t)(spp*2) : 0;
    const uint32_t descOff = auxOff + bpsLen;
    uint32_t pixOff = descOff + (uint32_t)desc.size();
    pixOff += pixOff % 2;
    std::vector<uint8_t> v;
    v.push_back(be ? 'M' : 'I'); v.push_back(be ? 'M' : 'I');
    put16(v, 42, be); put32(v, 8, be); put16(v, (uint16_t)nEntries, be);
    auto entry = [&](uint16_t tag, uint16_t typ, uint32_t cnt){ put16(v, tag, be); put16(v, typ, be); put32(v, cnt, be); };
    auto shortVal = [&](uint16_t s){ put16(v, s, be); v.push_back(0); v.push_back(0); };
    entry(256, 3, 1); shortVal((uint16_t)x);
    entry(257, 3, 1); shortVal((uint16_t)y);
    entry(258, 3, (uint32_t)spp); if(spp == 1) shortVal((uint16_t)bits); else put32(v, auxOff, be);
    entry(259, 3, 1); shortVal(1);
    entry(262, 3, 1); shortVal(spp == 3 ? 2 : 1);
    entry(270, 2, (uint32_t)desc.size()); put32(v, descOff, be);
    entry(273, 4, 1); put32(v, pixOff, be);
    entry(277, 3, 1); shortVal((uint16_t)spp);
    entry(278, 3, 1); shortVal((uint16_t)y);
    entry(279, 4, 1); put32(v, (uint32_t)(x*y*spp*es), be);
    entry(284, 3, 1); shortVal(1);
    put32(v, 0, be);
    if(spp > 1) for(uint64_t i = 0; i < spp; i++) put16(v, (uint16_t)bits, be);
    v.insert(v.end(), desc.begin(), desc.end());
    while(v.size() < pixOff) v.push_back(0);
    for(uint64_t i = 0; i < x*y*z*spp; i++)
        for(uint64_t b = 0; b < es; b++) v.push_back(data[i*es + (be ? es - 1 - b : b)]);
    FILE* fp = std::fopen(path.c_str(), "wb");
    if(!fp) return false;
    const bool ok = std::fwrite(v.data(), 1, v.size(), fp) == v.size();
    return std::fclose(fp) == 0 && ok;
}

// The reads read on the CPU (the reference) and on the GPU in one call, compared; prints a FAIL line. expectFail:
// the CPU read fails, and the GPU read must too.
static bool compare(const std::vector<std::string> &paths, const std::vector<std::vector<uint64_t>> &ranges, bool flip,
                    const std::string &label, bool expectFail = false){
    total++;
    std::vector<std::vector<uint8_t>> cpu(paths.size());
    std::vector<uint64_t> bytes(paths.size());
    bool cpuFailed = false;
    for(size_t i = 0; i < paths.size(); i++){
        TiffInfo info;
        if(getTiffInfo(paths[i].c_str(), info)){ cpuFailed = true; continue; }
        uint64_t z = info.z;
        if(ranges[i].size()) z = ranges[i].size() == 2 ? ranges[i][1] - ranges[i][0] : 1;
        bytes[i] = info.x*info.y*info.samples*(info.bits/8)*z;
        cpu[i].resize(bytes[i]);
        if(!readTiffParallelInfo(paths[i].c_str(), info, cpu[i].data(), flip ? 1 : 0, ranges[i])) cpuFailed = true;
    }
    if(cpuFailed != expectFail){
        std::printf("FAIL  %s: the CPU read %s\n", label.c_str(), cpuFailed ? "failed" : "did not fail");
        failures++;
        return false;
    }
    std::vector<CUdeviceptr> d(paths.size(), 0);
    std::vector<GpuTiffRead> reads;
    for(size_t i = 0; i < paths.size(); i++){
        ck(A->cuMemAlloc(&d[i], std::max<uint64_t>(bytes[i], 1)), "cuMemAlloc");
        ck(A->cuMemsetD8Async(d[i], 0xab, std::max<uint64_t>(bytes[i], 1), nullptr), "cuMemsetD8Async");
        reads.push_back(GpuTiffRead{paths[i], ranges[i], (void*)(uintptr_t)d[i]});
    }
    std::string err;
    const uint8_t r = gpuReadTiff(reads, flip, nullptr, 4, err);
    bool ok = expectFail ? r != 0 : r == 0;
    if(!ok) std::printf("FAIL  %s: the GPU read %s%s\n", label.c_str(), r ? "failed: " : "did not fail", err.c_str());
    for(size_t i = 0; ok && !expectFail && i < paths.size(); i++){
        std::vector<uint8_t> gpu(bytes[i]);
        ck(A->cuMemcpyDtoHAsync(gpu.data(), d[i], bytes[i], nullptr), "cuMemcpyDtoHAsync");
        ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
        if(std::memcmp(gpu.data(), cpu[i].data(), bytes[i])){
            uint64_t at = 0;
            while(at < bytes[i] && gpu[at] == cpu[i][at]) at++;
            std::printf("FAIL  %s: read %zu differs from the CPU's at byte %llu of %llu\n", label.c_str(), i,
                        (unsigned long long)at, (unsigned long long)bytes[i]);
            ok = false;
        }
    }
    for(CUdeviceptr p : d) A->cuMemFree(p);
    if(!ok){ failures++; return false; }
    return true;
}

int main(int argc, char** argv){
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::string why;
    if(!gpuReadTiffAvailable(why)){
        std::printf("GPU tests skipped: %s\n", why.c_str());
        return 0;
    }
    A = api(&why);
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    ck(A->cuDeviceGet(&dev, 0), "cuDeviceGet");
    ck(A->cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    std::printf("GPU reads with %s (nvCOMP %d)\n", A->nvcompPath.c_str(), A->nvcompMajor);
    std::mt19937 rng(91);
    // (smooth data, so that the predictor and the codecs have something to do, with noise in the low bits)
    auto make = [&](const Spec &s){
        const uint64_t es = s.bits/8, n = s.x*s.y*s.z*s.spp;
        std::vector<uint8_t> v(n*es);
        for(uint64_t i = 0; i < n; i++){
            const uint64_t val = (i/s.spp % s.x)*3 + (i/(s.spp*s.x) % s.y)*5 + (rng() & 7);
            for(uint64_t b = 0; b < es; b++) v[i*es + b] = (uint8_t)(val >> (8*b));
        }
        return v;
    };
    int fileNo = 0;
    auto fileFor = [&](const Spec &s){
        const std::string path = dir + "/gpu_" + std::to_string(fileNo++) + ".tif";
        if(!writeStack(path, s, make(s))){ std::printf("FAIL  could not write %s\n", path.c_str()); failures++; return std::string(); }
        return path;
    };

    // Codecs, predictors, sample sizes, byte orders, strips and tiles, color, both layouts, slice ranges
    const uint16_t codecs[] = {COMPRESSION_NONE, COMPRESSION_LZW, COMPRESSION_ZSTD, COMPRESSION_ADOBE_DEFLATE};
    const char* codecNames[] = {"none", "lzw", "zstd", "deflate"};
    struct Layout { const char* name; uint64_t x, y, z, spp; uint32_t rps, tile; bool planar; };
    const Layout layouts[] = {
        {"strips7", 50, 37, 5, 1, 7, 0, false},
        {"strip1", 33, 20, 3, 1, 1, 0, false},
        {"whole", 64, 48, 4, 1, 48, 0, false},
        {"tiles", 70, 45, 3, 1, 0, 32, false},
        {"rgb", 40, 30, 3, 3, 8, 0, false},
        {"rgba-tiles", 40, 30, 2, 4, 0, 16, false},
        {"planar", 40, 30, 3, 3, 10, 0, true},
    };
    for(int ci = 0; ci < 4; ci++)
        for(const Layout &l : layouts)
            for(uint64_t bits : {8, 16, 32, 64})
                for(bool be : {false, true})
                    for(uint16_t pred : {(uint16_t)PREDICTOR_NONE, (uint16_t)PREDICTOR_HORIZONTAL}){
                        if(codecs[ci] == COMPRESSION_NONE && pred != PREDICTOR_NONE) continue;
                        if(bits == 8 && be) continue;
                        const Spec s{l.x, l.y, l.z, l.spp, bits, (uint16_t)(bits >= 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT),
                                     codecs[ci], pred, l.rps, l.tile, l.planar, be};
                        const std::string path = fileFor(s);
                        if(path.empty()) continue;
                        const std::string label = std::string(codecNames[ci]) + " " + l.name + " " + std::to_string(bits) + "-bit" +
                                                  (be ? " BE" : "") + (pred == PREDICTOR_HORIZONTAL ? " predictor" : "");
                        for(bool flip : {false, true}){
                            compare({path}, {{}}, flip, label + (flip ? " flip" : ""));
                            compare({path}, {{1, l.z}}, flip, label + " [1,z)" + (flip ? " flip" : ""));
                            compare({path}, {{l.z - 1}}, flip, label + " last" + (flip ? " flip" : ""));
                        }
                        std::remove(path.c_str());
                    }

    // ImageJ stacks, both byte orders, gray and RGB, a slice range
    for(bool be : {false, true})
        for(uint64_t spp : {1, 3})
            for(uint64_t bits : {8, 16, 32}){
                const Spec s{48, 32, 6, spp, bits, 1, COMPRESSION_NONE, PREDICTOR_NONE, 32, 0, false, be};
                const std::string path = dir + "/gpu_ij_" + std::to_string(fileNo++) + ".tif";
                if(!writeImageJ(path, make(s), s.x, s.y, s.z, spp, bits, be)){ failures++; continue; }
                const std::string label = std::string("imagej ") + std::to_string(bits) + "-bit spp " + std::to_string(spp) + (be ? " BE" : "");
                for(bool flip : {false, true}){
                    compare({path}, {{}}, flip, label + (flip ? " flip" : ""));
                    compare({path}, {{2, 5}}, flip, label + " [2,5)" + (flip ? " flip" : ""));
                }
                std::remove(path.c_str());
            }

    // A batch: files of different codecs and types read in one call, with ranges
    {
        const Spec a{60, 40, 7, 1, 16, 1, COMPRESSION_ZSTD, PREDICTOR_HORIZONTAL, 9, 0, false, false};
        const Spec b{30, 50, 4, 3, 8, 1, COMPRESSION_LZW, PREDICTOR_NONE, 5, 0, false, false};
        const Spec c{45, 45, 5, 1, 32, 3, COMPRESSION_ADOBE_DEFLATE, PREDICTOR_NONE, 0, 16, false, true};
        const std::string pa = fileFor(a), pb = fileFor(b), pc = fileFor(c);
        for(bool flip : {false, true})
            compare({pa, pb, pc, pa}, {{}, {1, 3}, {}, {6}}, flip, std::string("batch") + (flip ? " flip" : ""));
        std::remove(pa.c_str()); std::remove(pb.c_str()); std::remove(pc.c_str());
    }

    // A big stack in several parts (CPPTIFF_GPU_PART_MAX_MB=1 below makes each part 1 MB)
    {
        const Spec s{512, 512, 12, 1, 16, 1, COMPRESSION_ZSTD, PREDICTOR_NONE, 64, 0, false, false};
        const std::string path = fileFor(s);
        setEnv("CPPTIFF_GPU_PART_MAX_MB", "1");
        for(bool flip : {false, true}) compare({path}, {{}}, flip, std::string("parts") + (flip ? " flip" : ""));
        setEnv("CPPTIFF_GPU_PART_MAX_MB", nullptr);
        std::remove(path.c_str());
    }

    // A damaged deflate strip (its zlib header intact): the CPU read fails, and so must the GPU read
    {
        const Spec s{64, 64, 3, 1, 16, 1, COMPRESSION_ADOBE_DEFLATE, PREDICTOR_NONE, 16, 0, false, false};
        const std::string path = fileFor(s);
        TIFF* tif = TIFFOpen(path.c_str(), "r");
        bool made = false;
        if(tif){
            TIFFSetDirectory(tif, 1);
            const uint64_t at = TIFFGetStrileOffset(tif, 2), n = TIFFGetStrileByteCount(tif, 2);
            TIFFClose(tif);
            FILE* fp = std::fopen(path.c_str(), "r+b");
            if(fp && n > 12){
                std::vector<uint8_t> junk(n - 8, 0x5a);
                made = std::fseek(fp, (long)(at + 4), SEEK_SET) == 0 && std::fwrite(junk.data(), 1, junk.size(), fp) == junk.size();
            }
            if(fp) std::fclose(fp);
        }
        if(made) compare({path}, {{}}, false, "damaged deflate strip", true);
        else { std::printf("FAIL  could not damage the deflate file\n"); failures++; }
        std::remove(path.c_str());
    }

    if(failures){
        std::printf("%d of %d GPU reads FAILED\n", failures, total);
        return 1;
    }
    std::printf("%d/%d GPU reads match the CPU's\n", total, total);
    return 0;
}
