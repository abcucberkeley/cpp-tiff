// GPU writes (gpuwritetiff.h) against the CPU writer (writeTiffParallelHelper): uncompressed and LZW files must be
// the CPU writer's byte for byte, and zstd ones (compressed by nvCOMP) must read back as the data, on the CPU and on
// the GPU, and be the CPU writer's byte for byte when compressed on the host; every sample size, gray, RGB and
// RGBA, both layouts, images of one row, one column and one pixel, slices added to a file (mode "a"), writes in
// many parts, after work queued on a stream, from two threads at once, tiled writes (copied to the host), and
// errors. Exits 0 with a message when this machine has no NVIDIA GPU, driver or nvCOMP.
// Usage: gpuWriteTest <scratch folder>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "gpu/cudaapi.h"
#include "gpureadtiff.h"
#include "gpuwritetiff.h"
#include "helperfunctions.h"
#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "tiffio.h"

using namespace ctcuda;

static const Api* A = nullptr;
static int total = 0, failures = 0;

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

static std::vector<uint8_t> fileBytes(const std::string &path){
    std::vector<uint8_t> v;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if(!fp) return v;
    std::fseek(fp, 0, SEEK_END);
    const long n = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if(n > 0){
        v.resize((size_t)n);
        if(std::fread(v.data(), 1, v.size(), fp) != v.size()) v.clear();
    }
    std::fclose(fp);
    return v;
}

static bool fail(const std::string &label, const std::string &what){
    std::printf("FAIL  %s: %s\n", label.c_str(), what.c_str());
    failures++;
    return false;
}

struct Case {
    uint64_t x, y, z, spp, bits;
    uint16_t fmt;
};

static std::mt19937 rng(17);

// Row-major (z, y, x, c) data, smooth with noise in the low bits; and the same in MATLAB's (y, x, c, z)
static std::vector<uint8_t> make(const Case &c){
    const uint64_t es = c.bits/8, n = c.x*c.y*c.z*c.spp;
    std::vector<uint8_t> v(n*es);
    for(uint64_t i = 0; i < n; i++){
        const uint64_t val = (i/c.spp % c.x)*3 + (i/(c.spp*c.x) % c.y)*5 + (rng() & 15) + (i % c.spp)*40;
        for(uint64_t b = 0; b < es; b++) v[i*es + b] = (uint8_t)(val >> (8*b));
    }
    return v;
}
static std::vector<uint8_t> toF(const Case &c, const std::vector<uint8_t> &v){
    const uint64_t es = c.bits/8;
    std::vector<uint8_t> f(v.size());
    for(uint64_t s = 0; s < c.z; s++)
        for(uint64_t r = 0; r < c.y; r++)
            for(uint64_t j = 0; j < c.x; j++)
                for(uint64_t k = 0; k < c.spp; k++)
                    std::memcpy(&f[(r + c.y*(j + c.x*(k + c.spp*s)))*es], &v[(((s*c.y + r)*c.x + j)*c.spp + k)*es], es);
    return f;
}

static CUdeviceptr toGpu(const std::vector<uint8_t> &v){
    CUdeviceptr d = 0;
    ck(A->cuMemAlloc(&d, std::max<size_t>(v.size(), 1)), "cuMemAlloc");
    ck(A->cuMemcpyHtoDAsync(d, v.data(), v.size(), nullptr), "cuMemcpyHtoDAsync");
    ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
    return d;
}

// After the device's primary context is reset (as MATLAB's reset(gpuDevice) and cudaDeviceReset reset it): the context
// made current again, and GPU memory filled with 0x5a, which takes the addresses the freed buffers had (bufs, freed by
// unchanged()). unchanged: whether every byte of it is still 0x5a.
static std::vector<CUdeviceptr> resetAndFill(CUdevice dev, CUcontext &ctx, uint64_t each, int n){
    ck(A->cuDevicePrimaryCtxReset(dev), "cuDevicePrimaryCtxReset");
    ck(A->cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    std::vector<CUdeviceptr> bufs;
    for(int i = 0; i < n; i++){
        CUdeviceptr p = 0;
        if(A->cuMemAlloc(&p, each)) break;
        ck(A->cuMemsetD8Async(p, 0x5a, each, nullptr), "cuMemsetD8Async");
        bufs.push_back(p);
    }
    ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
    return bufs;
}
static bool unchanged(std::vector<CUdeviceptr> &bufs, uint64_t each){
    bool same = true;
    std::vector<uint8_t> h(each);
    for(CUdeviceptr p : bufs){
        ck(A->cuMemcpyDtoHAsync(h.data(), p, each, nullptr), "cuMemcpyDtoHAsync");
        ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
        for(uint8_t b : h) if(b != 0x5a){ same = false; break; }
        A->cuMemFree(p);
    }
    bufs.clear();
    return same;
}

// The file read back on the CPU (row-major) must be data
static bool readsBack(const std::string &path, const Case &c, const std::vector<uint8_t> &data, const std::string &label){
    TiffInfo info;
    if(getTiffInfo(path.c_str(), info)) return fail(label, "the file cannot be read");
    if(info.x != c.x || info.y != c.y || info.z != c.z || info.bits != c.bits || info.samples != c.spp)
        return fail(label, "the file's dimensions are "+std::to_string(info.x)+" x "+std::to_string(info.y)+" x "+
                    std::to_string(info.z)+", "+std::to_string(info.bits)+"-bit, "+std::to_string(info.samples)+" samples");
    std::vector<uint8_t> back(data.size());
    if(!readTiffParallelInfo(path.c_str(), info, back.data(), 0, {})) return fail(label, "the CPU read failed");
    if(back != data){
        uint64_t at = 0;
        while(at < data.size() && back[at] == data[at]) at++;
        return fail(label, "the file reads back differently at byte "+std::to_string(at)+" of "+std::to_string(data.size()));
    }
    return true;
}

// The file read back on the GPU (row-major) must be data
static bool gpuReadsBack(const std::string &path, const std::vector<uint8_t> &data, const std::string &label){
    CUdeviceptr d = 0;
    ck(A->cuMemAlloc(&d, data.size()), "cuMemAlloc");
    std::string err;
    std::vector<uint8_t> back(data.size());
    const uint8_t r = gpuReadTiff({GpuTiffRead{path, {}, (void*)(uintptr_t)d}}, false, nullptr, 4, err);
    if(!r){
        ck(A->cuMemcpyDtoHAsync(back.data(), d, data.size(), nullptr), "cuMemcpyDtoHAsync");
        ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
    }
    A->cuMemFree(d);
    if(r) return fail(label, "the GPU read failed: "+err);
    if(back != data) return fail(label, "the file reads back differently on the GPU");
    return true;
}

// A GPU write of the case against the CPU writer's file of it
static bool check(const Case &c, const std::string &comp, bool flip, const std::string &dir, const std::string &label,
                  bool sameBytes){
    total++;
    const std::vector<uint8_t> data = make(c);
    const std::vector<uint8_t> in = flip ? toF(c, data) : data;
    const std::string pg = dir + "/gw_gpu.tif", pc = dir + "/gw_cpu.tif";
    const CUdeviceptr d = toGpu(in);
    std::string err;
    const uint8_t r = gpuWriteTiff(pg.c_str(), (void*)(uintptr_t)d, c.x, c.y, c.z, c.bits, c.spp, c.fmt, comp, "w", flip, 0,
                                   nullptr, 4, err);
    A->cuMemFree(d);
    if(r) return fail(label, "the GPU write failed: "+err);
    if(writeTiffParallelHelper(pc.c_str(), in.data(), c.bits, "w", c.x, c.y, c.z, 0, flip, comp, c.fmt, 0, c.spp))
        return fail(label, "the CPU write failed");
    if(sameBytes && fileBytes(pg) != fileBytes(pc)) return fail(label, "the file is not the CPU writer's");
    if(!readsBack(pg, c, data, label)) return false;
    if(comp == "zstd" && !gpuReadsBack(pg, data, label)) return false;
    return true;
}

int main(int argc, char** argv){
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::string why;
    if(!gpuWriteTiffAvailable(why)){
        std::printf("GPU write tests skipped: %s\n", why.c_str());
        return 0;
    }
    A = api(&why);
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    ck(A->cuDeviceGet(&dev, 0), "cuDeviceGet");
    ck(A->cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    std::printf("GPU writes with %s (nvCOMP %d)\n", A->nvcompPath.c_str(), A->nvcompMajor);
    // (every zstd write here must be compressed on the GPU, unless a case says otherwise)
    setEnv("CPPTIFF_GPU_STRICT", "1");

    const Case shapes[] = {
        {50, 37, 5, 1, 0, 0}, {33, 20, 3, 1, 0, 0}, {1, 1, 1, 1, 0, 0}, {1, 77, 2, 1, 0, 0}, {77, 1, 3, 1, 0, 0},
        {300, 211, 4, 1, 0, 0}, {40, 30, 3, 3, 0, 0}, {41, 29, 2, 4, 0, 0}, {1, 9, 2, 3, 0, 0},
    };
    for(const char* comp : {"none", "lzw", "zstd"})
        for(Case c : shapes)
            for(uint64_t bits : {8, 16, 32, 64})
                for(bool flip : {false, true}){
                    c.bits = bits;
                    c.fmt = bits >= 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT;
                    const std::string label = std::string(comp)+" "+std::to_string(c.x)+"x"+std::to_string(c.y)+"x"+
                                              std::to_string(c.z)+" spp "+std::to_string(c.spp)+" "+std::to_string(bits)+"-bit"+
                                              (flip ? " flip" : "");
                    check(c, comp, flip, dir, label, std::string(comp) != "zstd");
                }
    // Signed integers
    for(const char* comp : {"none", "zstd"}) check({64, 40, 3, 1, 16, SAMPLEFORMAT_INT}, comp, false, dir, std::string(comp)+" int16", std::string(comp) != "zstd");

    // zstd compressed on the host (CPPTIFF_GPU_WRITE_HOST=1): the CPU writer's file
    setEnv("CPPTIFF_GPU_WRITE_HOST", "1");
    for(bool flip : {false, true}){
        check({300, 211, 4, 1, 16, SAMPLEFORMAT_UINT}, "zstd", flip, dir, std::string("zstd on the host")+(flip ? " flip" : ""), true);
        check({40, 30, 3, 3, 8, SAMPLEFORMAT_UINT}, "zstd", flip, dir, std::string("zstd on the host rgb")+(flip ? " flip" : ""), true);
    }
    setEnv("CPPTIFF_GPU_WRITE_HOST", nullptr);

    // Rows longer than nvCOMP compresses in one frame (19 MB): compressed on the host, as the CPU writer does
    setEnv("CPPTIFF_GPU_STRICT", nullptr);
    check({600000, 2, 1, 4, 64, SAMPLEFORMAT_IEEEFP}, "zstd", false, dir, "zstd rows of 19 MB", true);
    setEnv("CPPTIFF_GPU_STRICT", "1");

    // Many parts (CPPTIFF_GPU_WRITE_PART_MB=1: 1 MB each), with the profile printed
    setEnv("CPPTIFF_GPU_WRITE_PART_MB", "1");
    setEnv("CPPTIFF_GPU_PROFILE", "1");
    for(const char* comp : {"none", "lzw", "zstd"})
        for(bool flip : {false, true})
            check({512, 512, 12, 1, 16, SAMPLEFORMAT_UINT}, comp, flip, dir, std::string("parts ")+comp+(flip ? " flip" : ""),
                  std::string(comp) != "zstd");
    setEnv("CPPTIFF_GPU_PROFILE", nullptr);
    setEnv("CPPTIFF_GPU_WRITE_PART_MB", nullptr);

    // Slices added to a file (mode "a"): the CPU writer's file made the same way
    for(const char* comp : {"none", "lzw", "zstd"})
        for(bool flip : {false, true}){
            total++;
            const std::string label = std::string("append ")+comp+(flip ? " flip" : "");
            const Case c{70, 50, 6, 1, 16, SAMPLEFORMAT_UINT}, h{70, 50, 3, 1, 16, SAMPLEFORMAT_UINT};
            const std::vector<uint8_t> data = make(c);
            const uint64_t half = data.size()/2;
            const std::vector<uint8_t> a(data.begin(), data.begin() + half), b(data.begin() + half, data.end());
            const std::vector<uint8_t> ia = flip ? toF(h, a) : a, ib = flip ? toF(h, b) : b;
            const std::string pg = dir + "/gw_gpu.tif", pc = dir + "/gw_cpu.tif";
            std::string err;
            bool ok = true;
            for(int part = 0; part < 2 && ok; part++){
                const CUdeviceptr d = toGpu(part ? ib : ia);
                ok = !gpuWriteTiff(pg.c_str(), (void*)(uintptr_t)d, 70, 50, 3, 16, 1, SAMPLEFORMAT_UINT, comp, part ? "a" : "w", flip,
                                   0, nullptr, 4, err);
                A->cuMemFree(d);
                ok = ok && !writeTiffParallelHelper(pc.c_str(), (part ? ib : ia).data(), 16, part ? "a" : "w", 70, 50, 3, 0, flip,
                                                     comp, SAMPLEFORMAT_UINT, 0, 1);
            }
            if(!ok){ fail(label, "a write failed: "+err); continue; }
            if(std::string(comp) != "zstd" && fileBytes(pg) != fileBytes(pc)){ fail(label, "the file is not the CPU writer's"); continue; }
            readsBack(pg, c, data, label);
        }

    // After work queued on a stream of the caller's: the data the stream's memset leaves is written
    {
        total++;
        CUstream st = nullptr;
        ck(A->cuStreamCreate(&st, 1), "cuStreamCreate");
        const uint64_t n = 1024*1024*16;
        CUdeviceptr d = 0;
        ck(A->cuMemAlloc(&d, n*2), "cuMemAlloc");
        ck(A->cuMemsetD8Async(d, 0x11, n*2, st), "cuMemsetD8Async");
        ck(A->cuStreamSynchronize(st), "cuStreamSynchronize");
        ck(A->cuMemsetD8Async(d, 0x5a, n*2, st), "cuMemsetD8Async");
        std::string err;
        const std::string pg = dir + "/gw_gpu.tif";
        const uint8_t r = gpuWriteTiff(pg.c_str(), (void*)(uintptr_t)d, 4096, 4096, 1, 16, 1, SAMPLEFORMAT_UINT, "zstd", "w", false, 0,
                                       (void*)st, 4, err);
        A->cuMemFree(d);
        if(r) fail("stream", "the GPU write failed: "+err);
        else{
            std::vector<uint8_t> want(n*2, 0x5a);
            readsBack(pg, {4096, 4096, 1, 1, 16, SAMPLEFORMAT_UINT}, want, "stream");
        }
    }

    // Two writes at once, from two threads
    {
        total++;
        const Case c{256, 256, 8, 1, 16, SAMPLEFORMAT_UINT};
        const std::vector<uint8_t> d1 = make(c), d2 = make(c);
        const CUdeviceptr g1 = toGpu(d1), g2 = toGpu(d2);
        std::string e1, e2;
        uint8_t r1 = 1, r2 = 1;
        const std::string p1 = dir + "/gw_t1.tif", p2 = dir + "/gw_t2.tif";
        std::thread t1([&]{
            ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
            r1 = gpuWriteTiff(p1.c_str(), (void*)(uintptr_t)g1, 256, 256, 8, 16, 1, SAMPLEFORMAT_UINT, "zstd", "w", false, 0, nullptr, 2, e1);
        });
        std::thread t2([&]{
            ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
            r2 = gpuWriteTiff(p2.c_str(), (void*)(uintptr_t)g2, 256, 256, 8, 16, 1, SAMPLEFORMAT_UINT, "lzw", "w", false, 0, nullptr, 2, e2);
        });
        t1.join();
        t2.join();
        A->cuMemFree(g1);
        A->cuMemFree(g2);
        if(r1 || r2) fail("two threads", "a write failed: "+e1+" "+e2);
        else if(readsBack(p1, c, d1, "two threads (1)")) readsBack(p2, c, d2, "two threads (2)");
        std::remove(p1.c_str());
        std::remove(p2.c_str());
    }

    // A tiled write: copied to the host and written by the CPU writer, the CPU writer's file (refused in strict mode)
    for(bool flip : {false, true}){
        total++;
        const std::string label = std::string("tiled")+(flip ? " flip" : "");
        const Case c{70, 45, 3, 1, 16, SAMPLEFORMAT_UINT};
        const std::vector<uint8_t> data = make(c), in = flip ? toF(c, data) : data;
        const CUdeviceptr d = toGpu(in);
        const std::string pg = dir + "/gw_gpu.tif", pc = dir + "/gw_cpu.tif";
        std::string err;
        const uint8_t refused = gpuWriteTiff(pg.c_str(), (void*)(uintptr_t)d, 70, 45, 3, 16, 1, SAMPLEFORMAT_UINT, "lzw", "w", flip, 32,
                                             nullptr, 4, err);
        setEnv("CPPTIFF_GPU_STRICT", nullptr);
        const uint8_t r = gpuWriteTiff(pg.c_str(), (void*)(uintptr_t)d, 70, 45, 3, 16, 1, SAMPLEFORMAT_UINT, "lzw", "w", flip, 32,
                                       nullptr, 4, err);
        setEnv("CPPTIFF_GPU_STRICT", "1");
        A->cuMemFree(d);
        if(!refused){ fail(label, "strict mode did not refuse it"); continue; }
        if(r){ fail(label, "the GPU write failed: "+err); continue; }
        if(writeTiffParallelHelper(pc.c_str(), in.data(), 16, "w", 70, 45, 3, 0, flip, "lzw", SAMPLEFORMAT_UINT, 32, 1)){
            fail(label, "the CPU write failed");
            continue;
        }
        if(fileBytes(pg) != fileBytes(pc)){ fail(label, "the file is not the CPU writer's"); continue; }
        readsBack(pg, c, data, label);
    }

    // The buffers kept between writes freed (gpuReleaseBuffers), and the context reset (as MATLAB's reset(gpuDevice)
    // resets it): the writes after make new buffers, and write nothing into memory that is no longer theirs
    {
        const Case z{512, 512, 8, 1, 16, SAMPLEFORMAT_UINT}, w{300, 200, 4, 3, 8, SAMPLEFORMAT_UINT};
        auto both = [&](const std::string &label){
            for(bool flip : {false, true}){
                check(z, "zstd", flip, dir, label + " zstd" + (flip ? " flip" : ""), false);
                check(w, "lzw", flip, dir, label + " lzw rgb" + (flip ? " flip" : ""), true);
            }
        };
        both("before freeing the buffers:");
        gpuReleaseBuffers();
        gpuReleaseBuffers();
        both("after freeing the buffers:");
        if(A->cuDevicePrimaryCtxReset){
            const uint64_t each = (uint64_t)64 << 20;
            std::vector<CUdeviceptr> fill = resetAndFill(dev, ctx, each, 16);
            both("after a reset:");
            total++;
            if(!unchanged(fill, each)){ std::printf("FAIL  after a reset: a write wrote into memory allocated after it\n"); failures++; }
        }
        else std::printf("(no cuDevicePrimaryCtxReset: the reset case is left out)\n");
    }

    // Errors: host memory, an unknown compression or mode, a file that cannot be made
    {
        std::vector<uint8_t> host(64*64*2, 1);
        const CUdeviceptr d = toGpu(host);
        const std::string pg = dir + "/gw_gpu.tif";
        struct E { const char* label; const void* data; const char* comp; const char* mode; std::string path; };
        const E es[] = {
            {"host memory", host.data(), "lzw", "w", pg},
            {"unknown compression", (void*)(uintptr_t)d, "jpeg", "w", pg},
            {"unknown mode", (void*)(uintptr_t)d, "lzw", "r+", pg},
            {"a file that cannot be made", (void*)(uintptr_t)d, "zstd", "w", "/proc/cpptiff_no/such.tif"},
        };
        for(const E &e : es){
            total++;
            std::string err;
            const uint8_t r = gpuWriteTiff(e.path.c_str(), e.data, 64, 64, 1, 16, 1, SAMPLEFORMAT_UINT, e.comp, e.mode, false, 0, nullptr, 4, err);
            if(!r) fail(e.label, "the write did not fail");
            else if(err.empty()) fail(e.label, "the write failed without a reason");
            else std::printf("ok    %s: %s\n", e.label, err.c_str());
        }
        A->cuMemFree(d);
    }

    std::remove((dir + "/gw_gpu.tif").c_str());
    std::remove((dir + "/gw_cpu.tif").c_str());
    if(failures){
        std::printf("%d of %d GPU writes FAILED\n", failures, total);
        return 1;
    }
    std::printf("%d/%d GPU writes are right\n", total, total);
    return 0;
}
