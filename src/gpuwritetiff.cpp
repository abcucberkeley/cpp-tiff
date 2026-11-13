#include "gpuwritetiff.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include <omp.h>
#include <zstd.h>
#include "tiffio.h"
#include "lzwencode.h"
#include "parallelwritetiff.h"
#include "preparedwrite.h"
#include "gpu/cudaapi.h"
#include "gpu/gpucommon.h"

using namespace ctcuda;

namespace {

// How a write's strips get to the file: copied as they are, compressed by nvCOMP, or copied and compressed on the host
enum How { AsTheyAre, OnGpu, OnHost };

// The biggest strips nvCOMP compresses in one zstd frame (16 MiB, as cpp-zarr finds)
const uint64_t zstdMaxStream = (uint64_t)1 << 24;
// The CPU writer's zstd level, for strips compressed on the host
const int hostZstdLevel = 1;

double secondsSince(std::chrono::steady_clock::time_point t0){
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// GPU memory of at least need bytes in p (have of them), allocated as asked (growDevice adds a quarter): a write's
// shared buffers are sized once for its biggest part
void ensureDevice(const Api* a, CUdeviceptr &p, size_t &have, const size_t need){
    if(need <= have) return;
    if(p) a->cuMemFree(p);
    p = 0;
    have = 0;
    const size_t n = (size_t)alignUp(std::max<size_t>(need, 1), (uint64_t)2 << 20);
    ck(a, a->cuMemAlloc(&p, n), "Could not allocate GPU memory (cuMemAlloc)");
    have = n;
}

bool envOn(const char* name){
    const char* e = std::getenv(name);
    return e && *e && std::atoi(e) == 1;
}

// A write's strips: strip i is strip i % perSlice of slice i / perSlice
struct Strips {
    uint64_t x = 0, y = 0, spp = 0, es = 0, rows = 0, perSlice = 0, rowBytes = 0, sliceBytes = 0;
    uint64_t rowsOf(uint64_t i) const { return std::min(rows, y - (i % perSlice)*rows); }
    uint64_t bytes(uint64_t i) const { return rowsOf(i)*rowBytes; }
    // where strip i starts in row-major data (the strips follow each other)
    uint64_t offset(uint64_t i) const { return (i / perSlice)*sliceBytes + (i % perSlice)*rows*rowBytes; }
};

// A run of strips that goes to the GPU and back at once
struct Part {
    uint64_t begin = 0, end = 0, bytes = 0;
};

// Each part's tables in its slot's work buffer, in 8-byte words: those the host fills (copied to the GPU at
// once), then those the GPU fills (copied back at once)
struct Tables {
    uint64_t n = 0;
    uint64_t gather = 0, src = 0, inOff = 0, inPtr = 0, inLen = 0, outPtr = 0, hostWords = 0;
    uint64_t outLen = 0, off = 0, total = 0, status = 0, words = 0;
    Tables(uint64_t strips, bool gathers){
        n = strips;
        uint64_t w = 0;
        gather = w; w += gathers ? 4*n : 0;
        src = w; w += n;
        inOff = w; w += n;
        inPtr = w; w += n;
        inLen = w; w += n;
        outPtr = w; w += n;
        hostWords = w;
        outLen = w; w += n;
        off = w; w += n;
        total = w; w += 1;
        status = w; w += (n + 1)/2;   // (statuses are 4 bytes)
        words = w;
    }
    uint64_t resultBytes() const { return 8*(words - hostWords); }
};

}  // namespace

bool gpuWriteTiffAvailable(std::string &why){
    const Api* a = api(&why);
    if(!a) return false;
    CUdevice dev = 0;
    if(a->cuDeviceGet(&dev, 0)){
        why = "no NVIDIA GPU was found";
        return false;
    }
    return true;
}

uint8_t gpuWriteTiff(const char* fileName, const void* data, const uint64_t x, const uint64_t y, const uint64_t z,
                     const uint64_t bits, const uint64_t samples, const uint16_t sampleFormat,
                     const std::string &compression, const char* modeGiven, const bool flipXY, const uint64_t tileSize,
                     void* stream, const int threadsGiven, std::string &errString){
    errString.clear();
    const std::string name = fileName ? fileName : "";
    std::string comp(compression);
    for(char &ch : comp) ch = (char)std::tolower((unsigned char)ch);
    if(comp.empty()) comp = "none";
    const char* mode = modeGiven ? modeGiven : "w";
    if(!fileName || !*fileName){ errString = "no file name was given"; return 1; }
    if(comp != "lzw" && comp != "zstd" && comp != "none"){
        errString = "unknown compression \""+compression+"\" (use lzw, zstd or none)";
        return 1;
    }
    if(strcmp(mode, "w") && strcmp(mode, "a")){
        errString = std::string("mode \"")+mode+"\" is not supported. Use w or a for mode type";
        return 1;
    }
    if(bits != 8 && bits != 16 && bits != 32 && bits != 64){
        errString = std::to_string(bits)+"-bit samples are not supported";
        return 1;
    }
    if(samples != 1 && samples != 3 && samples != 4){
        errString = std::to_string(samples)+" samples per pixel are not supported (1 = grayscale, 3 = RGB, 4 = RGBA)";
        return 1;
    }
    if(!x || !y || !z){
        errString = "there is nothing to write ("+std::to_string(x)+" x "+std::to_string(y)+" pixels, "+std::to_string(z)+" slices)";
        return 1;
    }
    if(x > 0xFFFFFFFFull || y > 0xFFFFFFFFull){
        errString = "a TIFF image is at most 4294967295 pixels wide and high";
        return 1;
    }
    if(tileSize % 16){
        errString = "tile size "+std::to_string(tileSize)+" is not a multiple of 16";
        return 1;
    }
    std::string why;
    const Api* a = api(&why);
    if(!a){
        errString = "GPU writes cannot run here: "+why;
        return 1;
    }
    const int threads = threadsGiven > 0 ? threadsGiven : std::max(1, omp_get_max_threads());
    const CUstream s = (CUstream)stream;
    const bool profile = std::getenv("CPPTIFF_GPU_PROFILE") && *std::getenv("CPPTIFF_GPU_PROFILE");
    const bool strict = envOn("CPPTIFF_GPU_STRICT");
    const auto tStart = std::chrono::steady_clock::now();
    double tGpu = 0, tWriter = 0, tEncode = 0;
    const uint64_t es = bits/8;
    Strips g;
    g.x = x; g.y = y; g.spp = samples; g.es = es;
    g.rowBytes = x*samples*es;
    g.sliceBytes = y*g.rowBytes;
    const uint64_t totalBytes = g.sliceBytes*z;

    // The writer's state: its chunks, its progress, and whether either side failed
    std::unique_ptr<PreparedChunk[]> chunks;
    std::atomic<int64_t> writerChunk{0};
    std::atomic<bool> failed{false};
    std::string writerWhy;
    std::future<uint8_t> writer;
    WriteSlot* slots[2] = {nullptr, nullptr};
    Ctx* cx = nullptr;
    CUevent callerDone = nullptr;
    // At the end, and after an error: the writer ends (an error has set failed, which stops it), and the slots are
    // given back once their GPU work has finished (after an error, work may have been queued after the slot's last
    // event: a slot whose work does not finish within seconds is never reused)
    auto finish = [&](bool afterError){
        if(writer.valid()) writer.wait();
        if(!cx) return;
        CtxScope sc(a, cx->ctx);
        for(WriteSlot* &sl : slots){
            if(!sl) continue;
            if(afterError && sl->stream && sl->done){
                if(a->cuEventRecord(sl->done, sl->stream)) sl->lost = true;
                else{
                    try{ waitFor(a, sl->done, 10, "GPU write"); }
                    catch(...){ sl->lost = true; }
                }
            }
            else if(sl->done && a->cuEventQuery(sl->done) != 0) sl->lost = true;
            giveWriteSlot(*cx, *sl);
            sl = nullptr;
        }
        if(callerDone){ a->cuEventDestroy(callerDone); callerDone = nullptr; }
    };
    size_t nParts = 0;
    uint64_t gpuBytes = 0, pinnedBytes = 0;   // (the slots' buffers, for CPPTIFF_GPU_PROFILE)
    How how = AsTheyAre;
    try{
        if(!data || !isGpuMemory(a, data)) throw Fail("the data is not GPU memory");
        const CUcontext ctx = memoryContext(a, data);
        CtxScope scope(a, ctx);
        Ctx &c = context(a, ctx);
        cx = &c;

        // A tiled write: copied to the host whole and written by the CPU writer
        if(tileSize){
            if(strict) throw Fail("CPPTIFF_GPU_STRICT is set, and a tiled write is written from a copy in host memory");
            std::unique_ptr<uint8_t, decltype(&std::free)> host((uint8_t*)std::malloc(totalBytes), &std::free);
            if(!host) throw Fail("out of host memory for a copy of the data ("+std::to_string(totalBytes >> 20)+" MiB)");
            ck(a, a->cuMemcpyDtoHAsync(host.get(), (CUdeviceptr)(uintptr_t)data, totalBytes, s), "cuMemcpyDtoHAsync");
            ck(a, a->cuStreamSynchronize(s), "cuStreamSynchronize");
            if(writeTiffParallelHelper(fileName, host.get(), bits, mode, x, y, z, 0, flipXY, comp, sampleFormat, tileSize, samples))
                throw Fail("the CPU writer could not write \""+name+"\" (its message is above)");
            finish(false);
            return 0;
        }

        // How the strips are made, and their rows: the CPU writer's, but for zstd strips bigger than nvCOMP compresses
        g.rows = preparedStripRows(x, y, z, samples*es);
        how = comp == "zstd" ? OnGpu : comp == "lzw" ? OnHost : AsTheyAre;
        std::string hostWhy;
        if(how == OnGpu){
            if(envOn("CPPTIFF_GPU_WRITE_HOST")) hostWhy = "CPPTIFF_GPU_WRITE_HOST is set";
            else if(!a->compressAsync[Zstd]) hostWhy = "the nvCOMP loaded ("+a->nvcompPath+") does not compress zstd";
            else if(g.rowBytes > zstdMaxStream) hostWhy = "a row is more than nvCOMP compresses in one zstd frame";
            else if(g.rows*g.rowBytes > zstdMaxStream) g.rows = std::max<uint64_t>(1, zstdMaxStream/g.rowBytes);
            if(!hostWhy.empty()){
                if(strict && !envOn("CPPTIFF_GPU_WRITE_HOST"))
                    throw Fail("CPPTIFF_GPU_STRICT is set, and the zstd strips would be compressed on the host: "+hostWhy);
                how = OnHost;
            }
        }
        g.perSlice = (y + g.rows - 1)/g.rows;
        const uint64_t nChunks = z*g.perSlice;
        const uint64_t maxStrip = std::min(g.rows, y)*g.rowBytes;

        // nvCOMP's sizes and alignments
        NvcompOpts opts{};
        NvcompAlignments al{1, 1, 1};
        uint64_t outSlot = 0, inAlign = 16;
        if(how == OnGpu){
            if(a->compAlignments[Zstd](opts, &al)) al = {1, 1, 1};
            inAlign = std::max<uint64_t>(16, al.input);
            size_t maxOut = 0;
            const int r = a->compMaxOutput[Zstd](maxStrip, opts, &maxOut);
            if(r) throw Fail("nvCOMP could not size its compressed strips (status "+std::to_string(r)+")");
            outSlot = alignUp(maxOut, std::max<uint64_t>(16, al.output));
        }

        // (room for a strip compressed on the host: the CPU writer's for LZW, zstd's bound)
        const uint64_t encBound = how != OnHost ? 0 : comp == "lzw" ? (maxStrip + 2000*es)*2 + 1 : (uint64_t)ZSTD_compressBound((size_t)maxStrip);

        // The parts: runs of strips of at most partMax bytes. nvCOMP compresses a batch's strips side by side, so a part
        // of zstd strips holds about 6 of them for each of the GPU's multiprocessors (256 MB to 2 GB): writing 8 GB of
        // 2 MB strips to memory, parts of 256 MB took a B200 2.7 times as long as copying back and writing on the CPU,
        // parts of 1, 2 and 4 GB 0.79, 0.68 and 0.78 times as long (with 4 GB, the 2 parts overlapped little), and an
        // A100 0.51, 0.54 and 0.60 times. Other parts are 256 MB (LZW was as fast in parts of 1 GB). A part is at most
        // a 24th of the GPU's free memory, as a write takes four to five times a part there (the strips gathered,
        // nvCOMP's output and its temporary memory, about twice the part, and two parts' output packed): under a fifth
        // of it. CPPTIFF_GPU_WRITE_PART_MB sets the size instead.
        uint64_t partMax = (uint64_t)256 << 20;
        if(how == OnGpu)
            partMax = std::min<uint64_t>((uint64_t)2 << 30, std::max<uint64_t>(partMax, 6*(uint64_t)std::max(1, c.sms)*maxStrip));
        if(std::getenv("CPPTIFF_GPU_WRITE_PART_MB") && *std::getenv("CPPTIFF_GPU_WRITE_PART_MB"))
            partMax = std::max<uint64_t>(1, envNumber("CPPTIFF_GPU_WRITE_PART_MB", 256)) << 20;
        else{
            size_t freeMem = 0, totalMem = 0;
            if(!a->cuMemGetInfo(&freeMem, &totalMem) && freeMem)
                partMax = std::min<uint64_t>(partMax, std::max<uint64_t>(freeMem/24, (uint64_t)16 << 20));
        }
        std::vector<Part> parts;
        {
            Part p;
            for(uint64_t i = 0; i < nChunks; i++){
                const uint64_t b = g.bytes(i);
                if(i > p.begin && p.bytes + b > partMax){
                    p.end = i;
                    parts.push_back(p);
                    p = Part();
                    p.begin = i;
                }
                p.bytes += b;
            }
            p.end = nChunks;
            parts.push_back(p);
        }
        nParts = parts.size();
        const double timeout = std::getenv("CPPTIFF_GPU_TIMEOUT") ? (double)envNumber("CPPTIFF_GPU_TIMEOUT", 0)
                                                                   : 30.0 + (double)totalBytes/50e6;
        // (C-order strips nvCOMP can take where they are: every one aligned as it needs)
        bool inPlace = !flipXY;
        if(how == OnGpu && inPlace && al.input > 1)
            for(uint64_t i = 0; i < g.perSlice && inPlace; i++)
                inPlace = ((uintptr_t)data + g.offset(i)) % al.input == 0 && g.sliceBytes % al.input == 0;

        slots[0] = &takeWriteSlot(c);
        slots[1] = &takeWriteSlot(c);
        ck(a, a->cuEventCreate(&callerDone, 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
        ck(a, a->cuEventRecord(callerDone, s), "cuEventRecord");
        for(WriteSlot* sl : slots){
            if(!sl->stream) ck(a, a->cuStreamCreate(&sl->stream, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate");
            ck(a, a->cuStreamWaitEvent(sl->stream, callerDone, 0), "cuStreamWaitEvent");
            if(!sl->done) ck(a, a->cuEventCreate(&sl->done, 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
            if(!sl->sharedDone) ck(a, a->cuEventCreate(&sl->sharedDone, 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
        }
        // The buffers the parts share, in the first slot: the strips gathered, nvCOMP's output and its temporary memory,
        // each sized once for the biggest part. Kept a pair for each slot, they held 17.6 GB of a B200's memory after
        // a write of zstd strips (nvCOMP's temporary memory is about twice a part); shared, a part's GPU work waits for
        // the part before to be done with them (to be packed), which costs nothing when the GPU is what the write waits
        // for, and the rest of its work (the copies back, the writing) overlaps as before.
        WriteSlot &sh = *slots[0];
        const bool gathers = flipXY || (how == OnGpu && !inPlace);
        // (each strip's place in the shared input: packed, as they are copied back, or aligned for nvCOMP)
        auto inBytesOf = [&](const Part &p){
            uint64_t b = 0;
            for(uint64_t i = p.begin; i < p.end; i++) b += how == OnGpu ? alignUp(g.bytes(i), inAlign) : g.bytes(i);
            return b;
        };
        size_t tempNeed = 0;
        {
            uint64_t maxIn = 0, maxN = 0, maxBytes = 0;
            for(const Part &p : parts){
                if(gathers) maxIn = std::max(maxIn, inBytesOf(p));
                maxN = std::max(maxN, p.end - p.begin);
                maxBytes = std::max(maxBytes, p.bytes);
            }
            if(gathers) ensureDevice(a, sh.in, sh.inBytes, maxIn + 16);
            if(how == OnGpu){
                ensureDevice(a, sh.out, sh.outBytes, maxN*outSlot + 16);
                const int r = a->compTempSize6[Zstd] ? a->compTempSize6[Zstd](maxN, maxStrip, opts, &tempNeed, maxBytes, sh.stream)
                                                     : a->compTempSize5[Zstd](maxN, maxStrip, opts, &tempNeed, maxBytes);
                if(r) throw Fail("nvCOMP could not size its temporary memory (status "+std::to_string(r)+")");
                ensureDevice(a, sh.temp, sh.tempBytes, tempNeed + 16);
            }
        }

        // The writer, on a thread of its own, writing the strips as they become ready
        chunks.reset(new PreparedChunk[nChunks]);
        writer = std::async(std::launch::async, [&]{
            return writePreparedTiff(fileName, mode, x, y, z, bits, samples, sampleFormat, comp, g.rows, chunks.get(),
                                     writerChunk, failed, writerWhy);
        });
        // Waits until every strip before end is written
        auto waitWritten = [&](uint64_t end){
            const auto t0 = std::chrono::steady_clock::now();
            while(writerChunk.load(std::memory_order_acquire) < (int64_t)end){
                if(failed.load()) throw Fail(writerWhy.empty() ? "the writer failed" : writerWhy);
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            tWriter += secondsSince(t0);
        };
        auto waitGpu = [&](WriteSlot &sl){
            const auto t0 = std::chrono::steady_clock::now();
            try{ waitFor(a, sl.done, timeout, "GPU write"); }
            catch(...){ sl.lost = true; throw; }
            tGpu += secondsSince(t0);
        };

        // A part's GPU work: its strips put in row order (MATLAB's layout), compressed and packed together by the GPU
        // (in two steps: the packed size is needed to copy them back), or copied back as they are
        auto queueA = [&](size_t pi){
            const Part &p = parts[pi];
            WriteSlot &sl = *slots[pi % 2];
            const uint64_t n = p.end - p.begin;
            const Tables t(n, flipXY);
            growPinned(a, sl.tab, sl.tabBytes, 8*t.hostWords + 8);
            growPinned(a, sl.res, sl.resBytes, t.resultBytes() + 8);
            growDevice(a, sl.work, sl.workBytes, 8*t.words + 8);
            uint64_t* tab = reinterpret_cast<uint64_t*>(sl.tab);
            // each strip's place in the shared input (see inBytesOf)
            uint64_t inBytes = 0;
            for(uint64_t k = 0; k < n; k++){
                const uint64_t i = p.begin + k, b = g.bytes(i);
                tab[t.inOff + k] = inBytes;
                inBytes += how == OnGpu ? alignUp(b, inAlign) : b;
            }
            for(uint64_t k = 0; k < n; k++){
                const uint64_t i = p.begin + k;
                if(flipXY){
                    tab[t.gather + 4*k] = i / g.perSlice;
                    tab[t.gather + 4*k + 1] = (i % g.perSlice)*g.rows;
                    tab[t.gather + 4*k + 2] = g.rowsOf(i);
                    tab[t.gather + 4*k + 3] = tab[t.inOff + k]/es;
                }
                tab[t.src + k] = (uint64_t)(uintptr_t)data + g.offset(i);
                tab[t.inPtr + k] = gathers ? (uint64_t)sh.in + tab[t.inOff + k] : tab[t.src + k];
                tab[t.inLen + k] = g.bytes(i);
                tab[t.outPtr + k] = (uint64_t)sh.out + k*outSlot;
            }
            ck(a, a->cuMemcpyHtoDAsync(sl.work, sl.tab, 8*t.hostWords, sl.stream), "cuMemcpyHtoDAsync");
            // (the shared buffers once the part before is done with them)
            if(pi >= 1) ck(a, a->cuStreamWaitEvent(sl.stream, slots[(pi - 1) % 2]->sharedDone, 0), "cuStreamWaitEvent");
            const CUdeviceptr w = sl.work;
            unsigned nn = (unsigned)n;
            if(flipXY){
                const int ei = es == 1 ? 0 : es == 2 ? 1 : es == 4 ? 2 : 3;
                CUdeviceptr tg = w + 8*t.gather, din = (CUdeviceptr)(uintptr_t)data, dout = sh.in;
                unsigned ux = (unsigned)x, uy = (unsigned)y, us = (unsigned)samples;
                const uint64_t tiles = ((x + 31)/32)*((g.rows + 31)/32)*samples;
                const unsigned gy = (unsigned)std::min<uint64_t>({tiles, 65535, std::max<uint64_t>(4, (32*(uint64_t)c.sms + n - 1)/n)});
                void* args[] = {&tg, &nn, &din, &dout, &ux, &uy, &us};
                launch(a, c.gatherF[ei], nn, gy, 32, 8, sl.stream, args);
            }
            else if(gathers){
                // (C-order strips not aligned as nvCOMP needs them, copied to aligned places)
                CUdeviceptr ps = w + 8*t.src, pl = w + 8*t.inLen, po = w + 8*t.inOff, dst = sh.in;
                void* args[] = {&ps, &pl, &po, &nn, &dst};
                launch(a, c.pack, (unsigned)std::min<uint64_t>(n, 65535), 1, 256, 1, sl.stream, args);
            }
            if(how == OnGpu){
                const int cr = a->compressAsync[Zstd]((const void* const*)(uintptr_t)(w + 8*t.inPtr), (const size_t*)(uintptr_t)(w + 8*t.inLen),
                                                      maxStrip, n, (void*)(uintptr_t)sh.temp, tempNeed,
                                                      (void* const*)(uintptr_t)(w + 8*t.outPtr), (size_t*)(uintptr_t)(w + 8*t.outLen), opts,
                                                      (int*)(uintptr_t)(w + 8*t.status), (void*)sl.stream);
                if(cr) throw Fail("nvCOMP could not compress (status "+std::to_string(cr)+")");
                CUdeviceptr pl = w + 8*t.outLen, po = w + 8*t.off, pt = w + 8*t.total;
                void* args[] = {&pl, &po, &nn, &pt};
                launch(a, c.scan, 1, 1, 1024, 1, sl.stream, args);
                ck(a, a->cuMemcpyDtoHAsync(sl.res, w + 8*t.hostWords, t.resultBytes(), sl.stream), "cuMemcpyDtoHAsync");
            }
            else{
                growPinned(a, sl.host, sl.hostBytes, p.bytes);
                if(how == OnHost && sl.encBytes < n*encBound){
                    std::free(sl.enc);
                    sl.encBytes = 0;
                    sl.enc = (uint8_t*)std::malloc(n*encBound);
                    if(!sl.enc) throw Fail("out of host memory for the compressed strips");
                    sl.encBytes = n*encBound;
                }
                const CUdeviceptr from = flipXY ? sh.in : (CUdeviceptr)(uintptr_t)data + g.offset(p.begin);
                ck(a, a->cuMemcpyDtoHAsync(sl.host, from, p.bytes, sl.stream), "cuMemcpyDtoHAsync");
                ck(a, a->cuEventRecord(sl.sharedDone, sl.stream), "cuEventRecord");
            }
            ck(a, a->cuEventRecord(sl.done, sl.stream), "cuEventRecord");
        };
        // The compressed strips packed together and copied back, once their sizes are known; returns the bytes
        auto queueB = [&](size_t pi){
            const Part &p = parts[pi];
            WriteSlot &sl = *slots[pi % 2];
            const uint64_t n = p.end - p.begin;
            const Tables t(n, flipXY);
            const uint64_t* res = reinterpret_cast<const uint64_t*>(sl.res);
            const int32_t* status = reinterpret_cast<const int32_t*>(res + (t.status - t.hostWords));
            for(uint64_t k = 0; k < n; k++){
                const uint64_t len = res[t.outLen - t.hostWords + k];
                if(status[k] || !len || len > outSlot)
                    throw Fail("nvCOMP could not compress strip "+std::to_string((p.begin + k) % g.perSlice)+" of slice "+
                               std::to_string((p.begin + k)/g.perSlice)+" (status "+std::to_string(status[k])+")");
            }
            const uint64_t total = res[t.total - t.hostWords];
            growDevice(a, sl.packed, sl.packedBytes, total + 16);
            growPinned(a, sl.host, sl.hostBytes, total + 16);
            const CUdeviceptr w = sl.work;
            CUdeviceptr ps = w + 8*t.outPtr, pl = w + 8*t.outLen, po = w + 8*t.off, dst = sl.packed;
            unsigned nn = (unsigned)n;
            void* args[] = {&ps, &pl, &po, &nn, &dst};
            launch(a, c.pack, (unsigned)std::min<uint64_t>(n, 65535), 1, 256, 1, sl.stream, args);
            ck(a, a->cuEventRecord(sl.sharedDone, sl.stream), "cuEventRecord");
            ck(a, a->cuMemcpyDtoHAsync(sl.host, sl.packed, total, sl.stream), "cuMemcpyDtoHAsync");
            ck(a, a->cuEventRecord(sl.done, sl.stream), "cuEventRecord");
        };
        // The part's strips handed to the writer (but strips compressed on the host: see hostLoop)
        auto publish = [&](size_t pi){
            const Part &p = parts[pi];
            WriteSlot &sl = *slots[pi % 2];
            const uint64_t n = p.end - p.begin;
            if(how == OnGpu){
                const Tables t(n, flipXY);
                const uint64_t* res = reinterpret_cast<const uint64_t*>(sl.res);
                for(uint64_t k = 0; k < n; k++){
                    PreparedChunk &ch = chunks[p.begin + k];
                    ch.data = sl.host + res[t.off - t.hostWords + k];
                    ch.size.store((int64_t)res[t.outLen - t.hostWords + k], std::memory_order_release);
                }
                return;
            }
            const uint64_t base = g.offset(p.begin);
            for(uint64_t k = 0; k < n; k++){
                PreparedChunk &ch = chunks[p.begin + k];
                ch.data = sl.host + (g.offset(p.begin + k) - base);
                ch.size.store((int64_t)g.bytes(p.begin + k), std::memory_order_release);
            }
        };

        // Strips compressed on the host (LZW): a team of threads takes them in order as their parts come back, with no
        // wait at the end of each part (a part's last strips left most threads idle: the write ran 5-7% slower than
        // copying back and writing with the CPU writer), and this thread, which copies the parts back, compresses one
        // whenever it would otherwise wait. Strips before readyEnd are in host memory; nextStrip is the next to take.
        auto hostLoop = [&](){
            const bool lzw = comp == "lzw";
            std::atomic<uint64_t> readyEnd{0}, nextStrip{0};
            std::atomic<int> bad{0};
            auto encode = [&](uint64_t k){
                const size_t pi = (size_t)(std::upper_bound(parts.begin(), parts.end(), k,
                                                            [](uint64_t v, const Part &q){ return v < q.begin; }) - parts.begin()) - 1;
                WriteSlot &sl = *slots[pi % 2];
                const uint64_t b = g.bytes(k);
                uint8_t* src = sl.host + (g.offset(k) - g.offset(parts[pi].begin));
                uint8_t* dst = sl.enc + (k - parts[pi].begin)*encBound;
                uint64_t m = 0;
                if(lzw) m = lzwEncode(src, dst, (tmsize_t)b);
                else{
                    const size_t r = ZSTD_compress(dst, (size_t)encBound, src, (size_t)b, hostZstdLevel);
                    m = ZSTD_isError(r) ? 0 : (uint64_t)r;
                }
                if(!m || m > encBound){ bad = 1; failed = true; return; }
                chunks[k].data = dst;
                chunks[k].size.store((int64_t)m, std::memory_order_release);
            };
            // (one ready strip compressed by this thread, if there is one not taken)
            auto helpOne = [&]() -> bool {
                uint64_t k = nextStrip.load();
                if(k >= readyEnd.load(std::memory_order_acquire)) return false;
                if(nextStrip.compare_exchange_strong(k, k + 1)) encode(k);
                return true;
            };
            auto idle = [&](){ if(!helpOne()) std::this_thread::sleep_for(std::chrono::microseconds(20)); };
            const auto t0 = std::chrono::steady_clock::now();
            std::exception_ptr makerError;
            const int encoders = std::max(1, threads - 1);
            #pragma omp parallel num_threads(encoders + 1)
            {
                if(omp_get_thread_num() == 0){
                    try{
                        queueA(0);
                        for(size_t pi = 0; pi < parts.size(); pi++){
                            WriteSlot &sl = *slots[pi % 2];
                            const auto tg = std::chrono::steady_clock::now();
                            for(;;){
                                const CUresult r = a->cuEventQuery(sl.done);
                                if(r == 0) break;
                                if(r != 600 /* CUDA_ERROR_NOT_READY */){ sl.lost = true; ck(a, r, "GPU write"); }
                                if(timeout > 0 && secondsSince(tg) > timeout){
                                    sl.lost = true;
                                    throw Fail("the GPU did not finish within "+std::to_string((long long)timeout)+" s (CPPTIFF_GPU_TIMEOUT sets the limit)");
                                }
                                if(failed.load()) throw Fail(writerWhy.empty() ? "the writer failed" : writerWhy);
                                idle();
                            }
                            tGpu += secondsSince(tg);
                            readyEnd.store(parts[pi].end, std::memory_order_release);
                            if(pi + 1 < parts.size()){
                                // (the other slot's last part written first: its buffers are the next part's)
                                const auto tw = std::chrono::steady_clock::now();
                                while(pi >= 1 && writerChunk.load(std::memory_order_acquire) < (int64_t)parts[pi - 1].end){
                                    if(failed.load()) throw Fail(writerWhy.empty() ? "the writer failed" : writerWhy);
                                    idle();
                                }
                                tWriter += secondsSince(tw);
                                queueA(pi + 1);
                            }
                        }
                        while(nextStrip.load() < parts.back().end && !failed.load()) idle();
                    }
                    catch(...){
                        makerError = std::current_exception();
                        failed = true;
                    }
                }
                else{
                    for(;;){
                        const uint64_t k = nextStrip.fetch_add(1);
                        if(k >= parts.back().end) break;
                        while(readyEnd.load(std::memory_order_acquire) <= k && !failed.load())
                            std::this_thread::sleep_for(std::chrono::microseconds(20));
                        if(failed.load()) break;
                        encode(k);
                    }
                }
            }
            tEncode = secondsSince(t0);
            if(makerError) std::rethrow_exception(makerError);
            if(bad) throw Fail("a strip could not be compressed on the host");
            if(failed.load()) throw Fail(writerWhy.empty() ? "the writer failed" : writerWhy);
        };

        if(how == OnHost) hostLoop();
        else queueA(0);
        for(size_t pi = 0; how != OnHost && pi < parts.size(); pi++){
            WriteSlot &sl = *slots[pi % 2];
            waitGpu(sl);
            if(how == OnGpu) queueB(pi);
            if(pi + 1 < parts.size()){
                // (the other slot's last part written first: its buffers are the next part's)
                if(pi >= 1) waitWritten(parts[pi - 1].end);
                queueA(pi + 1);
            }
            if(how == OnGpu) waitGpu(sl);
            if(failed.load()) throw Fail(writerWhy.empty() ? "the writer failed" : writerWhy);
            publish(pi);
        }
        const auto tw = std::chrono::steady_clock::now();
        const uint8_t werr = writer.get();
        tWriter += secondsSince(tw);
        if(werr) throw Fail(writerWhy.empty() ? "the writer failed" : writerWhy);
        for(const WriteSlot* sl : slots){
            gpuBytes += sl->inBytes + sl->outBytes + sl->tempBytes + sl->workBytes + sl->packedBytes;
            pinnedBytes += sl->hostBytes + sl->tabBytes + sl->resBytes;
        }
        finish(false);
    }
    catch(const std::exception &e){
        failed = true;
        finish(true);
        errString = std::string("GPU write of \"")+name+"\": "+e.what();
        return 1;
    }
    if(profile){
        const double total = secondsSince(tStart);
        std::fprintf(stderr, "cpptiff GPU write %s: %.1f MB, %s, %zu parts: %.3f s (waiting for the GPU %.3f s, for the writer "
                     "%.3f s, compressing on the host %.3f s); buffers kept: %.0f MB on the GPU, %.0f MB pinned\n", name.c_str(),
                     totalBytes/1e6, how == OnGpu ? "zstd on the GPU" : how == OnHost ? (comp+" on the host").c_str() : "uncompressed",
                     nParts, total, tGpu, tWriter, tEncode, gpuBytes/1e6, pinnedBytes/1e6);
    }
    return 0;
}
