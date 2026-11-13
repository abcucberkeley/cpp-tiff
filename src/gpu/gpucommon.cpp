#include "gpucommon.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <thread>
#include "kernels_ptx.h"

namespace ctcuda {
namespace {
std::mutex ctxMutex;
std::vector<std::unique_ptr<Ctx>> ctxs;
}  // namespace

void ck(const Api* a, CUresult r, const char* what){
    if(r) throw Fail(std::string(what)+": "+errorText(a, r));
}

// Whether the context's kernels and buffers are still its own: not after it was reset or destroyed (its handle can
// come back for a new one), when its sentinel allocation is gone or belongs to another allocation
static bool stillValid(const Api* a, const Ctx &x){
    if(!x.sentinel) return true;
    unsigned long long id = 0;
    return !a->cuPointerGetAttribute(&id, 7 /* CU_POINTER_ATTRIBUTE_BUFFER_ID */, x.sentinel) && id == x.sentinelId;
}

Ctx& context(const Api* a, CUcontext c){
    std::lock_guard<std::mutex> g(ctxMutex);
    for(auto &x : ctxs){
        if(x->ctx != c || x->stale) continue;
        if(stillValid(a, *x)) return *x;
        // (reset: its kernels and buffers went with it; kept, not freed, in case a call still holds it)
        x->stale = true;
    }
    std::unique_ptr<Ctx> x(new Ctx());
    x->ctx = c;
    if(!a->cuMemAlloc(&x->sentinel, 256) &&
       a->cuPointerGetAttribute(&x->sentinelId, 7 /* CU_POINTER_ATTRIBUTE_BUFFER_ID */, x->sentinel)){
        a->cuMemFree(x->sentinel);
        x->sentinel = 0;
    }
    std::string ptx;
    for(const char* p : cpptiffKernelsPtx) ptx += p;
    char log[4096] = {0};
    int opts[2] = {5 /* CU_JIT_ERROR_LOG_BUFFER */, 6 /* CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES */};
    void* vals[2] = {log, (void*)(uintptr_t)sizeof(log)};
    const CUresult r = a->cuModuleLoadDataEx(&x->mod, ptx.c_str(), 2, opts, vals);
    if(r) throw Fail("The GPU kernels could not be loaded ("+errorText(a, r)+(log[0] ? ": "+std::string(log) : "")+
                     "): the NVIDIA driver may be too old");
    const char* names[4] = {"8", "16", "32", "64"};
    for(int i = 0; i < 4; i++){
        ck(a, a->cuModuleGetFunction(&x->placeC[i], x->mod, (std::string("ct_place_c")+names[i]).c_str()), "cuModuleGetFunction");
        ck(a, a->cuModuleGetFunction(&x->placeF[i], x->mod, (std::string("ct_place_f")+names[i]).c_str()), "cuModuleGetFunction");
    }
    ck(a, a->cuModuleGetFunction(&x->unpredict, x->mod, "ct_unpredict"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->check, x->mod, "ct_check"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->lzw, x->mod, "ct_lzw"), "cuModuleGetFunction");
    for(int i = 0; i < 4; i++)
        ck(a, a->cuModuleGetFunction(&x->gatherF[i], x->mod, (std::string("ct_gather_f")+names[i]).c_str()), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->scan, x->mod, "ct_scan"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->pack, x->mod, "ct_pack"), "cuModuleGetFunction");
    CUdevice dev = 0;
    ck(a, a->cuCtxGetDevice(&dev), "cuCtxGetDevice");
    a->cuDeviceGetAttribute(&x->sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, dev);
    for(int k = 0; k < NCodecs; k++){
        x->align[k] = {1, 1, 1};
        if(a->alignments[k]){
            NvcompOpts o{};
            if(a->alignments[k](o, &x->align[k])) x->align[k] = {1, 1, 1};
        }
    }
    ctxs.push_back(std::move(x));
    return *ctxs.back();
}

Slot& takeSlot(Ctx &c){
    std::lock_guard<std::mutex> g(c.m);
    for(auto &s : c.slots) if(!s->busy && !s->lost){ s->busy = true; return *s; }
    c.slots.emplace_back(new Slot());
    c.slots.back()->busy = true;
    return *c.slots.back();
}

void giveSlot(Ctx &c, Slot &s){
    std::lock_guard<std::mutex> g(c.m);
    s.busy = false;
}

WriteSlot& takeWriteSlot(Ctx &c){
    std::lock_guard<std::mutex> g(c.m);
    for(auto &s : c.writeSlots) if(!s->busy && !s->lost){ s->busy = true; return *s; }
    c.writeSlots.emplace_back(new WriteSlot());
    c.writeSlots.back()->busy = true;
    return *c.writeSlots.back();
}

void giveWriteSlot(Ctx &c, WriteSlot &s){
    std::lock_guard<std::mutex> g(c.m);
    s.busy = false;
}

void releaseBuffers(const Api* a){
    std::lock_guard<std::mutex> g(ctxMutex);
    for(auto &x : ctxs){
        if(x->stale || !stillValid(a, *x)) continue;
        CtxScope scope(a, x->ctx);
        std::lock_guard<std::mutex> gs(x->m);
        // (a slot no call holds has finished its GPU work: slots are given back once it has, or marked lost)
        for(auto &s : x->slots){
            if(s->busy || s->lost) continue;
            for(CUdeviceptr* p : {&s->comp, &s->dec, &s->temp, &s->work}) if(*p){ a->cuMemFree(*p); *p = 0; }
            if(s->host){ a->cuMemFreeHost(s->host); s->host = nullptr; }
            s->compBytes = s->decBytes = s->tempBytes = s->workBytes = s->hostBytes = 0;
        }
        for(auto &s : x->writeSlots){
            if(s->busy || s->lost) continue;
            for(CUdeviceptr* p : {&s->in, &s->out, &s->temp, &s->work, &s->packed}) if(*p){ a->cuMemFree(*p); *p = 0; }
            for(uint8_t** p : {&s->host, &s->tab, &s->res}) if(*p){ a->cuMemFreeHost(*p); *p = nullptr; }
            std::free(s->enc);
            s->enc = nullptr;
            s->inBytes = s->outBytes = s->tempBytes = s->workBytes = s->packedBytes = 0;
            s->hostBytes = s->tabBytes = s->resBytes = s->encBytes = 0;
        }
    }
}

void waitFor(const Api* a, CUevent ev, double timeout, const char* what){
    const auto t0 = std::chrono::steady_clock::now();
    for(;;){
        const CUresult r = a->cuEventQuery(ev);
        if(r == 0) return;
        if(r != 600 /* CUDA_ERROR_NOT_READY */) ck(a, r, what);
        if(timeout > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout)
            throw Fail(std::string(what)+": the GPU did not finish within "+std::to_string((long long)timeout)+
                       " s (damaged data can make nvCOMP hang; CPPTIFF_GPU_TIMEOUT sets the limit)");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void launch(const Api* a, CUfunction f, unsigned gx, unsigned gy, unsigned bx, unsigned by, CUstream s, void** args){
    ck(a, a->cuLaunchKernel(f, gx, gy, 1, bx, by, 1, 0, s, args, nullptr), "cuLaunchKernel");
}

uint64_t envNumber(const char* name, uint64_t dflt){
    const char* e = std::getenv(name);
    if(!e || !*e) return dflt;
    const long long v = std::atoll(e);
    return v >= 0 ? (uint64_t)v : dflt;
}

void growDevice(const Api* a, CUdeviceptr &p, size_t &have, const size_t need){
    if(need <= have) return;
    if(p) a->cuMemFree(p);
    p = 0;
    have = 0;
    const size_t n = std::max<size_t>(need+need/4, 1 << 20);
    ck(a, a->cuMemAlloc(&p, n), "Could not allocate GPU memory (cuMemAlloc)");
    have = n;
}

void growPinned(const Api* a, uint8_t* &p, size_t &have, const size_t need){
    if(need <= have) return;
    if(p) a->cuMemFreeHost(p);
    p = nullptr;
    have = 0;
    const size_t n = std::max<size_t>(need+need/4, 1 << 20);
    void* q = nullptr;
    ck(a, a->cuMemHostAlloc(&q, n, 0), "Could not allocate pinned host memory (cuMemHostAlloc)");
    p = static_cast<uint8_t*>(q);
    have = n;
}

CUcontext memoryContext(const Api* a, const void* p){
    CUcontext c = nullptr;
    ck(a, a->cuPointerGetAttribute(&c, 1 /* CU_POINTER_ATTRIBUTE_CONTEXT */, (CUdeviceptr)(uintptr_t)p), "cuPointerGetAttribute");
    if(c) return c;
    int ordinal = -1;
    ck(a, a->cuPointerGetAttribute(&ordinal, 9 /* CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL */, (CUdeviceptr)(uintptr_t)p),
       "cuPointerGetAttribute (device)");
    // (each device's primary context retained once: the runtime keeps it anyway)
    static std::mutex m;
    static std::map<int, CUcontext> primary;
    std::lock_guard<std::mutex> g(m);
    auto it = primary.find(ordinal);
    if(it != primary.end()) return it->second;
    CUdevice dev = 0;
    ck(a, a->cuDeviceGet(&dev, ordinal), "cuDeviceGet");
    ck(a, a->cuDevicePrimaryCtxRetain(&c, dev), "cuDevicePrimaryCtxRetain");
    primary[ordinal] = c;
    return c;
}

bool isGpuMemory(const Api* a, const void* p){
    unsigned int mt = 0;
    return !a->cuPointerGetAttribute(&mt, 2 /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE */, (CUdeviceptr)(uintptr_t)p) &&
           (mt == 2 /* DEVICE */ || mt == 4 /* UNIFIED */);
}

}  // namespace ctcuda
