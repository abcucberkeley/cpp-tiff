#include "gpucommon.h"
#include <algorithm>
#include <cstring>
#include <map>
#include "kernels_ptx.h"

namespace ctcuda {
namespace {
std::mutex ctxMutex;
std::vector<std::unique_ptr<Ctx>> ctxs;
}  // namespace

void ck(const Api* a, CUresult r, const char* what){
    if(r) throw Fail(std::string(what)+": "+errorText(a, r));
}

Ctx& context(const Api* a, CUcontext c){
    std::lock_guard<std::mutex> g(ctxMutex);
    for(auto &x : ctxs) if(x->ctx == c) return *x;
    std::unique_ptr<Ctx> x(new Ctx());
    x->ctx = c;
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
