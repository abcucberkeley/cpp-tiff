// What GPU reads (gpureadtiff.cpp) need of the GPU beyond the driver API: each CUDA context's kernels and
// reusable buffers, and the context GPU memory belongs to. Internal to the library (not installed);
// adapted from cpp-zarr's src/gpu/gpucommon.h.
#ifndef CPPTIFF_GPUCOMMON_H
#define CPPTIFF_GPUCOMMON_H
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "cudaapi.h"

namespace ctcuda {

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };

// Throws Fail (what: the driver's error) unless r is 0
void ck(const Api* a, CUresult r, const char* what);

// A part of a GPU read's reusable buffers
struct Slot {
    bool busy = false;
    bool lost = false;               // (its GPU work never finished: its buffers are never reused)
    uint8_t* host = nullptr;         // pinned staging: stored bytes, units decoded on the host, tables
    size_t hostBytes = 0;
    uint8_t* small = nullptr;        // pinned: the decompression check's results
    CUdeviceptr comp = 0, dec = 0, temp = 0, work = 0;   // the staging's copy, nvCOMP's output and temporary memory, statuses
    size_t compBytes = 0, decBytes = 0, tempBytes = 0, workBytes = 0;
    CUevent done = nullptr;          // recorded after the part's GPU work
    CUstream stream = nullptr;       // its parts' GPU work: the copy, the decompression, the kernels (two slots' parts overlap)
};

// The kernels and reusable buffers of one CUDA context
struct Ctx {
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction placeC[4] = {nullptr, nullptr, nullptr, nullptr};   // (by element size: 1, 2, 4, 8 bytes)
    CUfunction placeF[4] = {nullptr, nullptr, nullptr, nullptr};
    CUfunction unpredict = nullptr, check = nullptr, lzw = nullptr;
    int sms = 1;
    NvcompAlignments align[NCodecs];
    std::mutex m;
    std::vector<std::unique_ptr<Slot>> slots;
};

// The context's kernels and buffers (loading its kernels the first time); c must be current
Ctx& context(const Api* a, CUcontext c);

// A slot of the context, free until given back
Slot& takeSlot(Ctx &c);
void giveSlot(Ctx &c, Slot &s);

// GPU memory of at least need bytes in p (have of them), its contents not kept
void growDevice(const Api* a, CUdeviceptr &p, size_t &have, const size_t need);

// Pinned host memory of at least need bytes in p (have of them), its contents not kept
void growPinned(const Api* a, uint8_t* &p, size_t &have, const size_t need);

// The CUDA context GPU memory belongs to: the one it was allocated in, or for memory that belongs to none
// (CUDA's virtual memory API or a stream-ordered pool, as PyTorch's expandable segments and CuPy's async pool
// are) its device's primary context, the one the CUDA runtime uses
CUcontext memoryContext(const Api* a, const void* p);

// Whether p is GPU memory (device or managed)
bool isGpuMemory(const Api* a, const void* p);

// c current on this thread for the scope's life
struct CtxScope {
    const Api* a;
    bool pushed = false;
    CtxScope(const Api* api, CUcontext c) : a(api){ pushed = a->cuCtxPushCurrent(c) == 0; }
    ~CtxScope(){ if(pushed){ CUcontext was = nullptr; a->cuCtxPopCurrent(&was); } }
    CtxScope(const CtxScope&) = delete;
    CtxScope& operator=(const CtxScope&) = delete;
};

inline uint64_t alignUp(uint64_t n, uint64_t a){ return a > 1 ? (n + a - 1)/a*a : n; }

}  // namespace ctcuda
#endif
