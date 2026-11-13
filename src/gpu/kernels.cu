// GPU kernels of cpp-tiff's GPU reads (gpureadtiff.cpp), compiled to PTX by gen_kernels_ptx.py into
// kernels_ptx.h; the CUDA driver compiles that PTX for the GPU it runs on (Volta and every later one).
// Plain C kernels, no includes, so that NVRTC compiles them as they are.
//
// A read's stored data reaches these kernels as units: each a strip or a tile of one slice of one
// file (of one color plane of it, in a planar file; all of each pixel's samples together otherwise),
// decompressed by nvCOMP into the decode buffer, or decoded on the host (or stored uncompressed) in
// the copy of the host's staging buffer. A unit's rows are `pitch` elements apart; it covers `rows`
// rows from row0 and `cols` pixels from col0 of its slice (a tile at the image's right or bottom edge
// holds more than lies inside the image). The kernels undo the horizontal predictor, swap the bytes
// of big-endian files and put each unit in its place in the output: row-major (z, y, x, c), or
// column-major (y, x, c, z) as MATLAB has it (flip).

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

// (64 bytes; gpureadtiff.cpp's Unit is the same)
struct Unit {
    u64 src;        // its decoded bytes: dec + src, or with bit 63 set host + the rest of src
    u64 out;        // its read's GPU memory
    u32 slice;      // its slice in that memory
    u32 row0, col0, rows, cols;
    u32 pitch;      // elements from one of its rows to the next
    u32 x, y;       // the image's width and height
    u16 samples;    // samples per pixel
    u16 plane;      // its color plane (planar files), 0xFFFF for all of each pixel's samples
    u8 es;          // bytes per element: 1, 2, 4 or 8
    u8 swap;        // its elements' bytes are swapped (a big-endian file decompressed on the GPU)
    u8 predictor;   // the predictor still to undo: 1 none, 2 horizontal
    u8 pad[9];
};

#define HOSTBIT (1ull << 63)

__device__ __forceinline__ u8* unitData(const Unit& u, u8* dec, u8* host){
    return (u.src & HOSTBIT) ? host + (u.src & ~HOSTBIT) : dec + u.src;
}

__device__ __forceinline__ u16 bswap16(u16 v){ return (u16)((v >> 8) | (v << 8)); }
__device__ __forceinline__ u32 bswap32(u32 v){ return __byte_perm(v, 0, 0x0123); }
__device__ __forceinline__ u64 bswap64(u64 v){
    return ((u64)bswap32((u32)v) << 32) | (u64)bswap32((u32)(v >> 32));
}

// The horizontal predictor (TIFF's predictor 2), undone in place: each of a row's samples has the one
// a pixel before it added back, after its bytes are swapped to the host's order when they need it
// (as libtiff does). One block per unit of `list`, its threads a row each. The unit is then in the
// host's byte order.
#define UNPREDICT(T, SWAP)                                                                             \
    {                                                                                                  \
        T* p = (T*)(base + (u64)r * u.pitch * sizeof(T));                                              \
        if(u.swap) for(u32 k = 0; k < n; k++) p[k] = (T)SWAP(p[k]);                                    \
        for(u32 k = spp; k < n; k++) p[k] = (T)(p[k] + p[k - spp]);                                    \
    }
#define NOSWAP(v) (v)

extern "C" __global__ void ct_unpredict(const Unit* __restrict__ units, const u32* __restrict__ list, u32 nlist,
                                        u8* dec, u8* host){
    const u32 i = blockIdx.x;
    if(i >= nlist) return;
    const Unit u = units[list[i]];
    u8* base = unitData(u, dec, host);
    const u32 spp = u.plane == 0xFFFF ? u.samples : 1;
    // (the whole stored row: a tile's columns past the image's edge are part of the differencing)
    const u32 n = u.pitch;
    for(u32 r = threadIdx.x; r < u.rows; r += blockDim.x){
        switch(u.es){
            case 1: UNPREDICT(u8, NOSWAP) break;
            case 2: UNPREDICT(u16, bswap16) break;
            case 4: UNPREDICT(u32, bswap32) break;
            case 8: UNPREDICT(u64, bswap64) break;
        }
    }
}

// Units to row-major output (z, y, x, c). Block (u, g): unit u's groups of RPB rows g, g + gridDim.y,
// ..., the threads the elements across a row, so that reads and writes both run along memory. Elements are swapped on the way when the unit asks
// (its predictor already undone does that itself).
#define RPB 4
#define PLACEC(NAME, T, SWAP)                                                                          \
extern "C" __global__ void NAME(const Unit* __restrict__ units, u32 nunits, u8* dec, u8* host){         \
    const u32 ui = blockIdx.x;                                                                         \
    if(ui >= nunits) return;                                                                           \
    const Unit u = units[ui];                                                                          \
    const T* src = (const T*)unitData(u, dec, host);                                                   \
    T* out = (T*)u.out;                                                                                \
    const bool swap = u.swap && u.predictor == 1;                                                      \
    const u64 S = u.samples;                                                                           \
    for(u32 g = blockIdx.y; g * RPB < u.rows; g += gridDim.y)                                         \
    for(u32 r = g * RPB; r < u.rows && r < (g + 1) * RPB; r++){                                        \
        const T* in = src + (u64)r * u.pitch;                                                          \
        const u64 rowStart = (((u64)u.slice * u.y + u.row0 + r) * u.x + u.col0) * S;                   \
        if(u.plane == 0xFFFF){                                                                         \
            const u32 n = u.cols * (u32)S;                                                             \
            for(u32 k = threadIdx.x; k < n; k += blockDim.x){                                          \
                T v = in[k];                                                                           \
                out[rowStart + k] = swap ? (T)SWAP(v) : v;                                             \
            }                                                                                          \
        }                                                                                              \
        else{                                                                                          \
            for(u32 k = threadIdx.x; k < u.cols; k += blockDim.x){                                     \
                T v = in[k];                                                                           \
                out[rowStart + (u64)k * S + u.plane] = swap ? (T)SWAP(v) : v;                          \
            }                                                                                          \
        }                                                                                              \
    }                                                                                                  \
}
PLACEC(ct_place_c8, u8, NOSWAP)
PLACEC(ct_place_c16, u16, bswap16)
PLACEC(ct_place_c32, u32, bswap32)
PLACEC(ct_place_c64, u64, bswap64)

// Units to column-major output (y, x, c, z), MATLAB's layout: a transpose, through 32 x 32 tiles in
// shared memory, so that reads run along the unit's rows and writes along the output's columns. Block
// (u, t): unit u's tiles t, t + gridDim.y, ... (tiles of each of its samples in turn), 32 x 8
// threads.
#define TILE 32
#define PLACEF(NAME, T, SWAP)                                                                          \
extern "C" __global__ void NAME(const Unit* __restrict__ units, u32 nunits, u8* dec, u8* host){         \
    __shared__ T tile[TILE][TILE + 1];                                                                 \
    const u32 ui = blockIdx.x;                                                                         \
    if(ui >= nunits) return;                                                                           \
    const Unit u = units[ui];                                                                          \
    const T* src = (const T*)unitData(u, dec, host);                                                   \
    T* out = (T*)u.out;                                                                                \
    const bool swap = u.swap && u.predictor == 1;                                                      \
    const u32 spp = u.plane == 0xFFFF ? u.samples : 1;                                                 \
    const u32 tilesAcross = (u.cols + TILE - 1) / TILE, tilesDown = (u.rows + TILE - 1) / TILE;        \
    const u32 tiles = tilesAcross * tilesDown * spp;                                                   \
    const u64 plane = (u64)u.x * u.y;                                                                  \
    for(u32 t = blockIdx.y; t < tiles; t += gridDim.y){                                                \
        const u32 s = t / (tilesAcross * tilesDown), rest = t % (tilesAcross * tilesDown);             \
        const u32 c0 = (rest % tilesAcross) * TILE, r0 = (rest / tilesAcross) * TILE;                  \
        const u32 sample = u.plane == 0xFFFF ? s : u.plane;                                            \
        /* read: rows r0.., columns c0 + threadIdx.x */                                                \
        for(u32 j = threadIdx.y; j < TILE; j += blockDim.y){                                           \
            const u32 r = r0 + j, c = c0 + threadIdx.x;                                                \
            if(r < u.rows && c < u.cols){                                                              \
                T v = src[(u64)r * u.pitch + (u64)c * spp + (spp > 1 ? s : 0)];                        \
                tile[j][threadIdx.x] = swap ? (T)SWAP(v) : v;                                          \
            }                                                                                          \
        }                                                                                              \
        __syncthreads();                                                                               \
        /* write: columns c0.. of the output, rows r0 + threadIdx.x (consecutive in memory) */         \
        T* o = out + (u64)u.slice * plane * u.samples + (u64)sample * plane;                           \
        for(u32 j = threadIdx.y; j < TILE; j += blockDim.y){                                           \
            const u32 c = c0 + j, r = r0 + threadIdx.x;                                                \
            if(r < u.rows && c < u.cols) o[(u64)(u.col0 + c) * u.y + u.row0 + r] = tile[threadIdx.x][j]; \
        }                                                                                              \
        __syncthreads();                                                                               \
    }                                                                                                  \
}
PLACEF(ct_place_f8, u8, NOSWAP)
PLACEF(ct_place_f16, u16, bswap16)
PLACEF(ct_place_f32, u32, bswap32)
PLACEF(ct_place_f64, u64, bswap64)

// Whether every stream nvCOMP decompressed came out whole: counts those whose status is not 0 or whose
// size differs from the expected one into *bad, and keeps the first such stream's index in *first
extern "C" __global__ void ct_check(const int* __restrict__ status, const u64* __restrict__ got,
                                    const u64* __restrict__ want, u32 n, u32* bad, u32* first){
    for(u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
        if(status[i] != 0 || got[i] != want[i]){
            atomicAdd(bad, 1u);
            atomicMin(first, i);
        }
    }
}
