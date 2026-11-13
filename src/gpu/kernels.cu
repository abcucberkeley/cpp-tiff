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

#define LZW_CLEAR 256u
#define LZW_EOI 257u
#define LZW_FIRST 258u
#define LZW_FULL 0xFFFFFFFFu

// TIFF's LZW decoded on the GPU (nvCOMP has no LZW; CPPTIFF_GPU_LZW=0 decodes it on the host instead), a strip a warp,
// a segment (the codes between two Clear codes) at a time by all of the warp's lanes: in a segment every code's width
// follows from its place alone (9 bits for codes 0 to 253, then 10, 11 and 12 as the table grows by one entry a code),
// so the lanes read 32 codes at a time, and the segment ends at the first Clear, end-of-information or bad code. Each
// code's string is an earlier code's string and one byte more (or a byte), so its length and first byte come from
// pointer jumping along those earlier codes (at most 12 rounds); a warp scan of the lengths places the strings, and
// each lane writes its codes' strings from their last bytes back. Only the segments of a strip (a few hundred in a
// strip of 2 MB) follow one another. What the CPU decoder (lzwdecode.cpp) would hand to libtiff (old-style codes, bad
// or too little data), and tables that fill without a Clear code (libtiff's encoder clears them at 4,094 codes), get
// status 1, and their read is read on the CPU. Blocks of one warp, 27 KB of shared memory each.
// Measured, decoding on the host against this: an 8 GB stack of 2 MB strips read warm in 1.92 s against 0.62 s on an
// A100 node with 10 cores and in 0.53 s against 0.26 s on a B200 node with 16, 2 GB of 256 KB strips with the predictor
// in 0.70 s against 0.12 s and 0.33 s against 0.046 s, and cold from VAST 4.8 s against 3.0 s and 2.8 s against 2.0 s,
// with a fifth of the CPU time. Reading each strip's codes in order with one lane (the table in shared memory, the
// warp writing the strings at each Clear code) took 1.72 s and 0.98 s for the 8 GB stack, slower than the B200 node's
// host, and a thread a strip with the table in GPU memory took 26.6 s on the A100 (each code a chain of dependent
// reads of GPU memory).
#define LZW_CODES 3840u

__device__ __forceinline__ u32 lzwBit(u32 j){
    u32 b = 9u*min(j, 254u);
    if(j > 254u) b += 10u*min(j - 254u, 512u);
    if(j > 766u) b += 11u*min(j - 766u, 1024u);
    if(j > 1790u) b += 12u*(j - 1790u);
    return b;
}
__device__ __forceinline__ u32 lzwWidth(u32 j){ return j < 254u ? 9u : j < 766u ? 10u : j < 1790u ? 11u : 12u; }
// The w-bit code at bit b of in (n bytes; the caller has checked that it fits)
__device__ __forceinline__ u32 lzwCode(const u8* __restrict__ in, u64 n, u64 b, u32 w){
    const u64 i = b >> 3;
    u32 v = (u32)__ldg(in + i) << 16;
    if(i + 1 < n) v |= (u32)__ldg(in + i + 1) << 8;
    if(i + 2 < n) v |= (u32)__ldg(in + i + 2);
    return (v >> (24u - (u32)(b & 7) - w)) & ((1u << w) - 1);
}

extern "C" __global__ void __launch_bounds__(32) ct_lzw(const u64* __restrict__ inPtr, const u64* __restrict__ inBytes,
                                                         const u64* __restrict__ outBytes, const u64* __restrict__ outPtr,
                                                         u64* got, int* status, u32 n){
    __shared__ u16 codes[LZW_CODES];
    __shared__ u8 firstOf[LZW_CODES];     // each code's string's first byte
    __shared__ u32 jump[LZW_CODES];       // edges to an earlier code << 16 | that code (itself for a byte)
    const u32 lane = threadIdx.x;
    for(u32 si = blockIdx.x; si < n; si += gridDim.x){
        const u8* in = (const u8*)inPtr[si];
        const u64 inN = inBytes[si], occ64 = outBytes[si], inBits = inN*8;
        u8* out = (u8*)outPtr[si];
        const u32 occ = (u32)occ64;
        int state = 0;                     // 0 reading, 1 done, 2 refused
        if(inN < 2 || (in[0] == 0 && (in[1] & 1)) || occ64 == 0 || occ64 > 0xFFFFFFFFull) state = 2;
        u64 S = 0;                         // the bit the next code starts at
        u32 P = 0;                         // the bytes written
        if(!state){
            if(lzwCode(in, inN, 0, 9) != LZW_CLEAR) state = 2;
            else S = 9;
        }
        while(!state){
            // (Clear codes again, each of 9 bits)
            while(S + 9 <= inBits && lzwCode(in, inN, S, 9) == LZW_CLEAR) S += 9;
            // the segment's codes, until the code that ends it (J): kind 1 a Clear, 2 anything else
            u32 J = 0, kind = 2;
            for(u32 r = 0; r < LZW_CODES; r += 32){
                const u32 j = r + lane, w = lzwWidth(j);
                const u64 b = S + lzwBit(j);
                u32 k = 0;
                if(b + w > inBits) k = 2;
                else{
                    const u32 c = lzwCode(in, inN, b, w);
                    if(c == LZW_CLEAR) k = 1;
                    else if(c == LZW_EOI || j >= LZW_CODES - 1 || (j == 0 ? c >= 256u : (c >= LZW_FIRST && c > LZW_FIRST + j - 1))) k = 2;
                    else codes[j] = (u16)c;
                }
                const u32 stops = __ballot_sync(LZW_FULL, k != 0);
                if(stops){
                    const u32 first = __ffs(stops) - 1;
                    J = r + first;
                    kind = __shfl_sync(LZW_FULL, k, first);
                    break;
                }
            }
            __syncwarp();
            // each code's length and first byte: jump along the earlier codes whose strings it extends
            for(u32 j = lane; j < J; j += 32){
                const u32 c = codes[j];
                jump[j] = c < 256u ? j : (1u << 16) | (c - LZW_FIRST);
            }
            __syncwarp();
            // (in place: an entry read while another lane changes it is its old or its new pair, both right)
            volatile u32* jv = jump;
            for(int round = 0; round < 13; round++){
                bool changed = false;
                for(u32 j = lane; j < J; j += 32){
                    const u32 v = jv[j], k = v & 0xFFFFu, wv = jv[k];
                    if(wv >> 16){
                        jv[j] = ((v >> 16) + (wv >> 16)) << 16 | (wv & 0xFFFFu);
                        changed = true;
                    }
                }
                __syncwarp();
                if(!__any_sync(LZW_FULL, changed)) break;
            }
            for(u32 j = lane; j < J; j += 32) firstOf[j] = (u8)codes[jump[j] & 0xFFFFu];
            __syncwarp();
            // the strings, 32 codes at a time, placed by a scan of their lengths
            u32 base = P;
            for(u32 r = 0; r < J && base < occ; r += 32){
                const u32 k = r + lane;
                const u32 len = k < J ? (jump[k] >> 16) + 1 : 0;
                u32 inc = len;
                for(u32 d = 1; d < 32; d <<= 1){
                    const u32 v = __shfl_up_sync(LZW_FULL, inc, d);
                    if(lane >= d) inc += v;
                }
                const u32 p = base + inc - len;
                base += __shfl_sync(LZW_FULL, inc, 31);
                if(k < J && p < occ){
                    u32 c = codes[k], j = len;
                    while(c >= 256u){
                        const u32 e = c - 257u;          // (the code that made entry c: its string, then that code's first byte)
                        j--;
                        if(p + j < occ) out[p + j] = firstOf[e];
                        c = codes[e - 1];
                    }
                    out[p] = (u8)c;
                }
            }
            __syncwarp();
            if(base >= occ) state = 1;
            else if(kind == 1){
                P = base;
                S += lzwBit(J) + lzwWidth(J);
            }
            else state = 2;
        }
        if(lane == 0){
            status[si] = state == 1 ? 0 : 1;
            got[si] = state == 1 ? occ64 : 0;
        }
        __syncwarp();
    }
}

// ---- GPU writes (gpuwritetiff.cpp)

// MATLAB's column-major (y, x, c, z) data to row-major strips (rows, x, c), as the writer stores them: the transpose
// of ct_place_f, through 32 x 32 tiles in shared memory, so that reads run along the input's columns and writes
// along the strips' rows. strips: per strip its slice, first row, rows and the element its output starts at in out.
// Block (s, t): strip s's tiles t, t + gridDim.y, ... (tiles of each sample in turn), 32 x 8 threads.
#define GATHERF(NAME, T)                                                                               \
extern "C" __global__ void NAME(const u64* __restrict__ strips, u32 n, const T* __restrict__ in, T* out, \
                                u32 x, u32 y, u32 spp){                                                \
    __shared__ T tile[TILE][TILE + 1];                                                                 \
    const u32 si = blockIdx.x;                                                                         \
    if(si >= n) return;                                                                                \
    const u64 slice = strips[4*si], r0 = strips[4*si + 1], o = strips[4*si + 3];                       \
    const u32 rows = (u32)strips[4*si + 2];                                                            \
    const u64 plane = (u64)x * y;                                                                      \
    const u32 tilesAcross = (x + TILE - 1) / TILE, tilesDown = (rows + TILE - 1) / TILE;               \
    const u32 tiles = tilesAcross * tilesDown * spp;                                                   \
    for(u32 t = blockIdx.y; t < tiles; t += gridDim.y){                                                \
        const u32 c = t / (tilesAcross * tilesDown), rest = t % (tilesAcross * tilesDown);             \
        const u32 c0 = (rest % tilesAcross) * TILE, rr0 = (rest / tilesAcross) * TILE;                 \
        const T* src = in + slice * plane * spp + (u64)c * plane;                                      \
        /* read: columns c0 + j, rows rr0 + threadIdx.x (consecutive in memory) */                     \
        for(u32 j = threadIdx.y; j < TILE; j += blockDim.y){                                           \
            const u32 col = c0 + j, r = rr0 + threadIdx.x;                                             \
            if(col < x && r < rows) tile[j][threadIdx.x] = src[(u64)col * y + r0 + r];                 \
        }                                                                                              \
        __syncthreads();                                                                               \
        /* write: rows rr0 + j, columns c0 + threadIdx.x */                                            \
        for(u32 j = threadIdx.y; j < TILE; j += blockDim.y){                                           \
            const u32 r = rr0 + j, col = c0 + threadIdx.x;                                             \
            if(col < x && r < rows) out[o + ((u64)r * x + col) * spp + c] = tile[threadIdx.x][j];      \
        }                                                                                              \
        __syncthreads();                                                                               \
    }                                                                                                  \
}
GATHERF(ct_gather_f8, u8)
GATHERF(ct_gather_f16, u16)
GATHERF(ct_gather_f32, u32)
GATHERF(ct_gather_f64, u64)

// Where each compressed strip goes in the packed buffer: offsets[i] the sum of the sizes before it, each rounded up to
// 16 bytes, and *total the sum of them all. One block: each thread sums a run of the sizes, the block scans the
// runs' sums, and each thread then writes its run's offsets.
extern "C" __global__ void ct_scan(const u64* __restrict__ sizes, u64* offsets, u32 n, u64* total){
    __shared__ u64 sums[1024];
    const u32 t = threadIdx.x, T = blockDim.x;
    const u32 per = (n + T - 1) / T, b = t * per, e = b + per < n ? b + per : n;
    u64 s = 0;
    for(u32 i = b; i < e; i++) s += (sizes[i] + 15) & ~15ull;
    sums[t] = s;
    __syncthreads();
    for(u32 d = 1; d < T; d <<= 1){
        const u64 v = t >= d ? sums[t - d] : 0;
        __syncthreads();
        sums[t] += v;
        __syncthreads();
    }
    u64 run = t ? sums[t - 1] : 0;
    for(u32 i = b; i < e; i++){ offsets[i] = run; run += (sizes[i] + 15) & ~15ull; }
    if(t == T - 1) *total = sums[T - 1];
}

// Strips copied to their places in dst (the compressed strips packed together, or strips not aligned as nvCOMP
// needs them copied to aligned places): a block a strip at a time, 16 bytes a thread where both ends are 16-byte
// aligned, a byte a thread otherwise
extern "C" __global__ void ct_pack(const u64* __restrict__ src, const u64* __restrict__ sizes, const u64* __restrict__ offsets,
                                   u32 n, u8* dst){
    for(u32 i = blockIdx.x; i < n; i += gridDim.x){
        const u8* s = (const u8*)src[i];
        u8* d = dst + offsets[i];
        const u64 m = sizes[i];
        const u64 words = (((u64)s | (u64)d) & 15) ? 0 : m / 16;
        for(u64 k = threadIdx.x; k < words; k += blockDim.x) ((uint4*)d)[k] = ((const uint4*)s)[k];
        for(u64 k = words * 16 + threadIdx.x; k < m; k += blockDim.x) d[k] = s[k];
    }
}
