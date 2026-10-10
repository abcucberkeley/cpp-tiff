#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <vector>
#include <thread>
#include <omp.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "tiffio.h"
#include "lzwencode.h"
#include "helperfunctions.h"
#include "parallelwritetiff.h"

// zstd one-shot encode; symbols come from the libzstd statically linked into libtiff
extern "C" {
    size_t ZSTD_compress(void* dst, size_t dstCap, const void* src, size_t srcSize, int level);
}
#ifndef COMPRESSION_ZSTD
#define COMPRESSION_ZSTD 50000
#endif
static const int kZstdLevel = 1;

// ---- Tiled output -------------------------------------------------------
// A tile is a sub-rectangle, so unlike a strip it is not contiguous in the
// row-major image and has to be gathered before it can be handed to libtiff.
// libtiff also requires a full tile, so edge tiles are zero padded.
static inline void gatherTile(uint8_t* tileBuf, const uint8_t* slice, int64_t x,
                              int64_t tileX0, int64_t tileY0,
                              int64_t validW, int64_t validH,
                              int64_t tw, int64_t tl, int64_t px) {
    const int64_t inRow = x * px;      // image row stride
    const int64_t outRow = tw * px;    // padded tile row stride
    if(validW < tw || validH < tl) memset(tileBuf, 0, (size_t)(tw*tl*px));
    const uint8_t* s = slice + ((tileY0 * x) + tileX0) * px;
    uint8_t* o = tileBuf;
    for (int64_t r = 0; r < validH; r++) {
        memcpy(o, s, (size_t)(validW * px));
        o += outRow;
        s += inRow;
    }
}

// ---- Column-major (MATLAB) input ----------------------------------------
// MATLAB arrays are column-major (y, x, z), so each strip or tile is
// transposed into row-major order as it is prepared, instead of first copying
// the whole volume. Blocks keep both the strided reads and the writes in cache.
template <typename T>
static void transposeRows(T* out, const T* slice, uint64_t x, uint64_t y, uint64_t r0, uint64_t rows){
    const uint64_t B = 64;
    for(uint64_t rb = 0; rb < rows; rb += B){
        const uint64_t rEnd = std::min(rb + B, rows);
        for(uint64_t cb = 0; cb < x; cb += B){
            const uint64_t cEnd = std::min(cb + B, x);
            for(uint64_t r = rb; r < rEnd; r++){
                T* o = out + r*x;
                const T* in = slice + r0 + r;
                for(uint64_t c = cb; c < cEnd; c++) o[c] = in[c*y];
            }
        }
    }
}

template <typename T>
static void gatherTileTransposed(T* tile, const T* slice, uint64_t y, uint64_t tileX0, uint64_t tileY0,
                                 uint64_t validW, uint64_t validH, uint64_t tw, uint64_t tl){
    if(validW < tw || validH < tl) memset(tile, 0, (size_t)(tw*tl*sizeof(T)));
    for(uint64_t r = 0; r < validH; r++){
        T* o = tile + r*tw;
        const T* in = slice + (tileY0 + r) + tileX0*y;
        for(uint64_t c = 0; c < validW; c++) o[c] = in[c*y];
    }
}

namespace {

// One strip or tile on its way to the file. A producer fills data and then
// publishes size with release order; the writer reads size with acquire order
// before touching data, so it never sees a size before the bytes behind it.
struct Chunk {
    std::atomic<int64_t> size{0};   // 0 = not ready, > 0 = bytes in data, < 0 = failed
    uint8_t* data = nullptr;
};

// Chunk buffers go back to a pool once written and are reused. Producers stay
// within a small window of the writer (see writeJob), so a few dozen buffers
// carry a whole volume instead of every strip faulting in new memory. Every
// buffer has the same size.
class BufferPool {
public:
    explicit BufferPool(size_t bytes) : bytes_(bytes) {}
    ~BufferPool(){ for(uint8_t* b : free_) ::free(b); }
    uint8_t* get(){
        {
            std::lock_guard<std::mutex> g(m_);
            if(!free_.empty()){ uint8_t* b = free_.back(); free_.pop_back(); return b; }
        }
        return (uint8_t*)malloc(bytes_);
    }
    void put(uint8_t* b){
        if(!b) return;
        std::lock_guard<std::mutex> g(m_);
        free_.push_back(b);
    }
private:
    size_t bytes_;
    std::mutex m_;
    std::vector<uint8_t*> free_;
};

struct WriteJob {
    uint64_t x, y, nSlices, bits, bytes, stripSize, tileSize;
    uint64_t tilesAcross, tilesDown, chunksPerSlice;
    const uint8_t* data;     // first slice to write
    bool transpose;          // data is column-major (y, x, z) instead of row-major (z, y, x)
    int compression;         // COMPRESSION_NONE, COMPRESSION_LZW or COMPRESSION_ZSTD
    uint16_t sampleFormat;

    // Uncompressed row-major strips go straight from the caller's buffer to the file.
    bool direct() const { return compression == COMPRESSION_NONE && !transpose && !tileSize; }
    uint64_t sliceBytes() const { return x*y*bytes; }
    // Uncompressed bytes of chunk c (tiles are always full, zero padded).
    uint64_t chunkBytes(uint64_t c) const {
        if(tileSize) return tileSize*tileSize*bytes;
        return std::min<uint64_t>(stripSize, y - c*stripSize)*x*bytes;
    }
    uint64_t maxChunkBytes() const { return tileSize ? tileSize*tileSize*bytes : std::min<uint64_t>(stripSize, y)*x*bytes; }
    // Room for any chunk once prepared: compressed output can exceed its input.
    uint64_t chunkBufferBytes() const { return compression == COMPRESSION_NONE ? maxChunkBytes() : (maxChunkBytes() + 2000*bytes)*2 + 1; }
};

// Uncompressed bytes of chunk c of slice s, row-major.
void fillChunk(const WriteJob& j, uint64_t s, uint64_t c, uint8_t* dst){
    const uint8_t* slice = j.data + s*j.sliceBytes();
    if(j.tileSize){
        const uint64_t ty = c / j.tilesAcross, tx = c % j.tilesAcross;
        const uint64_t tileY0 = ty*j.tileSize, tileX0 = tx*j.tileSize;
        const uint64_t validH = std::min<uint64_t>(j.tileSize, j.y - tileY0), validW = std::min<uint64_t>(j.tileSize, j.x - tileX0);
        if(!j.transpose){
            gatherTile(dst, slice, (int64_t)j.x, (int64_t)tileX0, (int64_t)tileY0, (int64_t)validW, (int64_t)validH,
                       (int64_t)j.tileSize, (int64_t)j.tileSize, (int64_t)j.bytes);
            return;
        }
        switch(j.bytes){
            case 1: gatherTileTransposed((uint8_t*)dst,  (const uint8_t*)slice,  j.y, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
            case 2: gatherTileTransposed((uint16_t*)dst, (const uint16_t*)slice, j.y, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
            case 4: gatherTileTransposed((uint32_t*)dst, (const uint32_t*)slice, j.y, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
            case 8: gatherTileTransposed((uint64_t*)dst, (const uint64_t*)slice, j.y, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
        }
        return;
    }
    const uint64_t r0 = c*j.stripSize, rows = std::min<uint64_t>(j.stripSize, j.y - r0);
    if(!j.transpose){
        memcpy(dst, slice + r0*j.x*j.bytes, (size_t)(rows*j.x*j.bytes));
        return;
    }
    switch(j.bytes){
        case 1: transposeRows((uint8_t*)dst,  (const uint8_t*)slice,  j.x, j.y, r0, rows); break;
        case 2: transposeRows((uint16_t*)dst, (const uint16_t*)slice, j.x, j.y, r0, rows); break;
        case 4: transposeRows((uint32_t*)dst, (const uint32_t*)slice, j.x, j.y, r0, rows); break;
        case 8: transposeRows((uint64_t*)dst, (const uint64_t*)slice, j.x, j.y, r0, rows); break;
    }
}

// Prepare chunk c of slice s and publish it. scratch holds one uncompressed
// chunk for the cases that must be copied before they are compressed. Returns
// false if the chunk could not be made.
bool produceChunk(const WriteJob& j, uint64_t s, uint64_t c, Chunk* chunks, uint8_t* scratch, BufferPool& pool){
    Chunk& ch = chunks[s*j.chunksPerSlice + c];
    const uint64_t raw = j.chunkBytes(c);
    if(j.compression == COMPRESSION_NONE){
        uint8_t* out = pool.get();
        if(!out){ ch.size.store(-1, std::memory_order_release); return false; }
        fillChunk(j, s, c, out);
        ch.data = out;
        ch.size.store((int64_t)raw, std::memory_order_release);
        return true;
    }
    const uint8_t* src = scratch;
    if(!j.tileSize && !j.transpose) src = j.data + s*j.sliceBytes() + c*j.stripSize*j.x*j.bytes;   // compress in place
    else fillChunk(j, s, c, scratch);
    const uint64_t bound = j.chunkBufferBytes();
    uint8_t* out = pool.get();
    if(!out){ ch.size.store(-1, std::memory_order_release); return false; }
    // A zstd error comes back as a huge value, so anything past the bound is a failure.
    const uint64_t n = (j.compression == COMPRESSION_ZSTD)
        ? (uint64_t)ZSTD_compress(out, (size_t)bound, src, (size_t)raw, kZstdLevel)
        : lzwEncode(const_cast<uint8_t*>(src), out, (tmsize_t)raw);
    if(!n || n > bound){ pool.put(out); ch.size.store(-1, std::memory_order_release); return false; }
    ch.data = out;
    ch.size.store((int64_t)n, std::memory_order_release);
    return true;
}

// Write every slice in order: tags, its chunks (waiting for producers unless
// produceInline, where this thread makes each chunk just before writing it),
// then its directory. writerChunk tells producers how far the writer is.
// Returns 0 on success.
uint8_t writeSlices(const WriteJob& j, const char* fileName, const char* mode, Chunk* chunks,
                    std::atomic<int64_t>& writerChunk, std::atomic<bool>& failed,
                    bool produceInline, uint8_t* scratch, BufferPool* pool){
    TIFF* tif = NULL;
    if(!strcmp(mode,"w")) tif = TIFFOpen(fileName, "w8");
    else if(!strcmp(mode,"a")) tif = TIFFOpen(fileName, "a8");
    else{
        printf("Error: mode \"%s\" is not supported. Use w or a for mode type\n", mode);
        failed = true;
        return 1;
    }
    if(!tif){
        printf("Error: File \"%s\" cannot be opened\n",fileName);
        failed = true;
        return 1;
    }

    uint8_t err = 0;
    const char* what = "";
    // Explicit type tag: caller-provided format, or derive from bits for legacy callers.
    const uint16_t sf = j.sampleFormat ? j.sampleFormat : (j.bits >= 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
    for(uint64_t s = 0; s < j.nSlices && !err; s++){
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (uint32_t)j.x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (uint32_t)j.y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (uint16_t)j.bits);
        if(j.tileSize){
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, (uint32_t)j.tileSize);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, (uint32_t)j.tileSize);
        }
        else TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, (uint32_t)j.stripSize);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, j.compression);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, 1);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sf);

        for(uint64_t c = 0; c < j.chunksPerSlice; c++){
            const uint8_t* p;
            int64_t n;
            Chunk* ch = NULL;
            if(j.direct()){
                p = j.data + s*j.sliceBytes() + c*j.stripSize*j.x*j.bytes;
                n = (int64_t)j.chunkBytes(c);
            }
            else{
                writerChunk.store((int64_t)(s*j.chunksPerSlice + c), std::memory_order_release);
                if(produceInline && !produceChunk(j, s, c, chunks, scratch, *pool)){
                    err = 1; what = "could not compress or prepare a strip (out of memory?)";
                    break;
                }
                ch = &chunks[s*j.chunksPerSlice + c];
                while((n = ch->size.load(std::memory_order_acquire)) == 0){
                    if(failed.load()) break;
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                if(n <= 0){ err = 1; what = "could not compress or prepare a strip (out of memory?)"; break; }
                p = ch->data;
            }
            const tmsize_t w = j.tileSize ? TIFFWriteRawTile(tif, (uint32_t)c, (void*)p, (tmsize_t)n)
                                          : TIFFWriteRawStrip(tif, (uint32_t)c, (void*)p, (tmsize_t)n);
            if(ch){ pool->put(ch->data); ch->data = NULL; }
            if(w != (tmsize_t)n){ err = 1; what = "a strip could not be written (disk full or I/O error?)"; break; }
        }
        if(!err && !TIFFWriteDirectory(tif)){ err = 1; what = "a directory could not be written (disk full or I/O error?)"; }
    }
    if(!err && !TIFFFlush(tif)){ err = 1; what = "the file could not be flushed (disk full or I/O error?)"; }
#ifndef _WIN32
    // NFS can accept every write and only report a failure (such as a full
    // quota) when the file is closed, and TIFFClose discards that. Closing a
    // duplicate descriptor first makes the client flush and report it here.
    if(!err){
        const int fd = dup(TIFFFileno(tif));
        if(fd < 0 || close(fd) != 0){ err = 1; what = "the file could not be closed (disk full or I/O error?)"; }
    }
#endif
    TIFFClose(tif);
    if(err){
        printf("Error: writing \"%s\" failed: %s. The file is incomplete.\n", fileName, what);
        failed = true;
    }
    return err;
}

// Write a whole job. Strips and tiles are prepared (transposed, gathered and/or
// compressed) by `threads` threads, handed out one at a time in file order,
// while a separate writer thread writes them in order as they become ready.
// Handing each thread a contiguous block of slices instead left the writer
// waiting for the first block and then writing nearly everything after
// compression had finished, and compressed a single 2D image on one thread.
// Producers stay within a window ahead of the writer so a slow disk cannot
// make them buffer the whole volume.
uint8_t writeJob(const WriteJob& j, const char* fileName, const char* mode, int threads){
    std::unique_ptr<Chunk[]> chunks(j.direct() ? NULL : new (std::nothrow) Chunk[j.nSlices*j.chunksPerSlice + 1]);
    if(!j.direct() && !chunks){
        printf("Error: out of memory writing \"%s\"\n", fileName);
        return 1;
    }
    const bool needScratch = j.compression != COMPRESSION_NONE && (j.tileSize || j.transpose);
    std::atomic<int64_t> writerChunk{0};
    std::atomic<bool> failed{false};
    BufferPool pool((size_t)j.chunkBufferBytes());
    uint8_t err = 0;

    if(threads <= 1){
        std::unique_ptr<uint8_t[]> scratch(needScratch ? new (std::nothrow) uint8_t[j.maxChunkBytes()] : NULL);
        if(needScratch && !scratch){
            printf("Error: out of memory writing \"%s\"\n", fileName);
            return 1;
        }
        err = writeSlices(j, fileName, mode, chunks.get(), writerChunk, failed, true, scratch.get(), &pool);
    }
    else if(j.direct()){
        // Nothing needs preparing, but a new thread still does the writing, as
        // in v1.6.2. Under Linux's automatic NUMA balancing, a thread that has
        // been running a while, like the caller, keeps marking the process's
        // memory so that the next touch of each page faults and may move the
        // page. Writing from the caller took over a million such faults and
        // half a million page moves for a 5 GB volume, 30-90% slower than
        // v1.6.2 on two-socket machines. A new thread does not start marking
        // during a write, and the waiting caller marks nothing.
        err = std::async(std::launch::async, writeSlices, std::cref(j), fileName, mode, chunks.get(),
                         std::ref(writerChunk), std::ref(failed), true, (uint8_t*)NULL, &pool).get();
    }
    else{
        std::future<uint8_t> writer = std::async(std::launch::async, writeSlices, std::cref(j), fileName, mode,
                                                 chunks.get(), std::ref(writerChunk), std::ref(failed), false, (uint8_t*)NULL, &pool);
        // Producers stay within 64 MB (and at least two chunks per thread) of
        // the writer: enough to keep it busy, while a few dozen buffers carry
        // the whole volume. Uncompressed strips are prepared much faster than
        // they are written, so with a 1 GB window every call faulted in and freed
        // up to 1 GB of new buffers, which made MATLAB-layout writes slower than
        // v1.6.2 on some machines.
        const uint64_t kInFlightBytes = (uint64_t)64 << 20;
        const int64_t window = std::max<int64_t>(2*(int64_t)threads, (int64_t)(kInFlightBytes / std::max<uint64_t>(1, j.maxChunkBytes())));
        const int64_t nChunks = (int64_t)(j.nSlices*j.chunksPerSlice);
        #pragma omp parallel num_threads(threads)
        {
            std::unique_ptr<uint8_t[]> scratch(needScratch ? new (std::nothrow) uint8_t[j.maxChunkBytes()] : NULL);
            if(needScratch && !scratch) failed = true;
            #pragma omp for schedule(dynamic, 1)
            for(int64_t k = 0; k < nChunks; k++){
                // Far ahead of the writer: back off to 1 ms, well under the time the window takes to write.
                for(int us = 50; !failed.load() && k >= writerChunk.load(std::memory_order_acquire) + window; us = std::min(2*us, 1000))
                    std::this_thread::sleep_for(std::chrono::microseconds(us));
                if(failed.load()) continue;
                if(!produceChunk(j, (uint64_t)k / j.chunksPerSlice, (uint64_t)k % j.chunksPerSlice, chunks.get(), scratch.get(), pool)) failed = true;
            }
        }
        err = writer.get();
        if(failed.load() && !err){
            printf("Error: writing \"%s\" failed: could not compress or prepare a strip (out of memory?). The file is incomplete.\n", fileName);
            err = 1;
        }
    }
    // Only a failure leaves chunks that were made but never written.
    if(chunks) for(uint64_t i = 0; i < j.nSlices*j.chunksPerSlice; i++) pool.put(chunks[i].data);
    return err;
}

// Fill in a job from the public functions' arguments. Slices [startSlice, z)
// are written, taken from the start of data.
uint8_t makeJob(WriteJob& j, uint64_t x, uint64_t y, uint64_t z, const void* data, uint64_t bits, uint64_t startSlice,
                uint64_t stripSize, bool transpose, const std::string& compression, uint16_t sampleFormat, uint64_t tileSize){
    if(bits != 8 && bits != 16 && bits != 32 && bits != 64){
        printf("Error: %llu-bit samples are not supported\n", (unsigned long long)bits);
        return 1;
    }
    // The TIFF spec requires tile dimensions to be a multiple of 16.
    if(tileSize && tileSize % 16){
        printf("Error: tile size %llu is not a multiple of 16\n",(unsigned long long)tileSize);
        return 1;
    }
    if(!tileSize && !stripSize){
        printf("Error: strip size must be at least 1 row\n");
        return 1;
    }
    j.x = x; j.y = y; j.bits = bits; j.bytes = bits/8;
    j.nSlices = z > startSlice ? z - startSlice : 0;
    j.stripSize = stripSize; j.tileSize = tileSize;
    j.tilesAcross = tileSize ? (x + tileSize - 1)/tileSize : 0;
    j.tilesDown = tileSize ? (y + tileSize - 1)/tileSize : 0;
    j.chunksPerSlice = tileSize ? j.tilesAcross*j.tilesDown : (y + stripSize - 1)/stripSize;
    j.data = (const uint8_t*)data;
    j.transpose = transpose;
    j.compression = compression == "zstd" ? COMPRESSION_ZSTD
                  : (compression != "none" && !compression.empty()) ? COMPRESSION_LZW : COMPRESSION_NONE;
    j.sampleFormat = sampleFormat;
    return 0;
}

}

// Single-threaded write of row-major data.
uint8_t writeTiffSingle(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    (void)tiffOld; (void)transpose;   // tiff is already row-major
    WriteJob j;
    if(makeJob(j, x, y, z, tiff, bits, startSlice, stripSize, false, compression, sampleFormat, tileSize)) return 1;
    return writeJob(j, fileName, mode, 1);
}

// Parallel write of row-major data.
uint8_t writeTiffParallel(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    (void)tiffOld; (void)transpose;   // tiff is already row-major
    WriteJob j;
    if(makeJob(j, x, y, z, tiff, bits, startSlice, stripSize, false, compression, sampleFormat, tileSize)) return 1;
    return writeJob(j, fileName, mode, omp_get_max_threads());
}

// data is row-major (z, y, x), or column-major MATLAB order (y, x, z) when transpose.
uint8_t writeTiffParallelWrapper(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* data, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    WriteJob j;
    if(makeJob(j, x, y, z, data, bits, startSlice, stripSize, transpose, compression, sampleFormat, tileSize)) return 1;
    return writeJob(j, fileName, mode, omp_get_max_threads());
}

uint8_t writeTiffParallelHelper(const char* fileName, const void* tiffOld, uint64_t bits, const char* mode, uint64_t x, uint64_t y, uint64_t z, uint64_t startSlice, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize)
{
	// Check if folder exists, if not then make it (recursive if needed)
	char* folderName = strdup(fileName);
	char *lastSlash = NULL;
    #ifdef _WIN32
    lastSlash = strrchr(folderName, '\\');
    #else
    lastSlash = strrchr(folderName, '/');
    #endif
	if(lastSlash){
		*lastSlash = '\0';
		FILE* f = fopen(folderName,"r");
		if(f){
			fclose(f);
		}
		else{
			mkdirRecursive(folderName);
		}
	}
	free(folderName);

	TIFFSetWarningHandler(DummyHandler);

	if(!mode) mode = "w";

	// For 2D images MATLAB passes in the 3rd dim as 0 so we set it to 1;
	if(!z){
		z = 1;
	}

	uint64_t stripSize = 512;

    return writeTiffParallelWrapper(x, y, z, fileName, tiffOld, bits, startSlice, stripSize, mode, transpose, compression, sampleFormat, tileSize);
}
