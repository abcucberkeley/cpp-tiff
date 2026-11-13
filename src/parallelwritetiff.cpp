#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>
#include <thread>
#include <omp.h>
#ifndef _WIN32
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#endif
#include "tiffio.h"
#include "lzwencode.h"
#include "helperfunctions.h"
#include "parallelwritetiff.h"
#include "pathutil.h"
#include "tiffdirs.h"

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

// Color versions of the two above. A MATLAB color array is (y, x, c, z), so
// each color is its own column-major plane x*y apart, while a TIFF row holds
// every pixel's samples together (x, c). Grayscale keeps the functions above.
template <typename T>
static void transposeRowsColor(T* out, const T* slice, uint64_t x, uint64_t y, uint64_t samples, uint64_t r0, uint64_t rows){
    const uint64_t B = 64, plane = x*y;
    for(uint64_t rb = 0; rb < rows; rb += B){
        const uint64_t rEnd = std::min(rb + B, rows);
        for(uint64_t cb = 0; cb < x; cb += B){
            const uint64_t cEnd = std::min(cb + B, x);
            for(uint64_t r = rb; r < rEnd; r++){
                T* o = out + r*x*samples;
                const T* in = slice + r0 + r;
                for(uint64_t s = 0; s < samples; s++)
                    for(uint64_t c = cb; c < cEnd; c++) o[c*samples + s] = in[c*y + s*plane];
            }
        }
    }
}

template <typename T>
static void gatherTileTransposedColor(T* tile, const T* slice, uint64_t x, uint64_t y, uint64_t samples,
                                      uint64_t tileX0, uint64_t tileY0, uint64_t validW, uint64_t validH, uint64_t tw, uint64_t tl){
    if(validW < tw || validH < tl) memset(tile, 0, (size_t)(tw*tl*samples*sizeof(T)));
    const uint64_t plane = x*y;
    for(uint64_t r = 0; r < validH; r++){
        T* o = tile + r*tw*samples;
        const T* in = slice + (tileY0 + r) + tileX0*y;
        for(uint64_t s = 0; s < samples; s++)
            for(uint64_t c = 0; c < validW; c++) o[c*samples + s] = in[c*y + s*plane];
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
    uint64_t x, y, nSlices, bits, bytes, samples, stripSize, tileSize;
    uint64_t tilesAcross, tilesDown, chunksPerSlice;
    const uint8_t* data;     // first slice to write
    bool transpose;          // data is column-major (y, x[, c], z) instead of row-major (z, y, x[, c])
    int compression;         // COMPRESSION_NONE, COMPRESSION_LZW or COMPRESSION_ZSTD
    uint16_t sampleFormat;

    // Uncompressed row-major strips go straight from the caller's buffer to the file.
    bool direct() const { return compression == COMPRESSION_NONE && !transpose && !tileSize; }
    // A pixel's samples (1 gray, 3 RGB or 4 RGBA) are stored together.
    uint64_t pixelBytes() const { return bytes*samples; }
    uint64_t sliceBytes() const { return x*y*pixelBytes(); }
    // Where strip c starts in a row-major slice.
    uint64_t stripOffset(uint64_t c) const { return c*stripSize*x*pixelBytes(); }
    // Uncompressed bytes of chunk c (tiles are always full, zero padded).
    uint64_t chunkBytes(uint64_t c) const {
        if(tileSize) return tileSize*tileSize*pixelBytes();
        return std::min<uint64_t>(stripSize, y - c*stripSize)*x*pixelBytes();
    }
    uint64_t maxChunkBytes() const { return tileSize ? tileSize*tileSize*pixelBytes() : std::min<uint64_t>(stripSize, y)*x*pixelBytes(); }
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
                       (int64_t)j.tileSize, (int64_t)j.tileSize, (int64_t)j.pixelBytes());
            return;
        }
        if(j.samples > 1){
            switch(j.bytes){
                case 1: gatherTileTransposedColor((uint8_t*)dst,  (const uint8_t*)slice,  j.x, j.y, j.samples, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
                case 2: gatherTileTransposedColor((uint16_t*)dst, (const uint16_t*)slice, j.x, j.y, j.samples, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
                case 4: gatherTileTransposedColor((uint32_t*)dst, (const uint32_t*)slice, j.x, j.y, j.samples, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
                case 8: gatherTileTransposedColor((uint64_t*)dst, (const uint64_t*)slice, j.x, j.y, j.samples, tileX0, tileY0, validW, validH, j.tileSize, j.tileSize); break;
            }
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
        memcpy(dst, slice + j.stripOffset(c), (size_t)(rows*j.x*j.pixelBytes()));
        return;
    }
    if(j.samples > 1){
        switch(j.bytes){
            case 1: transposeRowsColor((uint8_t*)dst,  (const uint8_t*)slice,  j.x, j.y, j.samples, r0, rows); break;
            case 2: transposeRowsColor((uint16_t*)dst, (const uint16_t*)slice, j.x, j.y, j.samples, r0, rows); break;
            case 4: transposeRowsColor((uint32_t*)dst, (const uint32_t*)slice, j.x, j.y, j.samples, r0, rows); break;
            case 8: transposeRowsColor((uint64_t*)dst, (const uint64_t*)slice, j.x, j.y, j.samples, r0, rows); break;
        }
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
    if(!j.tileSize && !j.transpose) src = j.data + s*j.sliceBytes() + j.stripOffset(c);   // compress in place
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

#ifndef _WIN32
// File I/O for libtiff that batches writes (Linux and macOS). libtiff writes
// each strip with a seek and a write, and each directory with about ten small
// seeks, reads and writes (it reads the previous directory back to link the
// new one to it): about 25 calls per slice of an uncompressed stack. Here
// writes are queued in file order and sent with pwritev (writev elsewhere):
// small ones are copied, large ones from the caller's own array (stable) are
// queued by address, and large ones from buffers that are about to be reused
// are sent at once together with the queue. Seeks, and reads of queued data or
// of the small writes sent last, are answered from memory. Uncompressed writes of 500-slice stacks became
// 1-4% faster to RAM and local disk, with identical files.
#ifdef __linux__
// Whether the filesystem holding device dev is NFS, from /proc/self/mountinfo
// (read once per filesystem) rather than fstatfs: on NFS fstatfs asks the
// server (0.07-0.4 ms on the cluster's NFS, 1.3 ms over SSHFS), which every
// compressed file written would pay.
static bool onNfs(dev_t dev){
    // A filesystem on a disk (a block device) is not NFS, and needs no lookup:
    // NFS, like tmpfs and FUSE, has an anonymous device number (major 0).
    if(major(dev) != 0) return false;
    static std::mutex m;
    static std::vector<std::pair<dev_t, bool> > known;
    std::lock_guard<std::mutex> g(m);
    for(const auto& k : known) if(k.first == dev) return k.second;
    bool nfs = false;
    if(FILE* f = fopen("/proc/self/mountinfo", "r")){
        char line[4096], type[64];
        unsigned maj, mnr;
        while(fgets(line, sizeof line, f)){
            // "id parent major:minor root mountpoint options [optional...] - fstype source ..."
            const char* sep = strstr(line, " - ");
            if(!sep || sscanf(line, "%*s %*s %u:%u", &maj, &mnr) != 2 || sscanf(sep + 3, "%63s", type) != 1) continue;
            if(makedev(maj, mnr) == dev){
                nfs = !strcmp(type, "nfs") || !strcmp(type, "nfs4");
                break;
            }
        }
        fclose(f);
    }
    known.push_back(std::make_pair(dev, nfs));
    return nfs;
}
#endif

class BatchedFile {
public:
    // compressed: the data comes from compressing, so on NFS its sending is
    // started early (see open).
    BatchedFile(bool stable, bool compressed) : stable_(stable), compressed_(compressed) {}
    ~BatchedFile(){ if(fd_ >= 0) ::close(fd_); }
    bool open(const char* fileName, bool append){
        fd_ = ::open(fileName, O_RDWR | O_CREAT | (append ? 0 : O_TRUNC), 0666);
        struct stat st;
        if(fd_ < 0 || fstat(fd_, &st) != 0) return false;
        base_ = end_ = (uint64_t)st.st_size;
#ifdef __linux__
        // NFS sends what was written to the server at close, so a write would
        // compress everything first and then wait for the whole transfer.
        // Starting to send each part once it is written overlaps the two. Not
        // on local disks: nothing waits for them at close, and starting their
        // writeback early could make the write wait for the disk instead. Not
        // for uncompressed data either: it is written as fast as it is copied,
        // so there is nothing to overlap, and on VAST it was 5% slower.
        // Whether the file is on NFS is looked up only once a file gets that far
        // (see send), so small files never pay for the lookup.
        writebackStep_ = compressed_ ? kWritebackStep : 0;
        writebackFrom_ = base_;
        dev_ = st.st_dev;
#endif
        return true;
    }
    bool failed() const { return failed_; }

    tmsize_t write(const void* data, tmsize_t size){
        if(size <= 0) return size;
        const uint8_t* src = static_cast<const uint8_t*>(data);
        uint64_t n = (uint64_t)size;
        // Into data already sent: libtiff links each directory to the one
        // before it, which may have gone out with the last batch.
        if(pos_ < base_){
            const uint64_t k = std::min<uint64_t>(n, base_ - pos_);
            if(!sendAt(src, k, pos_)) return -1;
            patchSent(pos_, src, k);
            pos_ += k; src += k; n -= k;
        }
        // Into the queue (directory entries written before the data they point to).
        while(n && pos_ < end_){
            const Seg& sg = segFor(pos_);
            if(sg.ext){   // libtiff does not rewrite strip data, but stay correct if it does
                if(!flush() || !sendAt(src, n, pos_)) return -1;
                pos_ += n;
                return size;
            }
            const uint64_t k = std::min<uint64_t>(n, sg.off + sg.len - pos_);
            memcpy(&small_[sg.at + (pos_ - sg.off)], src, (size_t)k);
            pos_ += k; src += k; n -= k;
        }
        if(!n) return size;
        // Past the end, after a gap that libtiff fills in later.
        if(pos_ > end_) appendSmall(NULL, pos_ - end_);
        if(n >= kLarge && stable_){
            segs_.push_back(Seg{end_, n, src, 0});
            end_ += n;
            queued_ += n;
        }
        else if(n >= kLarge){
            Seg sg{end_, n, src, 0};
            if(!send(&sg)) return -1;
        }
        else appendSmall(src, n);
        pos_ += n;
        if((queued_ >= kBatchBytes || segs_.size() >= kMaxSegs) && !flush()) return -1;
        return size;
    }

    tmsize_t read(void* data, tmsize_t size){
        uint8_t* dst = static_cast<uint8_t*>(data);
        uint64_t got = 0;
        while(got < (uint64_t)size && pos_ + got < end_){
            const uint64_t p = pos_ + got, want = (uint64_t)size - got;
            if(p >= base_){
                const Seg& sg = segFor(p);
                const uint64_t k = std::min<uint64_t>(want, sg.off + sg.len - p);
                memcpy(dst + got, sg.ext ? sg.ext + (p - sg.off) : &small_[sg.at + (p - sg.off)], (size_t)k);
                got += k;
                continue;
            }
            const Seg* r = sentFor(p);
            if(r){
                const uint64_t k = std::min<uint64_t>(want, r->off + r->len - p);
                memcpy(dst + got, &sent_[r->at + (p - r->off)], (size_t)k);
                got += k;
                continue;
            }
            const ssize_t k = pread(fd_, dst + got, (size_t)std::min<uint64_t>(want, base_ - p), (off_t)p);
            if(k <= 0) break;
            got += (uint64_t)k;
        }
        pos_ += got;
        return (tmsize_t)got;
    }

    toff_t seek(toff_t off, int whence){
        pos_ = whence == SEEK_SET ? off : whence == SEEK_CUR ? pos_ + off : end_ + off;
        return pos_;
    }
    toff_t size() const { return end_; }

    // Sends the queue and closes the file. NFS can accept every write and only
    // report a failure (such as a full quota) at close, so that counts too.
    int close(){
        const bool ok = flush();
        const int r = fd_ >= 0 ? ::close(fd_) : 0;
        fd_ = -1;
        if(!ok || r != 0) failed_ = true;
        return failed_ ? -1 : 0;
    }

    bool flush(){ return send(NULL); }

private:
    // A run of bytes at file offset off: borrowed from the caller (ext), or
    // copied into small_ (or, once sent, sent_) at offset at.
    struct Seg { uint64_t off, len; const uint8_t* ext; uint64_t at; };
    static const uint64_t kLarge = (uint64_t)64 << 10, kBatchBytes = (uint64_t)64 << 20;
    // How much is written before its sending is started, on NFS
    static const uint64_t kWritebackStep = (uint64_t)16 << 20;
    static const size_t kMaxSegs = 512;

    const Seg& segFor(uint64_t p) const {
        size_t lo = 0, hi = segs_.size();   // last segment starting at or before p
        while(hi - lo > 1){ const size_t mid = (lo + hi)/2; if(segs_[mid].off <= p) lo = mid; else hi = mid; }
        return segs_[lo];
    }
    const Seg* sentFor(uint64_t p) const {
        for(const Seg& s : sentSegs_) if(p >= s.off && p < s.off + s.len) return &s;
        return NULL;
    }
    void patchSent(uint64_t p, const uint8_t* src, uint64_t n){
        for(const Seg& s : sentSegs_){
            const uint64_t a = std::max(p, s.off), b = std::min(p + n, s.off + s.len);
            if(a < b) memcpy(&sent_[s.at + (a - s.off)], src + (a - p), (size_t)(b - a));
        }
    }
    // Copy n bytes (zeros when src is NULL) to the end of the queue.
    void appendSmall(const uint8_t* src, uint64_t n){
        if(segs_.empty() || segs_.back().ext || segs_.back().at + segs_.back().len != small_.size())
            segs_.push_back(Seg{end_, 0, NULL, small_.size()});
        if(src) small_.insert(small_.end(), src, src + n);
        else small_.insert(small_.end(), (size_t)n, (uint8_t)0);
        segs_.back().len += n;
        end_ += n;
        queued_ += n;
    }
    bool sendAt(const uint8_t* src, uint64_t n, uint64_t off){
        while(n){
            const ssize_t k = pwrite(fd_, src, (size_t)n, (off_t)off);
            if(k <= 0){ failed_ = true; return false; }
            src += k; n -= (uint64_t)k; off += (uint64_t)k;
        }
        return true;
    }
    // Send the queue, followed by extra (a large write that cannot wait) if any.
    bool send(const Seg* extra){
        if(failed_) return false;
        std::vector<struct iovec> iov;
        iov.reserve(segs_.size() + 1);
        for(const Seg& s : segs_)
            iov.push_back(iovec{const_cast<uint8_t*>(s.ext ? s.ext : &small_[s.at]), (size_t)s.len});
        if(extra) iov.push_back(iovec{const_cast<uint8_t*>(extra->ext), (size_t)extra->len});
        size_t i = 0;
        const size_t iovMax = kIovMax;
        uint64_t off = base_;
#ifndef __linux__
        if(!iov.empty() && lseek(fd_, (off_t)off, SEEK_SET) != (off_t)off){ failed_ = true; return false; }
#endif
        while(i < iov.size()){
            const int cnt = (int)std::min<size_t>(iov.size() - i, iovMax);
#ifdef __linux__
            const ssize_t k = pwritev(fd_, &iov[i], cnt, (off_t)off);
#else
            const ssize_t k = writev(fd_, &iov[i], cnt);
#endif
            if(k <= 0){ failed_ = true; return false; }
            off += (uint64_t)k;
            uint64_t left = (uint64_t)k;   // skip what was written, resume mid-vector after a short write
            while(i < iov.size() && left >= iov[i].iov_len){ left -= iov[i].iov_len; i++; }
            if(i < iov.size() && left){ iov[i].iov_base = static_cast<uint8_t*>(iov[i].iov_base) + left; iov[i].iov_len -= left; }
        }
        // Keep the small writes just sent: libtiff reads the last directory
        // back. A batch of strips alone leaves the previous ones in place.
        bool small = false;
        for(const Seg& s : segs_) small = small || !s.ext;
        if(small){
            sent_.swap(small_);
            sentSegs_.clear();
            for(const Seg& s : segs_) if(!s.ext) sentSegs_.push_back(s);
        }
        small_.clear();
        segs_.clear();
        if(extra) end_ += extra->len;
        base_ = end_;
        queued_ = 0;
#ifdef __linux__
        if(writebackStep_ && base_ - writebackFrom_ >= writebackStep_ && !onNfs(dev_)) writebackStep_ = 0;
        if(writebackStep_ && base_ - writebackFrom_ >= writebackStep_){
            // Only starts sending; a failure is still reported at close.
            sync_file_range(fd_, (off64_t)writebackFrom_, (off64_t)(base_ - writebackFrom_), SYNC_FILE_RANGE_WRITE);
            writebackFrom_ = base_;
        }
#endif
        return true;
    }

    static const size_t kIovMax =
#ifdef IOV_MAX
        IOV_MAX;
#else
        16;
#endif
    int fd_ = -1;
    bool stable_, compressed_, failed_ = false;
    uint64_t pos_ = 0, base_ = 0, end_ = 0, queued_ = 0;   // file [base_, end_) is queued
    uint64_t writebackStep_ = 0, writebackFrom_ = 0;   // sending was started for the file before writebackFrom_
    dev_t dev_ = 0;
    std::vector<Seg> segs_, sentSegs_;
    std::vector<uint8_t> small_, sent_;
};

tmsize_t bfRead(thandle_t h, void* p, tmsize_t n){ return static_cast<BatchedFile*>(h)->read(p, n); }
tmsize_t bfWrite(thandle_t h, void* p, tmsize_t n){ return static_cast<BatchedFile*>(h)->write(p, n); }
toff_t bfSeek(thandle_t h, toff_t off, int whence){ return static_cast<BatchedFile*>(h)->seek(off, whence); }
int bfClose(thandle_t h){ return static_cast<BatchedFile*>(h)->close(); }
toff_t bfSize(thandle_t h){ return static_cast<BatchedFile*>(h)->size(); }
int bfMap(thandle_t, void**, toff_t*){ return 0; }
void bfUnmap(thandle_t, void*, toff_t){}
#endif

// A file in memory, for libtiff to write a stack's directories into (see
// DirectoriesFirst)
struct MemFile {
    std::vector<uint8_t> buf;
    uint64_t pos = 0;
};
tmsize_t mfRead(thandle_t h, void* p, tmsize_t n){
    MemFile* m = static_cast<MemFile*>(h);
    const uint64_t k = m->pos >= m->buf.size() ? 0 : std::min<uint64_t>((uint64_t)n, m->buf.size() - m->pos);
    if(k) memcpy(p, m->buf.data() + m->pos, (size_t)k);
    m->pos += k;
    return (tmsize_t)k;
}
tmsize_t mfWrite(thandle_t h, void* p, tmsize_t n){
    MemFile* m = static_cast<MemFile*>(h);
    if(m->pos + (uint64_t)n > m->buf.size()) m->buf.resize((size_t)(m->pos + (uint64_t)n), 0);
    memcpy(m->buf.data() + m->pos, p, (size_t)n);
    m->pos += (uint64_t)n;
    return n;
}
toff_t mfSeek(thandle_t h, toff_t off, int whence){
    MemFile* m = static_cast<MemFile*>(h);
    m->pos = whence == SEEK_SET ? off : whence == SEEK_CUR ? m->pos + off : m->buf.size() + off;
    return m->pos;
}
int mfClose(thandle_t){ return 0; }
toff_t mfSize(thandle_t h){ return static_cast<MemFile*>(h)->buf.size(); }
int mfMap(thandle_t, void**, toff_t*){ return 0; }
void mfUnmap(thandle_t, void*, toff_t){}

// The layout mode "w" writes: every slice's directory, then their strip (or
// tile) arrays, all before the image data, as GDAL writes cloud-optimized
// TIFFs, so that a reader finds every directory in the first megabytes of the
// file instead of one after each slice's data (a walk of a stack's directories
// cold from network storage took a read for each). libtiff writes the
// directories into memory, with their arrays deferred; here the arrays are laid
// out after them and every link to a directory (the header's to the first,
// each directory's to the next) is 0, and all of it goes to the start of the
// file. As each slice is finished, its data in the file, its arrays are filled
// in and then its directory is linked (finish), so that a file cut short (a
// write killed, a full disk) holds exactly the slices that were finished, as a
// file of the old layout (mode "wc") does: a directory not yet linked is never
// read. The directories are written by libtiff, the rest by the writer itself:
// no directory is read back. Every slice has the same tags, so libtiff writes
// the first three directories and the others are copies of the first, if the
// second and third are (libtiff took 5 us a directory: 0.012 s of a 0.1 s write
// of 2,500 slices of 128 KB to /dev/shm, before any data could be written);
// a value outside its entry would be at another place in each directory (none
// of setTags's is: each fits in its 8 bytes), and then libtiff writes them all.
class DirectoriesFirst {
public:
    // The stack's directories as libtiff writes them (setTags sets a
    // directory's tags), and the arrays after them; false (with why) if they
    // are not as libtiff writes them (BigTIFF in this machine's byte order,
    // with deferred arrays).
    bool build(uint64_t slices, uint64_t chunks, bool tiled, const std::function<void(TIFF*)>& setTags, const char*& why){
        // (CPPTIFF_COPY_DIRECTORIES=0: libtiff writes every one, for tests)
        const char* copyEnv = getenv("CPPTIFF_COPY_DIRECTORIES");
        const bool copy = slices > 3 && !(copyEnv && !strcmp(copyEnv, "0"));
        if(!(copy && libtiffDirs(3, tiled, setTags, why) && copyDirs(slices)) && !libtiffDirs(slices, tiled, setTags, why)) return false;
        why = "the directories are not as libtiff writes them";
        const uint64_t end = meta_.size();
        meta_.reserve((end + 7)/8*8 + (chunks > 1 ? 2*slices*chunks*8 : 0));
        meta_.resize((end + 7)/8*8, 0);
        where_.assign(2*slices, 0);
        nextAt_.assign(slices, 0);
        chunks_ = chunks;
        const uint16_t tags[2] = {(uint16_t)(tiled ? TIFFTAG_TILEOFFSETS : TIFFTAG_STRIPOFFSETS),
                                  (uint16_t)(tiled ? TIFFTAG_TILEBYTECOUNTS : TIFFTAG_STRIPBYTECOUNTS)};
        const uint16_t long8 = TIFF_LONG8;
        for(uint64_t s = 0; s < slices; s++){
            const uint64_t off = ifdAt_[s];
            uint64_t n = 0;
            if(off + 8 > end) return false;
            memcpy(&n, &meta_[off], 8);
            if(off + 8 + n*20 + 8 > end) return false;
            int found = 0;
            for(uint64_t i = 0; i < n; i++){
                const uint64_t e = off + 8 + i*20;   // (the entry: tag, type, count, then the value or where it is)
                uint16_t tag, type;
                uint64_t count, value;
                memcpy(&tag, &meta_[e], 2); memcpy(&type, &meta_[e + 2], 2); memcpy(&count, &meta_[e + 4], 8); memcpy(&value, &meta_[e + 12], 8);
                for(int k = 0; k < 2; k++){
                    if(tag != tags[k]) continue;
                    if(type || count || value) return false;
                    memcpy(&meta_[e + 2], &long8, 2);
                    memcpy(&meta_[e + 4], &chunks, 8);
                    if(chunks == 1) where_[2*s + k] = e + 12;
                    else{
                        const uint64_t at = meta_.size();
                        meta_.resize(at + chunks*8, 0);
                        memcpy(&meta_[e + 12], &at, 8);
                        where_[2*s + k] = at;
                    }
                    found++;
                }
            }
            if(found != 2) return false;
            nextAt_[s] = off + 8 + n*20;
        }
        // (every link 0 until its slice is finished)
        const uint64_t zero = 0;
        memcpy(&meta_[8], &zero, 8);
        for(uint64_t s = 0; s < slices; s++) memcpy(&meta_[nextAt_[s]], &zero, 8);
        why = "";
        return true;
    }
    // Where the image data starts
    uint64_t dataStart() const { return meta_.size(); }
    // Chunk c of slice s is at offset, bytes long
    void set(uint64_t s, uint64_t c, uint64_t offset, uint64_t bytes){
        memcpy(&meta_[where_[2*s] + c*8], &offset, 8);
        memcpy(&meta_[where_[2*s + 1] + c*8], &bytes, 8);
    }
    // Writes all of it at the start of the file (every directory unlinked)
    bool writeAll(TIFF* tif) const { return put(tif, 0, meta_.size()); }
    // Slices [s0, s1), their data in the file: their arrays written, then the
    // links to their directories
    bool finish(TIFF* tif, uint64_t s0, uint64_t s1){
        if(s0 >= s1) return true;
        uint64_t a = UINT64_MAX, b = 0;
        for(uint64_t s = s0; s < s1; s++)
            for(int k = 0; k < 2; k++){
                a = std::min<uint64_t>(a, where_[2*s + k]);
                b = std::max<uint64_t>(b, where_[2*s + k] + chunks_*8);
            }
        if(!put(tif, a, b)) return false;
        a = UINT64_MAX; b = 0;
        for(uint64_t s = s0; s < s1; s++){
            const uint64_t at = s ? nextAt_[s - 1] : 8;
            memcpy(&meta_[at], &ifdAt_[s], 8);
            a = std::min<uint64_t>(a, at);
            b = std::max<uint64_t>(b, at + 8);
        }
        return put(tif, a, b);
    }
private:
    // libtiff writes k directories into memory: meta_, and where each is
    bool libtiffDirs(uint64_t k, bool tiled, const std::function<void(TIFF*)>& setTags, const char*& why){
        MemFile mem;
        TIFF* m = TIFFClientOpen("directories", "w8", (thandle_t)&mem, mfRead, mfWrite, mfSeek, mfClose, mfSize, mfMap, mfUnmap);
        if(!m){ why = "the directories could not be made (out of memory?)"; return false; }
        bool ok = true;
        for(uint64_t s = 0; s < k && ok; s++){
            setTags(m);
            ok = TIFFDeferStrileArrayWriting(m) && TIFFWriteCheck(m, tiled ? 1 : 0, "writeSlices") && TIFFWriteDirectory(m);
        }
        ifdAt_.assign(k, 0);
        for(uint64_t s = 0; s < k && ok; s++) ok = _TIFFGetOffsetFromDirNumber(m, (tdir_t)s, &ifdAt_[s]) != 0;
        ok = ok && TIFFIsBigTIFF(m) && !TIFFIsByteSwapped(m);
        meta_.swap(mem.buf);
        TIFFClose(m);
        if(!ok) why = "the directories could not be made (out of memory?)";
        return ok;
    }
    // libtiff's first three directories (meta_) made into the stack's: the
    // first copied for every slice, each the same distance after the one
    // before; false if libtiff's second and third are not such copies (their
    // links aside, which build sets to 0)
    bool copyDirs(uint64_t slices){
        if(ifdAt_.size() != 3 || ifdAt_[1] <= ifdAt_[0] || ifdAt_[2] <= ifdAt_[1]) return false;
        const uint64_t at = ifdAt_[0], stride = ifdAt_[1] - at, len = meta_.size() - ifdAt_[2];
        uint64_t n = 0;
        if(ifdAt_[2] - ifdAt_[1] != stride || len > stride || len < 16) return false;
        memcpy(&n, &meta_[at], 8);
        if(n > (len - 16)/20) return false;
        const uint64_t link = 8 + n*20;   // (where in a directory its link to the next is)
        for(uint64_t k = 1; k < 3; k++)
            if(memcmp(&meta_[ifdAt_[k]], &meta_[at], link) || memcmp(&meta_[ifdAt_[k] + link + 8], &meta_[at + link + 8], len - link - 8)) return false;
        for(uint64_t k = 0; k < 2; k++)
            for(uint64_t i = ifdAt_[k] + len; i < ifdAt_[k + 1]; i++)
                if(meta_[i]) return false;
        std::vector<uint8_t> all(at + (slices - 1)*stride + len, 0);
        memcpy(all.data(), meta_.data(), at);   // (the header)
        ifdAt_.resize(slices);
        for(uint64_t s = 0; s < slices; s++){
            ifdAt_[s] = at + s*stride;
            memcpy(&all[ifdAt_[s]], &meta_[at], len);
        }
        meta_.swap(all);
        return true;
    }
    // meta_[a, b) to the file, where it is
    bool put(TIFF* tif, uint64_t a, uint64_t b) const {
        thandle_t h = TIFFClientdata(tif);
        return TIFFGetSeekProc(tif)(h, (toff_t)a, SEEK_SET) == (toff_t)a
               && (uint64_t)TIFFGetWriteProc(tif)(h, (void*)&meta_[a], (tmsize_t)(b - a)) == b - a;
    }
    std::vector<uint8_t> meta_;
    std::vector<uint64_t> where_;    // per slice: where in meta_ its offsets go, then its byte counts
    std::vector<uint64_t> ifdAt_;    // per slice: where its directory is
    std::vector<uint64_t> nextAt_;   // per slice: where its directory's link to the next is
    uint64_t chunks_ = 0;
};

// Hands out the chunks to prepare, in file order. A producer takes the next
// one and waits while it is too far ahead of the writer. The writer, when the
// chunk it needs next is not ready yet, takes one within the window and
// prepares it itself instead of waiting, as long as at least `reserve` chunks
// are left for the producers: one taken near the end kept it from writing the
// last chunks for as long as compressing it took (up to tens of ms for large
// LZW strips).
struct ChunkClaims {
    std::atomic<int64_t> next{0};
    int64_t total = 0, window = 0, reserve = 0;
    uint8_t* writerScratch = nullptr;
    // A chunk the writer, at chunk `at`, may prepare now, or -1.
    int64_t forWriter(int64_t at){
        int64_t k = next.load();
        while(k < total - reserve && k < at + window)
            if(next.compare_exchange_weak(k, k + 1)) return k;
        return -1;
    }
};

// Write every slice in order: its chunks (waiting for producers unless
// produceInline, where this thread makes each chunk just before writing it),
// and in mode "w" its arrays and the link to its directory (every directory is
// written first: see DirectoriesFirst), in "wc" and "a" then its directory.
// writerChunk tells producers how far the writer is; while it waits, the
// writer prepares chunks from claims, if given. Returns 0 on success.
uint8_t writeSlices(const WriteJob& j, const char* fileName, const char* mode, Chunk* chunks,
                    std::atomic<int64_t>& writerChunk, std::atomic<bool>& failed,
                    bool produceInline, uint8_t* scratch, BufferPool* pool, ChunkClaims* claims){
    // "w": a new file, every directory first; "wc": a new file of the old
    // layout, each slice's directory after its data; "a": slices added to a
    // file, as "wc" writes them ("wf", the prototype's name, is "w")
    if(strcmp(mode,"w") && strcmp(mode,"wc") && strcmp(mode,"wf") && strcmp(mode,"a")){
        printf("Error: mode \"%s\" is not supported. Use w, wc or a for mode type\n", mode);
        failed = true;
        return 1;
    }
    const bool append = !strcmp(mode,"a");
    const bool dirsFirst = !strcmp(mode,"w") || !strcmp(mode,"wf");
    // (directories first: the finished slices are linked once at least
    // kLinkBytes of their data has been written since the last link, and at the
    // end, so that slices of 1 MB or more are linked one by one: linked one at a
    // time, 2,500 uncompressed slices of 128 KB took 1.02-1.03 times as long to
    // write to a local SSD and to /dev/shm, three system calls more a slice (its
    // data sent, its arrays and its link). A write cut short loses at most 1 MB
    // of its last finished slices, where the classic layout's queue of small
    // strips holds up to 64 MB. CPPTIFF_LINK_SLICES=N links every N slices
    // instead, for tests.)
    const uint64_t kLinkBytes = (uint64_t)1 << 20;
    const char* linkEnv = getenv("CPPTIFF_LINK_SLICES");
    const uint64_t linkEvery = linkEnv && *linkEnv ? std::max<uint64_t>(1, strtoull(linkEnv, NULL, 10)) : 0;
    uint64_t linkedTo = 0, linkedAt = 0;   // (slices finished and linked, and where their data ended)
    DirectoriesFirst firstDirs;
    uint64_t dataAt = 0;   // (directories first: where the next chunk goes)
    TIFF* tif = NULL;
#ifndef _WIN32
    // Uncompressed row-major strips come straight from the caller's array,
    // which outlives the write, so they can wait in the queue.
    BatchedFile io(j.direct(), j.compression != COMPRESSION_NONE);
    if(io.open(fileName, append))
        tif = TIFFClientOpen(fileName, append ? "a8" : "w8", (thandle_t)&io, bfRead, bfWrite, bfSeek, bfClose, bfSize, bfMap, bfUnmap);
#else
    tif = tiffOpenPath(fileName, append ? "a8" : "w8");
#endif
    if(!tif){
        printf("Error: File \"%s\" cannot be opened\n",fileName);
        failed = true;
        return 1;
    }

    uint8_t err = 0;
    const char* what = "";
    // Explicit type tag: caller-provided format, or derive from bits for legacy callers.
    const uint16_t sf = j.sampleFormat ? j.sampleFormat : (j.bits >= 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
    auto setTags = [&](TIFF* tif){
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (uint32_t)j.x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (uint32_t)j.y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (uint16_t)j.bits);
        if(j.tileSize){
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, (uint32_t)j.tileSize);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, (uint32_t)j.tileSize);
        }
        else TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, (uint32_t)j.stripSize);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, j.compression);
        // Color is RGB with each pixel's samples together (chunky). RGBA arrays
        // are normally not premultiplied, so the fourth sample is straight
        // (unassociated) alpha.
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, j.samples > 1 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)j.samples);
        if(j.samples == 4){
            const uint16_t alpha = EXTRASAMPLE_UNASSALPHA;
            TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, (uint16_t)1, &alpha);
        }
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sf);
    };
    // Directories first (see DirectoriesFirst): every directory, unlinked, at
    // the start of the file, and the data after it
    if(dirsFirst){
        if(!firstDirs.build(j.nSlices, j.chunksPerSlice, j.tileSize != 0, setTags, what)) err = 1;
        else if(!firstDirs.writeAll(tif)){ err = 1; what = "the directories could not be written (disk full or I/O error?)"; }
        dataAt = linkedAt = firstDirs.dataStart();
    }
    for(uint64_t s = 0; s < j.nSlices && !err; s++){
        if(!dirsFirst) setTags(tif);

        for(uint64_t c = 0; c < j.chunksPerSlice; c++){
            const uint8_t* p;
            int64_t n;
            Chunk* ch = NULL;
            if(j.direct()){
                p = j.data + s*j.sliceBytes() + j.stripOffset(c);
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
                    const int64_t k = claims ? claims->forWriter((int64_t)(s*j.chunksPerSlice + c)) : -1;
                    if(k < 0) std::this_thread::sleep_for(std::chrono::microseconds(20));
                    else if(!produceChunk(j, (uint64_t)k / j.chunksPerSlice, (uint64_t)k % j.chunksPerSlice, chunks, claims->writerScratch, *pool))
                        failed = true;
                }
                if(n <= 0){ err = 1; what = "could not compress or prepare a strip (out of memory?)"; break; }
                p = ch->data;
            }
            tmsize_t w = -1;
            if(dirsFirst){
                thandle_t h = TIFFClientdata(tif);
                if(TIFFGetSeekProc(tif)(h, (toff_t)dataAt, SEEK_SET) == (toff_t)dataAt && TIFFGetWriteProc(tif)(h, (void*)p, (tmsize_t)n) == (tmsize_t)n){
                    firstDirs.set(s, c, dataAt, (uint64_t)n);
                    dataAt += (uint64_t)n;
                    w = (tmsize_t)n;
                }
            }
            else w = j.tileSize ? TIFFWriteRawTile(tif, (uint32_t)c, (void*)p, (tmsize_t)n)
                                : TIFFWriteRawStrip(tif, (uint32_t)c, (void*)p, (tmsize_t)n);
            if(ch){ pool->put(ch->data); ch->data = NULL; }
            if(w != (tmsize_t)n){ err = 1; what = "a strip could not be written (disk full or I/O error?)"; break; }
        }
        if(!err && !dirsFirst && !TIFFWriteDirectory(tif)){ err = 1; what = "a directory could not be written (disk full or I/O error?)"; }
        // Directories first: the finished slices' data sent first (it may wait
        // in the queue), then their arrays, then the links to them, so that a
        // write cut short never links a slice whose data or arrays are not in
        // the file (written before the queue went out, the arrays had pointed
        // past the end of the file for a quarter of the slices of a write of
        // 20,000 killed near its end)
        const bool linkNow = s + 1 == j.nSlices || (linkEvery ? (s + 1) % linkEvery == 0 : dataAt - linkedAt >= kLinkBytes);
        if(!err && dirsFirst && linkNow){
#ifndef _WIN32
            if(!io.flush()){ err = 1; what = "a strip could not be written (disk full or I/O error?)"; break; }
#endif
            if(!firstDirs.finish(tif, linkedTo, s + 1)){ err = 1; what = "a directory could not be linked (disk full or I/O error?)"; break; }
            linkedTo = s + 1;
            linkedAt = dataAt;
        }
    }
    if(!err && !TIFFFlush(tif)){ err = 1; what = "the file could not be flushed (disk full or I/O error?)"; }
    TIFFClose(tif);
#ifndef _WIN32
    // TIFFClose sends the rest of the queue and closes the file, and ignores
    // the result (NFS, for one, can accept every write and only report a
    // failure such as a full quota at close).
    if(!err && io.failed()){ err = 1; what = "the file could not be written or closed (disk full or I/O error?)"; }
#endif
    if(err){
        printf("Error: writing \"%s\" failed: %s. The file is incomplete.\n", fileName, what);
        failed = true;
    }
    return err;
}

#ifdef __linux__
// The CPUs of the NUMA node holding most of [p, p + bytes), limited to those
// this thread may run on. Samples 64 pages. False when that cannot be told or
// would not narrow anything (a single node), and then nothing is pinned.
bool dataNodeCpus(const uint8_t* p, uint64_t bytes, cpu_set_t& cpus){
    int counts[64] = {0};
    int seen = 0;
    for(int i = 0; i < 64 && bytes; i++){
        int node = -1;
        // get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR): the node of the page at an address.
        if(syscall(SYS_get_mempolicy, &node, NULL, 0UL, (void*)(p + bytes/64*i), 3UL) == 0 && node >= 0 && node < 64){
            counts[node]++;
            seen++;
        }
    }
    if(!seen) return false;
    char path[64];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", (int)(std::max_element(counts, counts + 64) - counts));
    FILE* f = fopen(path, "r");
    if(!f) return false;
    char list[4096];
    const size_t n = fread(list, 1, sizeof list - 1, f);
    fclose(f);
    list[n] = 0;
    CPU_ZERO(&cpus);
    for(char* s = list; *s >= '0' && *s <= '9';){   // e.g. "0-23,48-71"
        char* e;
        const long a = strtol(s, &e, 10);
        long b = a;
        if(*e == '-') b = strtol(e + 1, &e, 10);
        for(long c = a; c <= b && c < CPU_SETSIZE; c++) CPU_SET((int)c, &cpus);
        s = (*e == ',') ? e + 1 : e;
    }
    cpu_set_t allowed;
    if(sched_getaffinity(0, sizeof allowed, &allowed) != 0) return false;
    CPU_AND(&cpus, &cpus, &allowed);
    return CPU_COUNT(&cpus) > 0 && !CPU_EQUAL(&cpus, &allowed);
}
#endif

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
        err = writeSlices(j, fileName, mode, chunks.get(), writerChunk, failed, true, scratch.get(), &pool, NULL);
    }
    else if(j.direct()){
        // Nothing needs preparing, but a new thread still does the writing, as
        // in v1.6.2. Under Linux's automatic NUMA balancing, a thread that has
        // been running a while, like the caller, keeps marking the process's
        // memory so that the next touch of each page faults and may move the
        // page. Writing a 5 GB volume from the caller took about 580,000 such
        // faults and 290,000 page moves per write, 17-93% slower than v1.6.2
        // on two-socket machines. A new thread does not start marking during a
        // write, and the waiting caller marks nothing.
        // On Linux that thread also stays on the NUMA node holding most of the
        // data: left to the scheduler, it landed on the other socket of a
        // 24-core node and a 3.7 GB write took 22% longer.
#ifdef __linux__
        cpu_set_t cpus;
        const bool pin = dataNodeCpus(j.data, j.sliceBytes()*j.nSlices, cpus);
#endif
        err = std::async(std::launch::async, [&]{
#ifdef __linux__
            if(pin) sched_setaffinity(0, sizeof cpus, &cpus);
#endif
            return writeSlices(j, fileName, mode, chunks.get(), writerChunk, failed, true, (uint8_t*)NULL, &pool, (ChunkClaims*)NULL);
        }).get();
    }
    else{
        // The writer needs a core of its own. With a producer on every core
        // besides the writer, it shared one with a busy producer (and at the
        // end with producers spinning in OpenMP's barrier), and the whole
        // pipeline waited on it: LZW writes of short stacks took 1.3-1.4x as
        // long, 2D images 1.15x and 500-slice stacks up to 7% longer on 24-
        // and 48-core nodes. So when the threads fill every core there is one
        // producer fewer. With cores to spare (fewer threads asked for than
        // cores), the writer has a core anyway and instead prepares chunks
        // too while the one it needs next is not ready, which made LZW writes
        // with 4 and 8 threads 1.1-1.25x faster; with every core busy that only
        // delayed the writer (RGB LZW writes 8% slower).
        const bool fullCores = threads >= omp_get_num_procs();
        const int producers = fullCores ? std::max(1, threads - 1) : threads;
        // Producers stay within 64 MB (and at least two chunks per thread) of
        // the writer: enough to keep it busy, while a few dozen buffers carry
        // the whole volume. Uncompressed strips are prepared much faster than
        // they are written, so with a 1 GB window every call faulted in and freed
        // up to 1 GB of new buffers, which made MATLAB-layout writes slower than
        // v1.6.2 on some machines.
        const uint64_t kInFlightBytes = (uint64_t)64 << 20;
        const int64_t window = std::max<int64_t>(2*(int64_t)threads, (int64_t)(kInFlightBytes / std::max<uint64_t>(1, j.maxChunkBytes())));
        const int64_t nChunks = (int64_t)(j.nSlices*j.chunksPerSlice);
        ChunkClaims claims;
        claims.total = nChunks;
        claims.window = window;
        claims.reserve = producers;
        std::unique_ptr<uint8_t[]> writerScratch(needScratch && !fullCores ? new (std::nothrow) uint8_t[j.maxChunkBytes()] : NULL);
        if(needScratch && !fullCores && !writerScratch){
            printf("Error: out of memory writing \"%s\"\n", fileName);
            return 1;
        }
        claims.writerScratch = writerScratch.get();
        std::future<uint8_t> writer = std::async(std::launch::async, writeSlices, std::cref(j), fileName, mode,
                                                 chunks.get(), std::ref(writerChunk), std::ref(failed), false, (uint8_t*)NULL, &pool,
                                                 fullCores ? (ChunkClaims*)NULL : &claims);
        #pragma omp parallel num_threads(producers)
        {
            std::unique_ptr<uint8_t[]> scratch(needScratch ? new (std::nothrow) uint8_t[j.maxChunkBytes()] : NULL);
            if(needScratch && !scratch) failed = true;
            for(int64_t k = claims.next.fetch_add(1); k < nChunks && !failed.load(); k = claims.next.fetch_add(1)){
                // Far ahead of the writer: back off to 1 ms, well under the time the window takes to write.
                for(int us = 50; !failed.load() && k >= writerChunk.load(std::memory_order_acquire) + window; us = std::min(2*us, 1000))
                    std::this_thread::sleep_for(std::chrono::microseconds(us));
                if(failed.load()) break;
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

// Rows per strip when the caller passes 0. Each strip is compressed by one
// thread, so 512-row strips left most cores idle on a 2D image (16 strips
// for 8k x 8k). So strips stay at 512 rows unless the write would have fewer
// than 256 in all, and then shrink toward 256, but never below 64 KB. Only
// the image size decides, so the same data gives the same file on any machine
// and thread count.
uint64_t autoStripRows(uint64_t x, uint64_t y, uint64_t nSlices, uint64_t pixelBytes){
    const uint64_t kRows = 512, kStrips = 256, kMinBytes = (uint64_t)64 << 10;
    const uint64_t slices = std::max<uint64_t>(nSlices, 1);
    const uint64_t perSlice = (kStrips + slices - 1) / slices;
    const uint64_t rowBytes = std::max<uint64_t>(x*pixelBytes, 1);
    const uint64_t rows = std::max((y + perSlice - 1) / perSlice, (kMinBytes + rowBytes - 1) / rowBytes);
    return std::max<uint64_t>(1, std::min(rows, kRows));
}

// Fill in a job from the public functions' arguments. Slices [startSlice, z)
// are written, taken from the start of data. A stripSize of 0 picks the rows
// per strip automatically (autoStripRows).
uint8_t makeJob(WriteJob& j, uint64_t x, uint64_t y, uint64_t z, const void* data, uint64_t bits, uint64_t startSlice,
                uint64_t stripSize, bool transpose, const std::string& compression, uint16_t sampleFormat, uint64_t tileSize,
                uint64_t samples){
    // A TIFF image has at least one pixel. An empty one was written from
    // whatever memory followed the data: the Python writer turned z = 0 into
    // one slice.
    if(!x || !y || z <= startSlice){
        printf("Error: there is nothing to write (%llu x %llu pixels, %llu slices)\n", (unsigned long long)x,
               (unsigned long long)y, (unsigned long long)(z > startSlice ? z - startSlice : 0));
        return 1;
    }
    // The compression, by name in any case: "lzw", "zstd", or "none" ("" too).
    // Other names were written as LZW without a word, and "ZSTD" and "NONE" too.
    std::string name(compression);
    for(char& ch : name) ch = (char)std::tolower((unsigned char)ch);
    if(name != "lzw" && name != "zstd" && name != "none" && !name.empty()){
        printf("Error: unknown compression \"%s\" (use lzw, zstd or none)\n", compression.c_str());
        return 1;
    }
    if(bits != 8 && bits != 16 && bits != 32 && bits != 64){
        printf("Error: %llu-bit samples are not supported\n", (unsigned long long)bits);
        return 1;
    }
    // The TIFF spec requires tile dimensions to be a multiple of 16.
    if(tileSize && tileSize % 16){
        printf("Error: tile size %llu is not a multiple of 16\n",(unsigned long long)tileSize);
        return 1;
    }
    if(samples != 1 && samples != 3 && samples != 4){
        printf("Error: %llu samples per pixel are not supported (1 = grayscale, 3 = RGB, 4 = RGBA)\n", (unsigned long long)samples);
        return 1;
    }
    j.x = x; j.y = y; j.bits = bits; j.bytes = bits/8; j.samples = samples;
    j.nSlices = z > startSlice ? z - startSlice : 0;
    if(!tileSize && !stripSize) stripSize = autoStripRows(x, y, j.nSlices, j.pixelBytes());
    j.stripSize = stripSize; j.tileSize = tileSize;
    j.tilesAcross = tileSize ? (x + tileSize - 1)/tileSize : 0;
    j.tilesDown = tileSize ? (y + tileSize - 1)/tileSize : 0;
    j.chunksPerSlice = tileSize ? j.tilesAcross*j.tilesDown : (y + stripSize - 1)/stripSize;
    j.data = (const uint8_t*)data;
    j.transpose = transpose;
    j.compression = name == "zstd" ? COMPRESSION_ZSTD : name == "lzw" ? COMPRESSION_LZW : COMPRESSION_NONE;
    j.sampleFormat = sampleFormat;
    return 0;
}

}

// Single-threaded write of row-major data.
uint8_t writeTiffSingle(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize, const uint64_t samples){
    (void)tiffOld; (void)transpose;   // tiff is already row-major
    WriteJob j;
    if(makeJob(j, x, y, z, tiff, bits, startSlice, stripSize, false, compression, sampleFormat, tileSize, samples)) return 1;
    return writeJob(j, fileName, mode, 1);
}

// Parallel write of row-major data.
uint8_t writeTiffParallel(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize, const uint64_t samples){
    (void)tiffOld; (void)transpose;   // tiff is already row-major
    WriteJob j;
    if(makeJob(j, x, y, z, tiff, bits, startSlice, stripSize, false, compression, sampleFormat, tileSize, samples)) return 1;
    return writeJob(j, fileName, mode, omp_get_max_threads());
}

// data is row-major (z, y, x[, c]), or column-major MATLAB order (y, x[, c], z) when transpose.
uint8_t writeTiffParallelWrapper(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* data, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize, const uint64_t samples){
    WriteJob j;
    if(makeJob(j, x, y, z, data, bits, startSlice, stripSize, transpose, compression, sampleFormat, tileSize, samples)) return 1;
    return writeJob(j, fileName, mode, omp_get_max_threads());
}

uint8_t writeTiffParallelHelper(const char* fileName, const void* tiffOld, uint64_t bits, const char* mode, uint64_t x, uint64_t y, uint64_t z, uint64_t startSlice, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize, const uint64_t samples)
{
	// Check if folder exists, if not then make it (recursive if needed)
	char* folderName = strdup(fileName);
	char *lastSlash = strrchr(folderName, '/');
    #ifdef _WIN32
    // Either separator, whichever comes last: a path named with / found no
    // folder, so that a write into a folder that did not exist yet failed
    char* lastBackslash = strrchr(folderName, '\\');
    if(lastBackslash && (!lastSlash || lastBackslash > lastSlash)) lastSlash = lastBackslash;
    #endif
	if(lastSlash){
		*lastSlash = '\0';
		mkdirRecursive(folderName);   // (which does nothing when the folder exists)
	}
	free(folderName);

	TIFFSetWarningHandler(DummyHandler);

	if(!mode) mode = "w";

	// For 2D images MATLAB passes in the 3rd dim as 0 so we set it to 1;
	if(!z){
		z = 1;
	}

	// 0 lets the writer choose the rows per strip from the image size.
	uint64_t stripSize = 0;

    return writeTiffParallelWrapper(x, y, z, fileName, tiffOld, bits, startSlice, stripSize, mode, transpose, compression, sampleFormat, tileSize, samples);
}
