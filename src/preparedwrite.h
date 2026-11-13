#ifndef CPPTIFF_PREPAREDWRITE_H
#define CPPTIFF_PREPAREDWRITE_H
#include <atomic>
#include <cstdint>
#include <string>

// Writes whose strips something other than the writer's own threads makes (GPU writes, which compress on the
// GPU): the writer's file, its tags, layout and I/O, the strips taken as they become ready. Internal to the
// library (not installed).

// One strip or tile on its way to the file (the writer's own chunks are these too). Its maker fills data and then
// publishes size with release order; the writer reads size with acquire order before touching data, so it never sees a
// size before the bytes behind it.
struct PreparedChunk {
    std::atomic<int64_t> size{0};   // 0 = not ready, > 0 = bytes in data, < 0 = failed
    uint8_t* data = nullptr;
};

// The rows per strip the writer chooses for a write of nSlices slices of x by y pixels of pixelBytes bytes
uint64_t preparedStripRows(uint64_t x, uint64_t y, uint64_t nSlices, uint64_t pixelBytes);

// Writes fileName (mode "w" or "a") as writeTiffParallelWrapper writes z slices of x by y pixels of samples samples of
// bits bits (sampleFormat; compression "lzw", "zstd" or "none"; stripSize rows per strip), its strips taken from chunks:
// strip c of slice s is chunks[s*stripsPerSlice + c], written once its size is set. writerChunk tells the maker how far
// the writer is: every chunk before it is written, and its bytes are free. failed is set when the writer fails (and
// the maker sets it to stop the writer). The parent folder is made if it does not exist. Returns 0 on success, or 1
// with the reason in why.
uint8_t writePreparedTiff(const char* fileName, const char* mode, uint64_t x, uint64_t y, uint64_t z, uint64_t bits,
                          uint64_t samples, uint16_t sampleFormat, const std::string &compression, uint64_t stripSize,
                          PreparedChunk* chunks, std::atomic<int64_t> &writerChunk, std::atomic<bool> &failed,
                          std::string &why);

#endif
