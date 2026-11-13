#ifndef GPUWRITETIFF_H
#define GPUWRITETIFF_H
#include <cstdint>
#include <string>

// Writing TIFF files from GPU memory (NVIDIA GPUs, CUDA): writeTiffParallelWrapper's write, with the data
// taken from the GPU. The GPU puts MATLAB's column-major data in the file's row order, and compresses zstd
// strips with nvCOMP (at nvCOMP's one level: the files read as the CPU writer's do, with other bytes). LZW,
// which nvCOMP does not have, and uncompressed strips are copied to the host a part at a time, the LZW ones
// compressed there as the CPU writer compresses them: those files are the CPU writer's, byte for byte. The
// file is written on the host as the CPU writer writes it (with its strips), while the GPU works on the next
// part.
//
// Needs, at run time only, what GPU reads need (gpureadtiff.h): an NVIDIA driver and nvCOMP 5 or later.
// Supported on Linux for now. Environment variables: CPPTIFF_GPU_WRITE_PART_MB (a write goes to the GPU in
// parts of at most this many MB of its data, so that writing one part overlaps the GPU's work on the next
// and the write's own buffers stay small; by default 256, and for zstd about 6 strips for each of the GPU's
// multiprocessors, up to 2 GB, so that nvCOMP has enough of them at once); CPPTIFF_GPU_PROFILE (each write
// prints where its time went to stderr); CPPTIFF_GPU_TIMEOUT (as for reads); CPPTIFF_GPU_WRITE_HOST=1 (for
// tests: zstd strips are compressed on the host too); CPPTIFF_GPU_STRICT=1 (for tests: a write that would be
// copied to the host whole and written by the CPU writer fails instead).

// Whether GPU writes can run here; if not, why
bool gpuWriteTiffAvailable(std::string &why);

// Writes z slices of x by y pixels of samples samples (1 grayscale, 3 RGB, 4 RGBA) of bits bits (8, 16, 32 or
// 64; sampleFormat as for writeTiffParallelWrapper, 0 to derive it from bits) from data, GPU memory holding
// them row-major (z, y, x[, c]), or with flipXY column-major (y, x[, c], z) as MATLAB has it, to fileName
// (mode "w", or "a" to add them to the file's slices), compressed as compression says ("lzw", "zstd" or
// "none"), in strips (tileSize 0) or tiles of tileSize pixels, with the GPU work on stream (a CUstream /
// cudaStream_t, nullptr for the default stream) after the work queued on it before; returns once the file is
// written: 0, or 1 with the reason in errString. threads: the host threads that compress strips nvCOMP does
// not (0: the call's threads, as for writeTiffParallelWrapper). A tiled write is copied to the host whole and
// written by the CPU writer.
uint8_t gpuWriteTiff(const char* fileName, const void* data, const uint64_t x, const uint64_t y, const uint64_t z,
                     const uint64_t bits, const uint64_t samples, const uint16_t sampleFormat,
                     const std::string &compression, const char* mode, const bool flipXY, const uint64_t tileSize,
                     void* stream, const int threads, std::string &errString);

#endif
