#ifndef GPUREADTIFF_H
#define GPUREADTIFF_H
#include <cstdint>
#include <string>
#include <vector>

// Reading TIFF files straight into GPU memory (NVIDIA GPUs, CUDA): each strip's (or tile's) stored bytes
// are read on the host, copied to the GPU as they are and decompressed there (zstd and deflate with
// nvCOMP, LZW, which nvCOMP does not have, with cpp-tiff's own decoder; the other codecs are decoded on
// the host, into the same copy), and the GPU undoes the horizontal predictor, swaps the bytes of
// big-endian files and puts each strip in its place: what readTiffParallelInfo gives, in GPU memory, for
// when the data is needed on the GPU anyway.
//
// Needs, at run time only, an NVIDIA driver and nvCOMP 5 or later (pip install nvidia-libnvcomp-cu12, or
// set CPPTIFF_NVCOMP_LIBRARY to its libnvcomp file): cpp-tiff loads them on the first GPU read, and builds
// and runs without them. Supported on Linux for now; macOS has no NVIDIA GPUs (gpuReadTiffAvailable says
// why a machine cannot read).
//
// Environment variables: CPPTIFF_NVCOMP_LIBRARY (nvCOMP's library file); CPPTIFF_GPU_TIMEOUT (seconds a
// read waits for the GPU before it fails, 0 for no limit; by default 30 plus 1 per 50 MB: nvCOMP can hang
// on damaged data); CPPTIFF_GPU_PART_MAX_MB (a read goes to the GPU in parts of at most this many MB of
// stored and decoded bytes, so that reading the next part overlaps the GPU's work on the one before and a
// read's own buffers stay small next to its output; by default about 4 compressed strips for each of the
// GPU's multiprocessors, 256 MB to 1 GB and at most an eighth of its free memory); CPPTIFF_GPU_LZW=0 (LZW
// strips are decoded on the host instead); CPPTIFF_GPU_STRICT=1 (for tests: a read that would be read on
// the CPU fails instead).

// Whether GPU reads can run here; if not, why (no NVIDIA driver or GPU, no nvCOMP, ...)
bool gpuReadTiffAvailable(std::string &why);

// Where to load nvCOMP from (a libnvcomp file), before the first GPU read
void gpuReadTiffSetNvcompLibrary(const std::string &path);

// One file to read: its slices (all of them, or zRange: {first} or {first, end}, as for
// readTiffParallelInfo) and the GPU memory they go to, every read's in one CUDA context (the device's
// primary context for memory that CUDA runtime users such as PyTorch, CuPy and MATLAB allocate)
struct GpuTiffRead {
    std::string fileName;
    std::vector<uint64_t> zRange;
    void* out;
};

// Reads the files' slices into their GPU memory, row-major (z, y, x[, c]), or with flipXY column-major
// (y, x[, c], z) as MATLAB has it, with the GPU work on stream (a CUstream / cudaStream_t, nullptr for the
// default stream), and returns once it is done: 0, or 1 with the reason in errString. threads: the host
// threads that read and decode (0: the call's threads, as for readTiffParallelInfo). The reads go to the
// GPU together, which does its work best in big batches. A file the GPU does not decode (strips nvCOMP
// rejects, or a layout the GPU path does not take) is read on the CPU and copied.
uint8_t gpuReadTiff(const std::vector<GpuTiffRead> &reads, const bool flipXY, void* stream, const int threads,
                    std::string &errString);

#endif
