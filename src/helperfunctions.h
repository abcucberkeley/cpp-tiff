#ifndef CPPTIFF_HELPERFUNCTIONS_H
#define CPPTIFF_HELPERFUNCTIONS_H
#include <cstdint>
#include <stdarg.h>
#include <string>
#include <vector>

#ifndef _WIN32
// The path with a leading ~ or ~name expanded to that home folder; any other path unchanged
std::string expandTilde(const std::string &path);
#endif

void mkdirRecursive(const char *dir);

void DummyHandler(const char* module, const char* fmt, va_list ap);

uint8_t isImageJIm(const char* fileName);

uint64_t imageJImGetZ(const char* fileName);

uint32_t getImageSizeZ(const char* fileName);

uint64_t* getImageSize(const char* fileName);

uint64_t getDataType(const char* fileName);

// Samples per pixel: 1 = grayscale, 3 = RGB, 4 = RGBA. Defaults to 1.
uint64_t getSamplesPerPixel(const char* fileName);

// Sample format: 1 = unsigned int, 2 = signed int, 3 = IEEE float. Defaults to 1.
uint64_t getSampleFormat(const char* fileName);

// What a read needs to know about a file, from one open and one walk of its
// directories (the functions above open the file once each).
struct TiffInfo {
    uint64_t x = 1, y = 1;
    // The directories, and the slices a read returns: the directories, or the
    // images= count of an ImageJ stack (one directory followed by every slice)
    uint64_t directories = 1, z = 1;
    uint64_t bits = 1, samples = 1;
    // 1 = unsigned int, 2 = signed int, 3 = IEEE float
    uint16_t sampleFormat = 1;
    // Rows per strip as libtiff reports it (2^32 - 1 when the tag is missing)
    uint64_t stripSize = 0;
    uint16_t compression = 1;
    bool planar = false, tiled = false;
    // Read through the ImageJ reader: an uncompressed, untiled, unplanar file
    // of one directory with an ImageJ description (isImageJIm)
    bool imageJ = false;
    // The ImageJ description's images= count (imageJImGetZ; 0 without one)
    uint64_t imageJImages = 0;
    // Where each directory starts in the file, as the walk found them: a read
    // given them (readTiffParallelInfo) reaches each of its slices' directories
    // with one read, where a new handle walked every directory before it. Empty
    // if the walk did not find them all; the readers then walk.
    std::vector<uint64_t> offsets;
};

// Fills info; returns 1 if the file cannot be opened as a TIFF.
uint8_t getTiffInfo(const char* fileName, TiffInfo &info);

// The threads the calling thread's reads and writes use (OpenMP's setting for
// that thread: OMP_NUM_THREADS, or every CPU), and setting it for the calling
// thread only. (The library's own OpenMP runtime: bindings need not link one.)
int callThreads();
void setCallThreads(int n);

#endif
