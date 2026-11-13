#ifndef WRITETIFFPARALLEL_H
#define WRITETIFFPARALLEL_H

#include <cstdint>
#include <string>

// samples is the samples per pixel: 1 = grayscale, 3 = RGB, 4 = RGBA. Color
// data keeps each pixel's samples together: row-major (z, y, x, c), or MATLAB
// column-major (y, x, c, z) when transpose is true.
//
// mode: "w" writes a new file with every slice's directory first, before the
// image data (as GDAL writes cloud-optimized TIFFs), so that a reader finds them
// all at the start of the file instead of one after each slice's data (a walk
// of a stack's directories cold from network storage took a read for each);
// each slice is linked into the file once its data is written, so that a write
// cut short holds exactly the slices that were finished. "wc" writes a new file
// of the classic layout, each slice's directory after its data (as cpp-tiff
// wrote "w" before). "a" adds slices to a file, of either layout, as "wc"
// writes them.
uint8_t writeTiffSingle(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, uint16_t sampleFormat = 0, uint64_t tileSize = 0, uint64_t samples = 1);

uint8_t writeTiffParallel(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, uint16_t sampleFormat = 0, uint64_t tileSize = 0, uint64_t samples = 1);

// stripSize is rows per strip; 0 chooses them from the image size, as the
// Python and MATLAB writers do.
uint8_t writeTiffParallelWrapper(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* data, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, uint16_t sampleFormat = 0, uint64_t tileSize = 0, uint64_t samples = 1);

// Returns 0 on success.
uint8_t writeTiffParallelHelper(const char* fileName, const void* tiffOld, uint64_t bits, const char* mode, uint64_t x, uint64_t y, uint64_t z, uint64_t startSlice, const bool transpose, const std::string &compression, uint16_t sampleFormat = 0, uint64_t tileSize = 0, uint64_t samples = 1);

#endif
