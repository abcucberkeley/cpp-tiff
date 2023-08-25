#ifndef WRITETIFFPARALLEL_H
#define WRITETIFFPARALLEL_H

#include <cstdint>

void writeTiffParallel(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, const void* tiffOld, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint64_t stripsPerDir, uint64_t* cSizes, const char* mode);

#endif