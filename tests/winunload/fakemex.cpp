// A stand-in for a MEX file (Windows): a DLL that calls the library and that unloadtest.exe loads and unloads, as
// MATLAB loads a MEX file and unloads it at "clear mex" and at exit
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "helperfunctions.h"
#include "parallelreadtiff.h"
#include "parallelwritetiff.h"

// Writes (if write) a 256 x 256 x 32 16-bit LZW stack with every thread, and reads it back: 0 if it reads as written
extern "C" __declspec(dllexport) int fakemexRun(const char* fileName, int write){
    const uint64_t x = 256, y = 256, z = 32;
    std::vector<uint16_t> data(x*y*z);
    for(size_t i = 0; i < data.size(); i++) data[i] = (uint16_t)(i*7 + i/x);
    if(write && writeTiffParallelWrapper(x, y, z, fileName, data.data(), 16, 0, 0, "w", false, "lzw", 1, 0)) return 10;
    uint64_t* size = getImageSize(fileName);
    if(!size) return 11;
    const bool sameSize = size[0] == y && size[1] == x && size[2] == z;
    free(size);
    if(!sameSize) return 12;
    void* back = readTiffParallelWrapperNoXYFlip(fileName);
    if(!back) return 13;
    const bool same = !std::memcmp(back, data.data(), data.size()*sizeof(uint16_t));
    free(back);
    return same ? 0 : 14;
}
