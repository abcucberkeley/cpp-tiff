// parallelReadTiffGPU(file) or parallelReadTiffGPU(file, [first last]): parallelReadTiff's read, into a gpuArray (the
// Parallel Computing Toolbox's GPU arrays), for data that is needed on the GPU anyway: the strips' stored bytes are read
// here, copied to the GPU and decompressed there (zstd and deflate with nvCOMP; LZW and the other codecs on the CPU),
// and put in MATLAB's layout, (y, x, z) or (y, x, c, z) for color, by the GPU (gpureadtiff.h). The values are
// parallelReadTiff's. Needs an NVIDIA GPU and nvCOMP (see gpureadtiff.h; setenv('CPPTIFF_NVCOMP_LIBRARY', <libnvcomp
// file>) before the first read tells it where nvCOMP is). Linux only for now.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "../src/gpureadtiff.h"
#include "../src/helperfunctions.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if(nrhs < 1 || nrhs > 2) mexErrMsgIdAndTxt("tiff:inputError","This function takes one or two arguments only");
    char* fileName = NULL;
    if(!mxIsClass(prhs[0], "string")){
        if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("tiff:inputError","The first argument must be a string");
        fileName = mxArrayToUTF8String(prhs[0]);   // (paths are UTF-8 on every platform)
    }
    else{
        mxArray* mString[1];
        mxArray* mCharA[1];
        mString[0] = mxDuplicateArray(prhs[0]);
        mexCallMATLAB(1, mCharA, 1, mString, "char");
        fileName = mxArrayToUTF8String(mCharA[0]);
    }
    #ifndef _WIN32
    // A leading ~ or ~name is that home folder; nothing else in the path is expanded
    std::string expandedName = expandTilde(fileName);
    fileName = &expandedName[0];
    #endif

    if(mxInitGPU() != MX_GPU_SUCCESS) mexErrMsgIdAndTxt("tiff:gpuError","No GPU MATLAB can use (mxInitGPU failed)");
    std::string why;
    if(!gpuReadTiffAvailable(why)) mexErrMsgIdAndTxt("tiff:gpuError","GPU reads are not available: %s",why.c_str());

    TiffInfo info;
    if(getTiffInfo(fileName, info)) mexErrMsgIdAndTxt("tiff:inputError","File \"%s\" cannot be opened",fileName);
    uint64_t z = info.z, startSlice = 0;
    std::vector<uint64_t> zRange;
    if(nrhs == 2){
        // [first last], 1-based and inclusive, two whole numbers of any numeric class (as parallelReadTiff takes it)
        if(!mxIsNumeric(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 2)
            mexErrMsgIdAndTxt("tiff:inputError","The range must be two numbers, [first last]");
        mxArray* range = NULL;
        mxArray* given = const_cast<mxArray*>(prhs[1]);
        mexCallMATLAB(1, &range, 1, &given, "double");
        const double first = mxGetPr(range)[0], last = mxGetPr(range)[1];
        mxDestroyArray(range);
        if(first != std::floor(first) || last != std::floor(last) || first < 1 || last < first || last > (double)info.z)
            mexErrMsgIdAndTxt("tiff:rangeOutOfBound","Range is out of bounds");
        startSlice = (uint64_t)first - 1;
        z = (uint64_t)last - startSlice;
        zRange = {startSlice, startSlice + z};
    }

    // MATLAB's layout: (y, x, z), or (y, x, c, z) for color
    std::vector<mwSize> dim = {(mwSize)info.y, (mwSize)info.x};
    if(info.samples > 1) dim.push_back((mwSize)info.samples);
    dim.push_back((mwSize)z);
    const uint64_t bits = info.bits, fmt = info.sampleFormat;
    mxClassID classID = mxUINT8_CLASS;
    if(bits == 8)       classID = fmt == 2 ? mxINT8_CLASS : mxUINT8_CLASS;
    else if(bits == 16) classID = fmt == 2 ? mxINT16_CLASS : mxUINT16_CLASS;
    else if(bits == 32) classID = fmt == 3 ? mxSINGLE_CLASS : fmt == 2 ? mxINT32_CLASS : mxUINT32_CLASS;
    else if(bits == 64) classID = fmt == 3 ? mxDOUBLE_CLASS : fmt == 2 ? mxINT64_CLASS : mxUINT64_CLASS;
    else mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");

    mxGPUArray* out = mxGPUCreateGPUArray((mwSize)dim.size(), dim.data(), classID, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    if(!out) mexErrMsgIdAndTxt("tiff:readError","Out of GPU memory");
    // (on MATLAB's device, with the GPU work on the default stream, done when gpuReadTiff returns)
    std::string err;
    if(gpuReadTiff({GpuTiffRead{fileName, zRange, mxGPUGetData(out)}}, true, nullptr, 0, err)){
        mxGPUDestroyGPUArray(out);
        mexErrMsgIdAndTxt("tiff:tiffError","%s",err.c_str());
    }
    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
    mxGPUDestroyGPUArray(out);
}
