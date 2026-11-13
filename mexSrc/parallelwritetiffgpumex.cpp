// parallelWriteTiffGPU(file, data[, mode[, compression[, tileSize[, rgb]]]]): parallelWriteTiff's write, of a gpuArray (the
// Parallel Computing Toolbox's GPU arrays), with the arguments parallelWriteTiff takes: the GPU puts the data in the file's
// row order and compresses zstd strips (nvCOMP, at its one level: the file reads as parallelWriteTiff's does), and LZW and
// uncompressed strips are copied back and written as parallelWriteTiff writes them (the same file, byte for byte)
// (gpuwritetiff.h). Needs an NVIDIA GPU and nvCOMP (see gpureadtiff.h; setenv('CPPTIFF_NVCOMP_LIBRARY', <libnvcomp file>)
// before the first write tells it where nvCOMP is). Linux only for now.
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "tiffio.h"
#include "../src/gpuwritetiff.h"
#include "../src/gpureadtiff.h"
#include "../src/helperfunctions.h"

// A char array or string argument as UTF-8
static std::string textArg(const mxArray* a, const char* what){
    mxArray* chars = nullptr;
    if(mxIsClass(a, "string")){
        mxArray* s = mxDuplicateArray(a);
        mexCallMATLAB(1, &chars, 1, &s, "char");
        a = chars;
    }
    else if(!mxIsChar(a)) mexErrMsgIdAndTxt("tiff:inputError", "The %s must be a string", what);
    char* c = mxArrayToUTF8String(a);
    std::string out = c ? c : "";
    mxFree(c);
    if(chars) mxDestroyArray(chars);
    return out;
}

// At clear (of this MEX file, or clear mex) and when MATLAB exits, the GPU memory GPU reads and writes keep for the next
// call is freed (gpuReleaseBuffers). After reset(gpuDevice), which frees it all, the next call makes new buffers.
static void releaseAtClear(){ gpuReleaseBuffers(); }

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    static bool registered = false;
    if(!registered){
        mexAtExit(releaseAtClear);
        registered = true;
    }
    (void)nlhs; (void)plhs;
    if(nrhs < 2 || nrhs > 6) mexErrMsgIdAndTxt("tiff:inputError","This function takes 2 to 6 arguments");
    std::string fileName = textArg(prhs[0], "first argument (the file name)");
    #ifndef _WIN32
    // A leading ~ or ~name is that home folder; nothing else in the path is expanded
    fileName = expandTilde(fileName.c_str());
    #endif
    if(mxInitGPU() != MX_GPU_SUCCESS) mexErrMsgIdAndTxt("tiff:gpuError","No GPU MATLAB can use (mxInitGPU failed)");
    if(!mxIsGPUArray(prhs[1])) mexErrMsgIdAndTxt("tiff:inputError","The data must be a gpuArray (parallelWriteTiff writes other arrays)");
    const std::string mode = nrhs >= 3 ? textArg(prhs[2], "third argument (the mode)") : "w";
    std::string compression = "lzw";
    if(nrhs >= 4){
        compression = textArg(prhs[3], "fourth argument (the compression)");
        for(char &ch : compression) ch = (char)std::tolower((unsigned char)ch);
        if(compression != "lzw" && compression != "zstd" && compression != "none" && !compression.empty())
            mexErrMsgIdAndTxt("tiff:inputError","The compression must be 'lzw', 'zstd' or 'none'");
    }
    uint64_t tileSize = 0;
    if(nrhs >= 5){
        if(!mxIsNumeric(prhs[4]) || mxIsEmpty(prhs[4]))
            mexErrMsgIdAndTxt("tiff:inputError","The fifth argument (tile size) must be a number");
        const double t = mxGetScalar(prhs[4]);
        if(t < 0) mexErrMsgIdAndTxt("tiff:inputError","Tile size cannot be negative");
        tileSize = (uint64_t)t;
        if(tileSize % 16) mexErrMsgIdAndTxt("tiff:inputError","Tile size must be a multiple of 16");
    }
    bool rgb = false;
    if(nrhs >= 6){
        if(!(mxIsLogical(prhs[5]) || mxIsNumeric(prhs[5])) || mxGetNumberOfElements(prhs[5]) != 1)
            mexErrMsgIdAndTxt("tiff:inputError","The sixth argument (rgb) must be true or false");
        rgb = mxGetScalar(prhs[5]) != 0;
    }

    const mxGPUArray* g = mxGPUCreateFromMxArray(prhs[1]);
    if(mxGPUGetComplexity(g) != mxREAL){
        mxGPUDestroyGPUArray(g);
        mexErrMsgIdAndTxt("tiff:dataTypeError","Complex data cannot be written");
    }
    const mwSize nDims = mxGPUGetNumberOfDimensions(g);
    const mwSize* dims = mxGPUGetDimensions(g);
    uint64_t d[4] = {1, 1, 1, 1};
    for(mwSize k = 0; k < nDims && k < 4; k++) d[k] = (uint64_t)dims[k];
    mxFree((void*)dims);
    const char* bad = nullptr;
    if(nDims > 4) bad = "Data must be 2D, 3D, or 4D (y, x, c, z) color";
    else if(!d[0] || !d[1] || !d[2] || !d[3]) bad = "All input data axes must be of at least size 1";
    const uint64_t y = d[0], x = d[1];
    uint64_t z = d[2], samples = 1;
    if(!bad && (nDims == 4 || rgb)){
        samples = d[2];
        z = d[3];
        if(nDims < 3) bad = "RGB data must be (y, x, c) or (y, x, c, z)";
        else if(samples != 3 && samples != 4) bad = "Color data must have 3 (RGB) or 4 (RGBA) samples in its third dimension";
    }
    uint64_t bits = 0;
    uint16_t sampleFormat = SAMPLEFORMAT_UINT;
    switch(mxGPUGetClassID(g)){
        case mxUINT8_CLASS:  bits = 8;  break;
        case mxINT8_CLASS:   bits = 8;  sampleFormat = SAMPLEFORMAT_INT; break;
        case mxUINT16_CLASS: bits = 16; break;
        case mxINT16_CLASS:  bits = 16; sampleFormat = SAMPLEFORMAT_INT; break;
        case mxUINT32_CLASS: bits = 32; break;
        case mxINT32_CLASS:  bits = 32; sampleFormat = SAMPLEFORMAT_INT; break;
        case mxSINGLE_CLASS: bits = 32; sampleFormat = SAMPLEFORMAT_IEEEFP; break;
        case mxUINT64_CLASS: bits = 64; break;
        case mxINT64_CLASS:  bits = 64; sampleFormat = SAMPLEFORMAT_INT; break;
        case mxDOUBLE_CLASS: bits = 64; sampleFormat = SAMPLEFORMAT_IEEEFP; break;
        default: if(!bad) bad = "Data type not suppported";
    }
    if(bad){
        mxGPUDestroyGPUArray(g);
        mexErrMsgIdAndTxt("tiff:inputError","%s",bad);
    }
    // (MATLAB's layout, (y, x[, c], z); on MATLAB's device, with the GPU work on the default stream after MATLAB's, done
    // when gpuWriteTiff returns)
    std::string err;
    const uint8_t r = gpuWriteTiff(fileName.c_str(), mxGPUGetDataReadOnly(g), x, y, z, bits, samples, sampleFormat, compression,
                                   mode.c_str(), true, tileSize, nullptr, 0, err);
    mxGPUDestroyGPUArray(g);
    if(r) mexErrMsgIdAndTxt("tiff:tiffError","%s",err.c_str());
}
