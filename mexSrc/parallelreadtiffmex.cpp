#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include "mex.h"
#include "tiffio.h"
#include "../src/helperfunctions.h"
#include "../src/parallelreadtiff.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
#if defined(__APPLE__)
    // keep the mex file loaded until MATLAB exits; unloading it crashes
    // MATLAB on exit on macOS (Windows exits cleanly without this)
    if (!mexIsLocked()) {
        mexLock();
    }
#endif
    if(nrhs < 1 || nrhs > 2) mexErrMsgIdAndTxt("tiff:inputError","This function takes one or two arguments only");
    // Check if the fileName is a char array or matlab style
    char* fileName = NULL;
    if(!mxIsClass(prhs[0], "string")){
        if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("tiff:inputError","The first argument must be a string");
        fileName = mxArrayToUTF8String(prhs[0]);   // (paths are UTF-8 on every platform)
    }
    else{ 
        mxArray* mString[1];
        mxArray* mCharA[1];

        // Convert string to char array
        mString[0] = mxDuplicateArray(prhs[0]);
        mexCallMATLAB(1, mCharA, 1, mString, "char");
        fileName = mxArrayToUTF8String(mCharA[0]);
    }

    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    // A leading ~ or ~name is that home folder; nothing else in the path is expanded
    std::string expandedName = expandTilde(fileName);
    fileName = &expandedName[0];
    #endif

    uint8_t flipXY = 1;
    //uint8_t flipXY = 0;


    //if(nrhs > 2){
    //    flipXY = (uint8_t)*(mxGetPr(prhs[2]));
    //}


    // Everything the read needs, from one open and one walk of the directories
    // (it was four or five opens).
    TiffInfo info;
    if(getTiffInfo(fileName, info)) mexErrMsgIdAndTxt("tiff:inputError","File \"%s\" cannot be opened",fileName);

    uint64_t x = info.x, y = info.y, z = 1, bits = info.bits, startSlice = 0;
    const uint64_t spp = info.samples;
    if(nrhs == 1){
        z = info.z;
    }
    else{
        // [first last], 1-based and inclusive: two whole numbers of any numeric
        // class. In unsigned arithmetic a reversed range like [3 2] got through as
        // no slices, which crashed the reader, and [0 2] and [5 2] wrapped around
        // the check; other classes than double were read as a double's bytes.
        if(!mxIsNumeric(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 2)
            mexErrMsgIdAndTxt("tiff:inputError","The range must be two numbers, [first last]");
        mxArray* range = NULL;
        mxArray* given = const_cast<mxArray*>(prhs[1]);
        mexCallMATLAB(1, &range, 1, &given, "double");
        const double first = mxGetPr(range)[0], last = mxGetPr(range)[1];
        mxDestroyArray(range);
        if(first != std::floor(first) || last != std::floor(last) || first < 1 || last < first || last > (double)info.z){
            mexErrMsgIdAndTxt("tiff:rangeOutOfBound","Range is out of bounds");
        }
        startSlice = (uint64_t)first - 1;
        z = (uint64_t)last - startSlice;
    }

    uint64_t samples = spp;

    // Channel-last MATLAB layout: 3D (y,x,z) grayscale, or 4D (y,x,c,z) for RGB/RGBA,
    // whether the file stores the colors interleaved (chunky) or as separate planes.
    uint64_t dim[4];
    dim[0] = y;
    dim[1] = x;
    mwSize ndim;
    if(samples > 1){ ndim = 4; dim[2] = samples; dim[3] = z; }
    else           { ndim = 3; dim[2] = z; }

    // Pick the MATLAB class from bit depth + sample format (2=signed int, 3=float, else unsigned).
    uint64_t sampleFormat = info.sampleFormat;
    mxClassID classID = mxUINT8_CLASS;
    if(bits == 8)       classID = (sampleFormat == 2) ? mxINT8_CLASS  : mxUINT8_CLASS;
    else if(bits == 16) classID = (sampleFormat == 2) ? mxINT16_CLASS : mxUINT16_CLASS;
    else if(bits == 32) classID = (sampleFormat == 3) ? mxSINGLE_CLASS : (sampleFormat == 2) ? mxINT32_CLASS : mxUINT32_CLASS;
    else if(bits == 64) classID = (sampleFormat == 3) ? mxDOUBLE_CLASS : (sampleFormat == 2) ? mxINT64_CLASS : mxUINT64_CLASS;
    else                mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");

    plhs[0] = mxCreateNumericArray(ndim,(mwSize*)dim,classID, mxREAL);
    void* tiff = mxGetData(plhs[0]);

    // Read as the library's other callers read: readTiffParallelInfo picks the
    // reader (and reads a stack of a few large slices one slice at a time with
    // every thread, which the choice made here did not) and gives the readers
    // where the directories start, from the walk above.
    if(!readTiffParallelInfo(fileName, info, tiff, flipXY, {startSlice, startSlice + z}))
        mexErrMsgIdAndTxt("tiff:tiffError","An Error occured within the read function");
}