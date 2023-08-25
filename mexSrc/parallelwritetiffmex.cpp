#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "../src/helperfunctions.h"
#include "../src/parallelwritetiff.h"
#include "tiffio.h"
#include "omp.h"
#include "mex.h"
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' '-L/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' -ltiff /clusterfs/fiona/matthewmueller/parallelTiffTesting/main.c
//mex COMPFLAGS='$COMPFLAGS /openmp' '-IC:\Program Files (x86)\tiff\include\' '-LC:\Program Files (x86)\tiff\lib\' -ltiffd.lib C:\Users\Matt\Documents\parallelTiff\main.cpp

//zlib
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' '-I/global/home/groups/consultsw/sl-7.x86_64/modules/zlib/1.2.11/include/' '-L/global/home/groups/consultsw/sl-7.x86_64/modules/zlib/1.2.11/lib' -lz '-L/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' -ltiff parallelWriteTiff.c

//lzw
//mex -v CXXOPTIMFLAGS="-O3 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' '-L/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' -ltiff parallelWriteTiff.c lzw.c

//libtiff 4.4.0
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/clusterfs/fiona/matthewmueller/software/tiff-4.4.0/include' '-L/clusterfs/fiona/matthewmueller/software/tiff-4.4.0/lib' -ltiff parallelWriteTiff.c lzwEncode.c

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if(nrhs < 2) mexErrMsgIdAndTxt("tiff:inputError","This function requires at least 2 arguments");

    // Check if the fileName is a char array or matlab style
    char* fileName = NULL;
    if(!mxIsClass(prhs[0], "string")){
        if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("tiff:inputError","The first argument must be a string");
        fileName = mxArrayToString(prhs[0]);
    }
    else{ 
        mxArray* mString[1];
        mxArray* mCharA[1];

        // Convert string to char array
        mString[0] = mxDuplicateArray(prhs[0]);
        mexCallMATLAB(1, mCharA, 1, mString, "char");
        fileName = mxArrayToString(mCharA[0]);
    }

    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    if(strchr(fileName,'~')) fileName = expandTilde(fileName);
    #endif

    // Check if folder exists, if not then make it (recursive if needed)
    char* folderName = strdup(fileName);
    char *lastSlash = NULL;
    #ifdef _WIN32
    lastSlash = strrchr(folderName, '\\');
    #else
    lastSlash = strrchr(folderName, '/');
    #endif
    if(lastSlash){
        *lastSlash = '\0';
        FILE* f = fopen(folderName,"r");
        if(f){
            fclose(f);
        }
        else{
            mkdirRecursive(folderName);
        }
    }
    free(folderName);

    TIFFSetWarningHandler(DummyHandler);
    const char* mode;
    if(nrhs > 2){
         mode = mxArrayToString(prhs[2]);
    }
    else{
        mode = "w";
    }
    int nDims = (int) mxGetNumberOfDimensions(prhs[1]);
    if(nDims < 2 || nDims > 3) mexErrMsgIdAndTxt("tiff:inputError","Data must be 2D or 3D");
    uint64_t* dims = (uint64_t*) mxGetDimensions(prhs[1]);



    uint64_t x = dims[1],y = dims[0],z = dims[2],bits = 0, startSlice = 0;

    // For 2D images MATLAB passes in the 3rd dim as 0 so we set it to 1;
    if(!z){
        z = 1;
    }

    mxClassID mDType = mxGetClassID(prhs[1]);
    if(mDType == mxUINT8_CLASS){
        bits = 8;
    }
    else if(mDType == mxUINT16_CLASS){
        bits = 16;
    }
    else if(mDType == mxSINGLE_CLASS){
        bits = 32;
    }
    else if(mDType == mxDOUBLE_CLASS){
        bits = 64;
    }

    //mexErrMsgIdAndTxt("tiff:inputError","TESTING");

    uint64_t stripSize = 512;
    uint64_t stripsPerDir = (uint64_t)ceil((double)y/(double)stripSize);
    uint64_t totalStrips = stripsPerDir*z;
    uint64_t* cSizes = (uint64_t*)malloc(totalStrips*sizeof(uint64_t));

    uint64_t dim[3];
    dim[0] = y;
    dim[1] = x;
    dim[2] = z;

    if(bits == 8){
        uint8_t* tiffOld = (uint8_t*)mxGetPr(prhs[1]);
        uint8_t* tiff = (uint8_t*)malloc(x*y*z*(bits/8));

        #pragma omp parallel for collapse(3)
        for(uint64_t dir = 0; dir < z; dir++){
            for(uint64_t j = 0; j < y; j++){
                for(uint64_t i = 0; i < x; i++){
                    ((uint8_t*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((uint8_t*)tiffOld)[j+(i*y)+((dir-startSlice)*(x*y))];
                }
            }
        }
        writeTiffParallel(x,y,z,fileName, (void*)tiff, (void*)tiffOld, bits, startSlice, stripSize, stripsPerDir, cSizes, mode);
        free(tiff);
    }
    else if(bits == 16){

        uint16_t* tiffOld = (uint16_t*)mxGetPr(prhs[1]);
        uint16_t* tiff = (uint16_t*)malloc(x*y*z*(bits/8));

        #pragma omp parallel for collapse(3)
        for(uint64_t dir = 0; dir < z; dir++){
            for(uint64_t j = 0; j < y; j++){
                for(uint64_t i = 0; i < x; i++){
                    ((uint16_t*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((uint16_t*)tiffOld)[j+(i*y)+((dir-startSlice)*(x*y))];
                }
            }
        }
        writeTiffParallel(x,y,z,fileName, (void*)tiff, (void*)tiffOld, bits, startSlice, stripSize, stripsPerDir, cSizes, mode);
        free(tiff);
    }
    else if(bits == 32){
        float* tiffOld = (float*)mxGetPr(prhs[1]);
        float* tiff = (float*)malloc(x*y*z*(bits/8));

        #pragma omp parallel for collapse(3)
        for(uint64_t dir = 0; dir < z; dir++){
            for(uint64_t j = 0; j < y; j++){
                for(uint64_t i = 0; i < x; i++){
                    ((float*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((float*)tiffOld)[j+(i*y)+((dir-startSlice)*(x*y))];
                }
            }
        }
        writeTiffParallel(x,y,z,fileName, (void*)tiff, (void*)tiffOld, bits, startSlice, stripSize, stripsPerDir, cSizes, mode);
        free(tiff);
    }
    else if(bits == 64){
        double* tiffOld = (double*)mxGetPr(prhs[1]);
        double* tiff = (double*)malloc(x*y*z*(bits/8));

        #pragma omp parallel for collapse(3)
        for(uint64_t dir = 0; dir < z; dir++){
            for(uint64_t j = 0; j < y; j++){
                for(uint64_t i = 0; i < x; i++){
                    ((double*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((double*)tiffOld)[j+(i*y)+((dir-startSlice)*(x*y))];
                }
            }
        }
        writeTiffParallel(x,y,z,fileName, (void*)tiff, (void*)tiffOld, bits, startSlice, stripSize, stripsPerDir, cSizes, mode);
        free(tiff);
    }
    else{
        mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");
    }
    free(cSizes);
}
