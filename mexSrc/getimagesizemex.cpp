#include "tiffio.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../src/helperfunctions.h"
#include "mex.h"
//mex -v COPTIMFLAGS="-O3 -fwrapv -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' '-L/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' -ltiff /clusterfs/fiona/matthewmueller/parallelTiffTesting/main.c
//mex COMPFLAGS='$COMPFLAGS /openmp' '-IC:\Program Files (x86)\tiff\include\' '-LC:\Program Files (x86)\tiff\lib\' -ltiffd.lib C:\Users\Matt\Documents\parallelTiff\main.cpp

//libtiff 4.4.0
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/clusterfs/fiona/matthewmueller/software/tiff-4.4.0/include' '-L/clusterfs/fiona/matthewmueller/software/tiff-4.4.0/lib' -ltiff getImageSize_mex.c

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 1) mexErrMsgIdAndTxt("tiff:inputError","This function requires one argument only");
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
    
    TIFFSetWarningHandler(DummyHandler);
    TIFF* tif = TIFFOpen(fileName, "r");
    if(!tif) mexErrMsgIdAndTxt("tiff:inputError","File \"%s\" cannot be opened",fileName);
    
    uint64_t x = 1,y = 1,z = 1;    
    if(nrhs == 1){
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &x);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &y);
        uint16_t s = 0, m = 0, t = 1;   
        while(TIFFSetDirectory(tif,t)){
            s = t;
            t *= 8;
            if(s > t){ 
                t = 65535;
                printf("Number of slices > 32768\n");
                break;
            }
        }
        while(s != t){
            m = (s+t+1)/2;
            if(TIFFSetDirectory(tif,m)){
                s = m;
            }
            else{
                if(m > 0) t = m-1;
                else t = m;
            }
        }
        z = s+1;
    }
    else{
        mexErrMsgIdAndTxt("tiff:inputError","Function only accepts one input argument");       
    }
  
    TIFFClose(tif);
	if(isImageJIm(fileName)){
		uint64_t tempZ = imageJImGetZ(fileName);
		if(tempZ) z = tempZ;
	}

    plhs[0] = mxCreateNumericMatrix(1,3,mxDOUBLE_CLASS, mxREAL);
    double* dims = (double*)mxGetPr(plhs[0]);
    dims[0] = y;
    dims[1] = x;
    dims[2] = z;
    
}
