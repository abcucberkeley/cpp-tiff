#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "tiffio.h"
#include "omp.h"
#include "mex.h"
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' '-L/global/home/groups/software/sl-7.x86_64/modules/libtiff/4.1.0/libtiff/' -ltiff parallelReadTiff.c
//mex COMPFLAGS='$COMPFLAGS /openmp' '-IC:\Program Files (x86)\tiff\include\' '-LC:\Program Files (x86)\tiff\lib\' -ltiffd.lib C:\Users\Matt\Documents\parallelTiff\main.cpp


void DummyHandler(const char* module, const char* fmt, va_list ap)
{
    // ignore errors and warnings
}

// Backup method in case there are errors reading strips
void readTiffParallelBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    int32_t batchSize = (z-1)/numWorkers+1;
    uint64_t bytes = bits/8;
    
    int32_t w;
    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){
        
        TIFF* tif = TIFFOpen(fileName, "r");
        if(!tif) mexErrMsgIdAndTxt("tiff:threadError","Thread %d: File \"%s\" cannot be opened\n",w,fileName);
        
        void* buffer = malloc(x*bytes);
        for(int64_t dir = startSlice+(w*batchSize); dir < startSlice+((w+1)*batchSize); dir++){
            if(dir>=z+startSlice) break;
            
            int counter = 0; 
            while(!TIFFSetDirectory(tif, (uint64_t)dir) && counter<3){
                printf("Thread %d: File \"%s\" Directory \"%d\" failed to open. Try %d\n",w,fileName,dir,counter+1);
                counter++;
            }

            for (int64_t i = 0; i < y; i++) 
            {
                TIFFReadScanline(tif, buffer, i, 0);
                if(!flipXY){
					memcpy(tiff+((i*x)*bytes),buffer,x*bytes);
					continue;
				}
                //loading the data into a buffer
                switch(bits){
                    case 8:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((uint8_t*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((uint8_t*)buffer)[j];
                        }
                        break;
                    case 16:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((uint16_t*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((uint16_t*)buffer)[j];
                        }
                        break;
                    case 32:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((float*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((float*)buffer)[j];
                        }
                        break;
                    case 64:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((double*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((double*)buffer)[j];
                        }
                        break;
                }
            }
        }
        free(buffer);
        TIFFClose(tif);
    }
}

void readTiffParallel(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    int32_t batchSize = (z-1)/numWorkers+1;
    uint64_t bytes = bits/8;

    int32_t w;
    uint8_t errBak = 0;
    uint8_t err = 0;
    char errString[10000];
    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){

        TIFF* tif = TIFFOpen(fileName, "r");
        if(!tif){
            #pragma omp critical
            {
                err = 1;
                sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
            }
        }
        void* buffer = malloc(x*stripSize*bytes);
        for(int64_t dir = startSlice+(w*batchSize); dir < startSlice+((w+1)*batchSize); dir++){
            if(dir>=z+startSlice || err) break;

            uint8_t counter = 0;
            while(!TIFFSetDirectory(tif, (uint64_t)dir) && counter<3){
                printf("Thread %d: File \"%s\" Directory \"%d\" failed to open. Try %d\n",w,fileName,dir,counter+1);
                counter++;
                if(counter == 3){
					#pragma omp critical
					{
						err = 1;
						sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
					}
				}
            }
            if(err) break;
            for (int64_t i = 0; i*stripSize < y; i++)
            {

                //loading the data into a buffer
                int64_t cBytes = TIFFReadEncodedStrip(tif, i, buffer, stripSize*x*bytes);
				if(cBytes < 0){
					#pragma omp critical
					{
                        errBak = 1;
						err = 1;
						sprintf(errString,"Thread %d: Strip %ld cannot be read\n",w,i);
					}
					break;
				}
				if(!flipXY){
					memcpy(tiff+((i*stripSize*x)*bytes),buffer,cBytes);
					continue;
				}
                switch(bits){
                    case 8:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t k = 0; k < stripSize; k++){
                            if((k+(i*stripSize)) >= y) break;
                            for(int64_t j = 0; j < x; j++){
                                ((uint8_t*)tiff)[((j*y)+(k+(i*stripSize)))+((dir-startSlice)*(x*y))] = ((uint8_t*)buffer)[j+(k*x)];
                            }
                        }
                        break;
                    case 16:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t k = 0; k < stripSize; k++){
                            if((k+(i*stripSize)) >= y) break;
                            for(int64_t j = 0; j < x; j++){
                                ((uint16_t*)tiff)[((j*y)+(k+(i*stripSize)))+((dir-startSlice)*(x*y))] = ((uint16_t*)buffer)[j+(k*x)];
                            }
                        }
                        break;
                    case 32:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t k = 0; k < stripSize; k++){
                            if((k+(i*stripSize)) >= y) break;
                            for(int64_t j = 0; j < x; j++){
                                ((float*)tiff)[((j*y)+(k+(i*stripSize)))+((dir-startSlice)*(x*y))] = ((float*)buffer)[j+(k*x)];
                            }
                        }
                        break;
                    case 64:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t k = 0; k < stripSize; k++){
                            if((k+(i*stripSize)) >= y) break;
                            for(int64_t j = 0; j < x; j++){
                                ((double*)tiff)[((j*y)+(k+(i*stripSize)))+((dir-startSlice)*(x*y))] = ((double*)buffer)[j+(k*x)];
                            }
                        }
                        break;
                }
            }
        }
        free(buffer);
        TIFFClose(tif);
    }
    if(err){
        if(errBak) readTiffParallelBak(x, y, z, fileName, tiff, bits, startSlice, flipXY);
        else mexErrMsgIdAndTxt("tiff:threadError",errString);
    }
}

// Backup method in case there are errors reading strips
void readTiffParallel2DBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    int32_t batchSize = (y-1)/numWorkers+1;
    uint64_t bytes = bits/8;
    
    int32_t w;
    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){
        
        TIFF* tif = TIFFOpen(fileName, "r");
        if(!tif) mexErrMsgIdAndTxt("tiff:threadError","Thread %d: File \"%s\" cannot be opened\n",w,fileName);
        
        void* buffer = malloc(x*bytes);
        for(int64_t dir = startSlice+(w*batchSize); dir < startSlice+((w+1)*batchSize); dir++){
            if(dir>=z+startSlice) break;
            
            int counter = 0; 
            while(!TIFFSetDirectory(tif, (uint64_t)0) && counter<3){
                printf("Thread %d: File \"%s\" Directory \"%d\" failed to open. Try %d\n",w,fileName,dir,counter+1);
                counter++;
            }

            for (int64_t i = (w*batchSize); i < ((w+1)*batchSize); i++) 
            {
                if(i >= y) break;
                TIFFReadScanline(tif, buffer, i, 0);
                if(!flipXY){
					memcpy(tiff+((i*x)*bytes),buffer,x*bytes);
					continue;
				}
                //loading the data into a buffer
                switch(bits){
                    case 8:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((uint8_t*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((uint8_t*)buffer)[j];
                        }
                        break;
                    case 16:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((uint16_t*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((uint16_t*)buffer)[j];
                        }
                        break;
                    case 32:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((float*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((float*)buffer)[j];
                        }
                        break;
                    case 64:
                        // Map Values to flip x and y for MATLAB
                        for(int64_t j = 0; j < x; j++){
                            ((double*)tiff)[((j*y)+i)+((dir-startSlice)*(x*y))] = ((double*)buffer)[j];
                        }
                        break;
                }
            }
        }
        free(buffer);
        TIFFClose(tif);
    }
}

void readTiffParallel2D(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    uint64_t stripsPerDir = (uint64_t)ceil((double)y/(double)stripSize);
    int32_t batchSize = (stripsPerDir-1)/numWorkers+1;
    uint64_t bytes = bits/8;

    int32_t w;
    uint8_t err = 0;
    uint8_t errBak = 0;
    char errString[10000];


    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){

        TIFF* tif = TIFFOpen(fileName, "r");
        if(!tif){
            #pragma omp critical
            {
                err = 1;
                sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
            }
        }

        void* buffer = malloc(x*stripSize*bytes);


        uint8_t counter = 0;
        while(!TIFFSetDirectory(tif, 0) && counter<3){
            printf("Thread %d: File \"%s\" Directory \"%d\" failed to open. Try %d\n",w,fileName,0,counter+1);
            counter++;
            if(counter == 3){
				#pragma omp critical
				{
					err = 1;
					sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
				}
			}
        }
        for (int64_t i = (w*batchSize); i < (w+1)*batchSize; i++)
        {
            if(i*stripSize >= y || err) break;
            //loading the data into a buffer
            int64_t cBytes = TIFFReadEncodedStrip(tif, i, buffer, stripSize*x*bytes);
			if(cBytes < 0){
				#pragma omp critical
				{
                    errBak = 1;
					err = 1;
					sprintf(errString,"Thread %d: Strip %ld cannot be read\n",w,i);
				}
				break;
			}
			if(!flipXY){
				memcpy(tiff+((i*stripSize*x)*bytes),buffer,cBytes);
				continue;
			}
            switch(bits){
                case 8:
                    // Map Values to flip x and y for MATLAB
                    for(int64_t k = 0; k < stripSize; k++){
                        if((k+(i*stripSize)) >= y) break;
                        for(int64_t j = 0; j < x; j++){
                            ((uint8_t*)tiff)[((j*y)+(k+(i*stripSize)))] = ((uint8_t*)buffer)[j+(k*x)];
                        }
                    }
                    break;
                case 16:
                    // Map Values to flip x and y for MATLAB
                    for(int64_t k = 0; k < stripSize; k++){
                        if((k+(i*stripSize)) >= y) break;
                        for(int64_t j = 0; j < x; j++){
                            ((uint16_t*)tiff)[((j*y)+(k+(i*stripSize)))] = ((uint16_t*)buffer)[j+(k*x)];
                        }
                    }
                    break;
                case 32:
                    // Map Values to flip x and y for MATLAB
                    for(int64_t k = 0; k < stripSize; k++){
                        if((k+(i*stripSize)) >= y) break;
                        for(int64_t j = 0; j < x; j++){
                            ((float*)tiff)[((j*y)+(k+(i*stripSize)))] = ((float*)buffer)[j+(k*x)];
                        }
                    }
                    break;
                case 64:
                    // Map Values to flip x and y for MATLAB
                    for(int64_t k = 0; k < stripSize; k++){
                        if((k+(i*stripSize)) >= y) break;
                        for(int64_t j = 0; j < x; j++){
                            ((double*)tiff)[((j*y)+(k+(i*stripSize)))] = ((double*)buffer)[j+(k*x)];
                        }
                    }
                    break;
            }
        }
        free(buffer);
        TIFFClose(tif);
    }
    if(err) {
        if(errBak) readTiffParallel2DBak(x, y, z, fileName, tiff, bits, startSlice, flipXY);
        else mexErrMsgIdAndTxt("tiff:threadError",errString);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    char* fileName = mxArrayToString(prhs[0]);
    
   
    uint8_t flipXY = 1;
    //if(nrhs > 2){
    //    flipXY = (uint8_t)*(mxGetPr(prhs[2]));
    //}


    TIFFSetWarningHandler(DummyHandler);
    TIFF* tif = TIFFOpen(fileName, "r");
    if(!tif) mexErrMsgIdAndTxt("tiff:inputError","File \"%s\" cannot be opened",fileName);

    uint64_t x = 1,y = 1,z = 1,bits = 1, startSlice = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &x);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &y);

    if(nrhs == 1){
        uint16_t s = 0, m = 0, t = 1;
        while(TIFFSetDirectory(tif,t)){
            s = t;
            t *= 8;
            if(s > t){
                t = 65535;
                printf("Number of slices > 32768");
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
        if(mxGetN(prhs[1]) != 2){
            mexErrMsgIdAndTxt("tiff:inputError","Input range is not 2");
        }
        else{
            startSlice = (uint64_t)*(mxGetPr(prhs[1]))-1;
            z = (uint64_t)*((mxGetPr(prhs[1])+1))-startSlice;
            if (!TIFFSetDirectory(tif,startSlice+z-1) || !TIFFSetDirectory(tif,startSlice)){
                mexErrMsgIdAndTxt("tiff:rangeOutOfBound","Range is out of bounds");
            }
        }
    }

    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    uint64_t stripSize = 1;
    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &stripSize);
    TIFFClose(tif);
    uint64_t dim[3];
    dim[0] = y;
    dim[1] = x;
    dim[2] = z;

    

    // Case for 2D
    if(z <= 1){
        if(bits == 8){
            plhs[0] = mxCreateNumericArray(3,dim,mxUINT8_CLASS, mxREAL);
            uint8_t* tiff = (uint8_t*)mxGetPr(plhs[0]);
            readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else if(bits == 16){
            plhs[0] = mxCreateNumericArray(3,dim,mxUINT16_CLASS, mxREAL);
            uint16_t* tiff = (uint16_t*)mxGetPr(plhs[0]);
            readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else if(bits == 32){
            plhs[0] = mxCreateNumericArray(3,dim,mxSINGLE_CLASS, mxREAL);
            float* tiff = (float*)mxGetPr(plhs[0]);
            readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else if(bits == 64){
            plhs[0] = mxCreateNumericArray(3,dim,mxDOUBLE_CLASS, mxREAL);
            double* tiff = (double*)mxGetPr(plhs[0]);
            readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else{
            mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");
        }
    }
    // Case for 2D
    else{
        if(bits == 8){
            plhs[0] = mxCreateNumericArray(3,dim,mxUINT8_CLASS, mxREAL);
            uint8_t* tiff = (uint8_t*)mxGetPr(plhs[0]);
            readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else if(bits == 16){
            plhs[0] = mxCreateNumericArray(3,dim,mxUINT16_CLASS, mxREAL);
            uint16_t* tiff = (uint16_t*)mxGetPr(plhs[0]);
            readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else if(bits == 32){
            plhs[0] = mxCreateNumericArray(3,dim,mxSINGLE_CLASS, mxREAL);
            float* tiff = (float*)mxGetPr(plhs[0]);
            readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else if(bits == 64){
            plhs[0] = mxCreateNumericArray(3,dim,mxDOUBLE_CLASS, mxREAL);
            double* tiff = (double*)mxGetPr(plhs[0]);
            readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
        }
        else{
            mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");
        }
    }
}
