#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <atomic>
#include <memory>
#include <limits.h>
#include <omp.h>
#include "tiffio.h"
#include "../src/helperfunctions.h"

// RAII for libtiff handles: TIFFClose runs on every exit path, so the worker
// functions don't carry a naked TIFF* with a manual close. (C++11: no make_unique.)
struct TiffCloser { void operator()(TIFF* t) const noexcept { if(t) TIFFClose(t); } };
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;


// Cache-blocked x/y transpose of one decoded strip into the column-major
// (MATLAB) output. The decoded strip `buf` is row-major (buf[j + k*x]); the
// output is column-major (out[j*y + (k+rowOff) + sliceOff]). A plain nested
// loop has a large stride (x or y) on whichever array is innermost, so one
// array is streamed through cache one element per line. Tiling by TILE keeps a
// TILE x TILE block of BOTH arrays resident, turning the strided accesses into
// cache hits. j = output column (input x), k = output row within strip (input y).
template <typename T>
static inline void flipTransposeStrip(T* out, const T* buf, int64_t x, int64_t y,
                                      int64_t stripRows, int64_t rowOff, int64_t sliceOff) {
    constexpr int64_t TILE = 128;
    for (int64_t jt = 0; jt < x; jt += TILE) {
        const int64_t jEnd = (jt + TILE < x) ? jt + TILE : x;
        for (int64_t kt = 0; kt < stripRows; kt += TILE) {
            const int64_t kEnd = (kt + TILE < stripRows) ? kt + TILE : stripRows;
            for (int64_t j = jt; j < jEnd; j++) {
                T* o = out + (j * y) + rowOff + sliceOff;   // contiguous in k
                const T* b = buf + j;                       // strided in k (step x)
                for (int64_t k = kt; k < kEnd; k++) o[k] = b[k * x];
            }
        }
    }
}

// Backup method in case there are errors reading strips
uint8_t readTiffParallelBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    int32_t batchSize = (z-1)/numWorkers+1;
    uint64_t bytes = bits/8;

    int32_t w;
	std::atomic<uint8_t> err{0};
	char errString[10000];
    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){
		if(err) continue;
        TIFF* tif = TIFFOpen(fileName, "r");
        if(!tif){
			sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
			err = 1;
		}
        void* buffer = malloc(x*bytes);
        for(int64_t dir = startSlice+(w*batchSize); dir < startSlice+((w+1)*batchSize); dir++){
            if(dir>=z+startSlice || err) break;

            int counter = 0;
            while(!TIFFSetDirectory(tif, (uint64_t)dir) && counter<3){
                printf("Thread %d: File \"%s\" Directory \"%d\" failed to open. Try %d\n",w,fileName,dir,counter+1);
                counter++;
            }

            for (int64_t i = 0; i < y; i++)
            {
                TIFFReadScanline(tif, buffer, i, 0);
                if(!flipXY){
                    // Probably need to fix this
                    memcpy(tiff+(((i*y)+((dir-startSlice)*(x*y)))*bytes),buffer,x*bytes);
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
	if(err){
		printf("%s", errString);
	}
	return err;
}

uint8_t readTiffParallel(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    int32_t batchSize = (z-1)/numWorkers+1;
    uint64_t bytes = bits/8;

    int32_t w;
    uint8_t errBak = 0;
    std::atomic<uint8_t> err{0};
    char errString[10000];
    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){

        uint8_t outCounter = 0;
        TiffPtr tif(TIFFOpen(fileName, "r"));   // RAII: TIFFClose on every exit path
        while(!tif){
            tif.reset(TIFFOpen(fileName, "r"));
            if(outCounter == 3){
                #pragma omp critical
                {
                    err = 1;
                    sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
                }
                break;
            }
            outCounter++;
        }

        // Only the flip path needs an intermediate bounce buffer; the no-flip
        // path decodes each strip directly into the output. new[] (not vector or
        // make_unique) keeps it RAII-owned without zero-filling before the read.
        std::unique_ptr<uint8_t[]> buffer;
        if(flipXY) buffer.reset(new uint8_t[x*stripSize*bytes]);
        for(int64_t dir = startSlice+(w*batchSize); dir < startSlice+((w+1)*batchSize); dir++){
            if(dir>=z+startSlice || err) break;

            uint8_t counter = 0;
            while(!TIFFSetDirectory(tif.get(), (uint64_t)dir) && counter<3){
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
                // No-flip: decode the strip straight into the output array (no
                // bounce buffer, no full-image memcpy). Flip: decode into the
                // buffer and transpose below.
                void* stripDst = flipXY
                    ? static_cast<void*>(buffer.get())
                    : static_cast<void*>(static_cast<uint8_t*>(tiff) + (((i*stripSize*x)+((dir-startSlice)*(x*y)))*bytes));
                int64_t cBytes = TIFFReadEncodedStrip(tif.get(), i, stripDst, stripSize*x*bytes);
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
                    continue;
                }
                // Flip x/y into the column-major MATLAB output with a cache-
                // blocked transpose (see flipTransposeStrip). stripRows handles
                // a partial final strip.
                int64_t stripRows = y - (i*stripSize);
                if(stripRows > (int64_t)stripSize) stripRows = (int64_t)stripSize;
                int64_t rowOff = i*stripSize;
                int64_t sliceOff = (dir-startSlice)*(x*y);
                switch(bits){
                    case 8:
                        flipTransposeStrip<uint8_t>(static_cast<uint8_t*>(tiff), buffer.get(), x,y,stripRows,rowOff,sliceOff);
                        break;
                    case 16:
                        flipTransposeStrip<uint16_t>(static_cast<uint16_t*>(tiff), reinterpret_cast<uint16_t*>(buffer.get()), x,y,stripRows,rowOff,sliceOff);
                        break;
                    case 32:
                        flipTransposeStrip<float>(static_cast<float*>(tiff), reinterpret_cast<float*>(buffer.get()), x,y,stripRows,rowOff,sliceOff);
                        break;
                    case 64:
                        flipTransposeStrip<double>(static_cast<double*>(tiff), reinterpret_cast<double*>(buffer.get()), x,y,stripRows,rowOff,sliceOff);
                        break;
                }
            }
        }
        // bounce buffer auto-freed (unique_ptr); tif auto-closed (TiffPtr)
    }
    if(err){
        if(errBak) return readTiffParallelBak(x, y, z, fileName, tiff, bits, startSlice, flipXY);
        else {
			printf("%s", errString);
		}
    }
	return err;
}

// Backup method in case there are errors reading strips
uint8_t readTiffParallel2DBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    int32_t batchSize = (y-1)/numWorkers+1;
    uint64_t bytes = bits/8;

    int32_t w;
	std::atomic<uint8_t> err{0};
	char errString[10000];
    #pragma omp parallel for
    for(w = 0; w < numWorkers; w++){
		if(err) continue;
        TIFF* tif = TIFFOpen(fileName, "r");
        if(!tif) {
			sprintf(errString,"tiff:threadError","Thread %d: File \"%s\" cannot be opened\n",w,fileName);
			err = 1;
		}
        void* buffer = malloc(x*bytes);
        for(int64_t dir = startSlice+(w*batchSize); dir < startSlice+((w+1)*batchSize); dir++){
            if(dir>=z+startSlice || err) break;

            int counter = 0;
            while(!TIFFSetDirectory(tif, startSlice) && counter<3){
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
	if(err){
		printf("%s", errString);
	}
	return err;
}


uint8_t readTiffParallel2D(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint8_t flipXY){
    int32_t numWorkers = omp_get_max_threads();
    uint64_t stripsPerDir = (uint64_t)ceil((double)y/(double)stripSize);
    int32_t batchSize = (stripsPerDir-1)/numWorkers+1;
    uint64_t bytes = bits/8;

    int32_t w;
    std::atomic<uint8_t> err{0};
    uint8_t errBak = 0;
    char errString[10000];
    uint16_t compressed = 1;
    TIFF* tif = TIFFOpen(fileName, "r");
    TIFFGetField(tif, TIFFTAG_COMPRESSION, &compressed);

    // The other method won't work on specific slices of 3D images for now
    // so start slice must also be 0
    if(numWorkers > 1 || compressed > 1){
        #pragma omp parallel for
        for(w = 0; w < numWorkers; w++){
    
            uint8_t outCounter = 0;
            TIFF* tif = TIFFOpen(fileName, "r");
            while(!tif){
                tif = TIFFOpen(fileName, "r");
                if(outCounter == 3){
                    #pragma omp critical
                    {
                        err = 1;
                        sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
                    }
                    break;
                }
                outCounter++;
            }
    
            void* buffer = malloc(x*stripSize*bytes);
    
    
            uint8_t counter = 0;
            while(!TIFFSetDirectory(tif, startSlice) && counter<3){
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
                // Same cache-blocked transpose as the 3D path (flipTransposeStrip),
                // here for a single image so the slice offset is 0.
                int64_t stripRows = y - (i*stripSize);
                if(stripRows > (int64_t)stripSize) stripRows = (int64_t)stripSize;
                int64_t rowOff = i*stripSize;
                switch(bits){
                    case 8:
                        flipTransposeStrip<uint8_t>((uint8_t*)tiff,(uint8_t*)buffer,x,y,stripRows,rowOff,0);
                        break;
                    case 16:
                        flipTransposeStrip<uint16_t>((uint16_t*)tiff,(uint16_t*)buffer,x,y,stripRows,rowOff,0);
                        break;
                    case 32:
                        flipTransposeStrip<float>((float*)tiff,(float*)buffer,x,y,stripRows,rowOff,0);
                        break;
                    case 64:
                        flipTransposeStrip<double>((double*)tiff,(double*)buffer,x,y,stripRows,rowOff,0);
                        break;
                }
            }
            free(buffer);
            TIFFClose(tif);
        }
    }
    else{
        void* tiffC = NULL;
        FILE *fp = fopen(fileName, "rb");
        if(!fp){ 
			printf("File \"%s\" cannot be opened from Disk\n",fileName);
			err = 1;
			return err;
		}

        if(!tif){ 
			printf("File \"%s\" cannot be opened\n",fileName);
			err = 1;
			return err;
		}
        
		uint64_t offset = 0;
        uint64_t* offsets = NULL;
        TIFFGetField(tif, TIFFTAG_STRIPOFFSETS, &offsets);
        if(!offsets){ 
			printf("Could not get offsets from the tiff file\n");
       		err = 1;
			return err;
		}
		offset = offsets[0];
        uint64_t zSize = x*y*bytes;
    
        fseek(fp, offset, SEEK_SET);


        TIFFClose(tif);
        
        if(!flipXY){
            fread(tiff, 1, zSize, fp);
        }
        else{
            uint64_t size = x*y*z*(bits/8);
            tiffC = malloc(size);
            fread(tiffC, 1, zSize, fp);
        }
        fclose(fp);
        if(flipXY){   
            for(uint64_t k = 0; k < z; k++){
                for(uint64_t j = 0; j < x; j++){
                    for(uint64_t i = 0; i < y; i++){
                        switch(bits){
                            case 8:
                                ((uint8_t*)tiff)[i+(j*y)+(k*x*y)] = ((uint8_t*)tiffC)[j+(i*x)+(k*x*y)];
                                break;
                            case 16:
                                ((uint16_t*)tiff)[i+(j*y)+(k*x*y)] = ((uint16_t*)tiffC)[j+(i*x)+(k*x*y)];
                                break;
                            case 32:
                                ((float*)tiff)[i+(j*y)+(k*x*y)] = ((float*)tiffC)[j+(i*x)+(k*x*y)];
                                break;
                            case 64:
                                ((double*)tiff)[i+(j*y)+(k*x*y)] = ((double*)tiffC)[j+(i*x)+(k*x*y)];
                                break;
                        }
                    }
                }
            }
            free(tiffC);
        }
    }

    if(err) {
        if(errBak) return readTiffParallel2DBak(x, y, z, fileName, tiff, bits, startSlice, flipXY);
        else printf("%s", errString);
    }
	return err;
}


// Reading images saved by ImageJ
uint8_t readTiffParallelImageJ(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint8_t flipXY){
    std::atomic<uint8_t> err{0};
    FILE *fp = fopen(fileName, "rb");
    if(!fp){ 
		printf("File \"%s\" cannot be opened from Disk\n",fileName);
		err = 1;
		return err;
	}
    TIFF* tif = TIFFOpen(fileName, "r");
    if(!tif){ 
		printf("File \"%s\" cannot be opened\n",fileName);
		err = 1;
		return err;
	}
    uint64_t offset = 0;
    uint64_t* offsets = NULL;
    TIFFGetField(tif, TIFFTAG_STRIPOFFSETS, &offsets);
    if(offsets) offset = offsets[0];

    TIFFClose(tif);

    fseek(fp, offset, SEEK_SET);

    uint64_t bytes = bits/8;
    //#pragma omp parallel for
    /*
    for(uint64_t i = 0; i < z; i++){
    uint64_t cOffset = x*y*bytes*i;
    //pread(fd,tiff+cOffset,x*y*bytes,offset+cOffset);
    read(fd,tiff+cOffset,x*y*bytes);
    }*/
    uint64_t chunk = 0;
    uint64_t tBytes = x*y*z*bytes;
    uint64_t bytesRead;
    uint64_t rBytes = tBytes;

    // Can probably read more than INT_MAX now that we use fread
    if(tBytes < INT_MAX) bytesRead = fread(tiff,1,tBytes,fp);
    else{
        while(chunk < tBytes){
            rBytes = tBytes-chunk;
            if(rBytes > INT_MAX) bytesRead = fread(static_cast<uint8_t*>(tiff)+chunk,1,INT_MAX,fp);
            else bytesRead = fread(static_cast<uint8_t*>(tiff)+chunk,1,rBytes,fp);
            chunk += bytesRead;
        }
    }
    fclose(fp);
    // Swap endianess for types greater than 8 bits
    // TODO: May need to change later because we may not always need to swap
    if(bits > 8){
        #pragma omp parallel for
        for(uint64_t i = 0; i < x*y*z; i++){
            switch(bits){
                case 16:
                    //((uint16_t*)tiff)[i] = ((((uint16_t*)tiff)[i] & 0xff) >> 8) | (((uint16_t*)tiff)[i] << 8);
                    //((uint16_t*)tiff)[i] = bswap_16(((uint16_t*)tiff)[i]);
                    ((uint16_t*)tiff)[i] = ((((uint16_t*)tiff)[i] << 8) & 0xff00) | ((((uint16_t*)tiff)[i] >> 8) & 0x00ff);
                    break;
                case 32:
                    //((num & 0xff000000) >> 24) | ((num & 0x00ff0000) >> 8) | ((num & 0x0000ff00) << 8) | (num << 24)
                    //((float*)tiff)[i] = bswap_32(((float*)tiff)[i]);
                    ((uint32_t*)tiff)[i] = ((((uint32_t*)tiff)[i] << 24) & 0xff000000 ) |
                        ((((uint32_t*)tiff)[i] <<  8) & 0x00ff0000 ) |
                        ((((uint32_t*)tiff)[i] >>  8) & 0x0000ff00 ) |
                        ((((uint32_t*)tiff)[i] >> 24) & 0x000000ff );
                    break;
                case 64:
                    //((double*)tiff)[i] = bswap_64(((double*)tiff)[i]);
                    ((uint64_t*)tiff)[i] = ( (((uint64_t*)tiff)[i] << 56) & 0xff00000000000000UL ) |
                        ( (((uint64_t*)tiff)[i] << 40) & 0x00ff000000000000UL ) |
                        ( (((uint64_t*)tiff)[i] << 24) & 0x0000ff0000000000UL ) |
                        ( (((uint64_t*)tiff)[i] <<  8) & 0x000000ff00000000UL ) |
                        ( (((uint64_t*)tiff)[i] >>  8) & 0x00000000ff000000UL ) |
                        ( (((uint64_t*)tiff)[i] >> 24) & 0x0000000000ff0000UL ) |
                        ( (((uint64_t*)tiff)[i] >> 40) & 0x000000000000ff00UL ) |
                        ( (((uint64_t*)tiff)[i] >> 56) & 0x00000000000000ffUL );
                    break;
            }

        }
    }
    // Find a way to do this in-place without making a copy
    if(flipXY){
        uint64_t size = x*y*z*(bits/8);
        void* tiffC = malloc(size);
        memcpy(tiffC,tiff,size);
        #pragma omp parallel for
        for(uint64_t k = 0; k < z; k++){
            for(uint64_t j = 0; j < x; j++){
                for(uint64_t i = 0; i < y; i++){
                    switch(bits){
                        case 8:
                            ((uint8_t*)tiff)[i+(j*y)+(k*x*y)] = ((uint8_t*)tiffC)[j+(i*x)+(k*x*y)];
                            break;
                        case 16:
                            ((uint16_t*)tiff)[i+(j*y)+(k*x*y)] = ((uint16_t*)tiffC)[j+(i*x)+(k*x*y)];
                            break;
                        case 32:
                            ((float*)tiff)[i+(j*y)+(k*x*y)] = ((float*)tiffC)[j+(i*x)+(k*x*y)];
                            break;
                        case 64:
                            ((double*)tiff)[i+(j*y)+(k*x*y)] = ((double*)tiffC)[j+(i*x)+(k*x*y)];
                            break;
                    }
                }
            }
        }
        free(tiffC);
    }
	return err;
}


// tiff pointer guaranteed to be NULL or the correct size array for the tiff file
void* readTiffParallelWrapperHelper(const char* fileName, void* tiff, uint8_t flipXY, const std::vector<uint64_t> &zRange = {})
{
	TIFFSetWarningHandler(DummyHandler);
	TIFF* tif = TIFFOpen(fileName, "r");
	if(!tif) return NULL;

	uint64_t x = 1,y = 1,z = 1,bits = 1, startSlice = 0;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &x);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &y);
    z = getImageSizeZ(fileName);

	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
	uint64_t stripSize = 1;
	TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &stripSize);
	TIFFClose(tif);

	// Check if image is an imagej image with imagej metadata
	// Get the correct
	uint8_t imageJIm = 0;
	if(isImageJIm(fileName)){
		imageJIm = 1;
		uint64_t tempZ = imageJImGetZ(fileName);
		if(tempZ) z = tempZ;
	}

    if(zRange.size()){
        if(zRange.size() == 2){
            startSlice = zRange[0];
            z = zRange[1];
        }
        else{
            startSlice = zRange[0];
            z = zRange[0]+1;
        }
    }


	if(imageJIm){
		if(bits == 8){
			if(!tiff) tiff = (uint8_t*)malloc(x*y*z*sizeof(uint8_t));
			readTiffParallelImageJ(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize,flipXY);
			return (void*)tiff;
		}
		else if(bits == 16){
			if(!tiff) tiff = (uint16_t*)malloc(x*y*z*sizeof(uint16_t));
			readTiffParallelImageJ(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 32){
			if(!tiff) tiff = (float*)malloc(x*y*z*sizeof(float));
			readTiffParallelImageJ(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 64){
			if(!tiff) tiff = (double*)malloc(x*y*z*sizeof(double));
			readTiffParallelImageJ(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else{
			return NULL;
		}
	}
	else if(z <= 1){
		if(bits == 8){
			if(!tiff) tiff = (uint8_t*)malloc(x*y*z*sizeof(uint8_t));
			readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize,flipXY);
			return (void*)tiff;
		}
		else if(bits == 16){
			if(!tiff) tiff = (uint16_t*)malloc(x*y*z*sizeof(uint16_t));
			readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 32){
			if(!tiff) tiff = (float*)malloc(x*y*z*sizeof(float));
			readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 64){
			if(!tiff) tiff = (double*)malloc(x*y*z*sizeof(double));
			readTiffParallel2D(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else{
			return NULL;
		}
	}
	else{
		if(bits == 8){
			if(!tiff) tiff = (uint8_t*)malloc(x*y*z*sizeof(uint8_t));
			readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 16){
			if(!tiff) tiff = (uint16_t*)malloc(x*y*z*sizeof(uint16_t));
			readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 32){
			if(!tiff) tiff = (float*)malloc(x*y*z*sizeof(float));
			readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else if(bits == 64){
			if(!tiff) tiff = (double*)malloc(x*y*z*sizeof(double));
			readTiffParallel(x,y,z,fileName, (void*)tiff, bits, startSlice, stripSize, flipXY);
			return (void*)tiff;
		}
		else{
			return NULL;
		}
	}

	// Should never get here but return NULL if we do
	return NULL;
}

void* readTiffParallelWrapper(const char* fileName)
{
	return readTiffParallelWrapperHelper(fileName,NULL,1);
}

void* readTiffParallelWrapperNoXYFlip(const char* fileName, const std::vector<uint64_t> &zRange)
{
	return readTiffParallelWrapperHelper(fileName,NULL,0,zRange);
}

// tTiff doesn't matter as tiff is set in the function
void readTiffParallelWrapperSet(const char* fileName, void* tiff){
	void* tTiff = readTiffParallelWrapperHelper(fileName,tiff,0);
}
