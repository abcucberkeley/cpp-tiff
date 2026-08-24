#include <cstdint>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <chrono>
#include <thread>
#include <future>
#include "tiffio.h"
#include "lzwencode.h"
#include "helperfunctions.h"

// zstd one-shot encode; symbols come from the libzstd statically linked into libtiff
extern "C" {
    size_t ZSTD_compress(void* dst, size_t dstCap, const void* src, size_t srcSize, int level);
}
#ifndef COMPRESSION_ZSTD
#define COMPRESSION_ZSTD 50000
#endif
static const int kZstdLevel = 1;

// ---- Tiled output -------------------------------------------------------
// A tile is a sub-rectangle, so unlike a strip it is not contiguous in the
// row-major image and has to be gathered before it can be handed to libtiff.
// libtiff also requires a full tile, so edge tiles are zero padded.
static inline void gatherTile(uint8_t* tileBuf, const uint8_t* slice, int64_t x,
                              int64_t tileX0, int64_t tileY0,
                              int64_t validW, int64_t validH,
                              int64_t tw, int64_t tl, int64_t px) {
    const int64_t inRow = x * px;      // image row stride
    const int64_t outRow = tw * px;    // padded tile row stride
    if(validW < tw || validH < tl) memset(tileBuf, 0, (size_t)(tw*tl*px));
    const uint8_t* s = slice + ((tileY0 * x) + tileX0) * px;
    uint8_t* o = tileBuf;
    for (int64_t r = 0; r < validH; r++) {
        memcpy(o, s, (size_t)(validW * px));
        o += outRow;
        s += inRow;
    }
}

uint8_t writeTiffSingle(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    TIFF* tif = NULL;
    if(!strcmp(mode,"w")){
        tif = TIFFOpen(fileName, "w8");
        if(!tif){
            printf("Error: File \"%s\" cannot be opened",fileName);
			return 1;
        }
    }
    else if(!strcmp(mode,"a")){
        tif = TIFFOpen(fileName, "a8");
        if(!tif){
            printf("Error: File \"%s\" cannot be opened",fileName);
			return 1;
        }
    }
    else{
        printf("Error: mode \"%s\" is not supported. Use w or a for mode type", mode);
        return 1;
    }

    uint64_t len = 0;
    int compressionType = COMPRESSION_NONE;
    if(compression == "zstd"){
        compressionType = COMPRESSION_ZSTD;
    }
    else if(compression != "none" && !compression.empty()){
        compressionType = COMPRESSION_LZW;
    }
    // Tiled output geometry. tileSize == 0 keeps the strip path unchanged.
    const uint64_t px = bits/8;
    const uint64_t tw = tileSize, tl = tileSize;
    const uint64_t tileBytes = tileSize ? tw*tl*px : 0;
    const uint64_t tilesAcross = tileSize ? (x + tw - 1)/tw : 0;
    const uint64_t tilesDown = tileSize ? (y + tl - 1)/tl : 0;
    uint8_t* tileBuf = tileSize ? (uint8_t*)malloc(tileBytes) : NULL;

    for(uint64_t dir = startSlice; dir < z; dir++){
        if(dir>=z+startSlice) break;
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits);
        if(tileSize){
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, (uint32_t)tileSize);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, (uint32_t)tileSize);
        }
        else TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, stripSize);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, compressionType);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, 1);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);

        // Explicit type tag: caller-provided format, or derive from bits for legacy callers.
        uint16_t sf = sampleFormat ? sampleFormat : (bits >= 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sf);

        if(tileSize){
            const uint8_t* slice = (const uint8_t*)tiff + ((dir-startSlice)*(x*y))*px;
            for(uint64_t ty = 0; ty < tilesDown; ty++){
                for(uint64_t tx = 0; tx < tilesAcross; tx++){
                    const int64_t tileY0 = (int64_t)(ty*tl), tileX0 = (int64_t)(tx*tw);
                    int64_t validH = (int64_t)y - tileY0; if(validH > (int64_t)tl) validH = (int64_t)tl;
                    int64_t validW = (int64_t)x - tileX0; if(validW > (int64_t)tw) validW = (int64_t)tw;
                    gatherTile(tileBuf, slice, (int64_t)x, tileX0, tileY0, validW, validH,
                               (int64_t)tw, (int64_t)tl, (int64_t)px);
                    TIFFWriteEncodedTile(tif, (uint32_t)(ty*tilesAcross + tx), tileBuf, (int64_t)tileBytes);
                }
            }
        }
        else for (int64_t i = 0; i*stripSize < y; i++)
        {
                if (stripSize*(i+1) > y){
                    len = (y-(stripSize*i))*x*(bits/8);
                }
                else{
                    len = stripSize*x*(bits/8);
                }
                TIFFWriteEncodedStrip(tif, i, (uint8_t*)tiff+((((i*stripSize)*x)+((dir-startSlice)*(x*y)))*(bits/8)), len);
                //if(transpose) TIFFWriteEncodedStrip(tif, i, (uint8_t*)tiff+((((i*stripSize)*x)+((dir-startSlice)*(x*y)))*(bits/8)), len);
                //else TIFFWriteEncodedStrip(tif,i,(uint8_t*)tiffOld+((((i*stripSize)*x)+((dir-startSlice)*(x*y)))*(bits/8)), len);
                //TIFFWriteRawStrip(tif,i,(uint8_t*)tiff+((((i*stripSize)*x)+((dir-startSlice)*(x*y)))*(bits/8)), len);
        }
        TIFFWriteDirectory(tif);
    }
    free(tileBuf);
    TIFFClose(tif);
    return 0;
}

uint8_t writeTiffThread(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const uint64_t stripsPerDir, uint8_t** &comprA, uint64_t* cSizes, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    TIFF* tif = NULL;
    if(!strcmp(mode,"w")){
        tif = TIFFOpen(fileName, "w8");
        if(!tif){
            printf("Error: File \"%s\" cannot be opened",fileName);
            return 1;
        }
    }
    else if(!strcmp(mode,"a")){
        tif = TIFFOpen(fileName, "a8");
        if(!tif){
            printf("Error: File \"%s\" cannot be opened",fileName);
            return 1;
        }
    }
    else{
        printf("Error: mode \"%s\" is not supported. Use w or a for mode type", mode);
        return 1;
    }

    uint64_t len = 0;
    int compressionType = COMPRESSION_NONE;
    bool compress = false;
    if(compression == "zstd"){
        compress = true;
        compressionType = COMPRESSION_ZSTD;
    }
    else if(compression != "none" && !compression.empty()){
        compress = true;
        compressionType = COMPRESSION_LZW;
    }
    // Tiled: every tile arrives through comprA (the parallel loop gathers, and
    // compresses when asked), so the writer thread only has to emit raw tiles.
    const uint64_t tilesAcross = tileSize ? (x + tileSize - 1)/tileSize : 0;
    const uint64_t tilesDown = tileSize ? (y + tileSize - 1)/tileSize : 0;
    const uint64_t tilesPerDir = tilesAcross*tilesDown;

    for(uint64_t dir = startSlice; dir < z; dir++){
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits);
        if(tileSize){
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, (uint32_t)tileSize);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, (uint32_t)tileSize);
        }
        else TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, stripSize);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, compressionType);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, 1);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);

        // Explicit type tag: caller-provided format, or derive from bits for legacy callers.
        uint16_t sf = sampleFormat ? sampleFormat : (bits >= 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sf);

        if(tileSize){
            for(uint64_t t = 0; t < tilesPerDir; t++){
                const uint64_t idx = t + (dir*tilesPerDir);
                while(!cSizes[idx]){
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                TIFFWriteRawTile(tif, (uint32_t)t, comprA[idx], cSizes[idx]);
                free(comprA[idx]);
            }
        }
        else for (int64_t i = 0; i*stripSize < y; i++)
        {
            if(compress){
                while(!cSizes[i+(dir*stripsPerDir)]){
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                TIFFWriteRawStrip(tif,i,comprA[i+(dir*stripsPerDir)], cSizes[i+(dir*stripsPerDir)]);
                free(comprA[i+(dir*stripsPerDir)]);
            }
            else{
                if (stripSize*(i+1) > y){
                    len = (y-(stripSize*i))*x*(bits/8);
                }
                else{
                    len = stripSize*x*(bits/8);
                }
                TIFFWriteRawStrip(tif,i,(uint8_t*)tiff+((((i*stripSize)*x)+((dir-startSlice)*(x*y)))*(bits/8)), len);
            }
        }
        TIFFWriteDirectory(tif);
    }
	free(comprA);
	free(cSizes);
    TIFFClose(tif);
    return 0;
}

uint8_t writeTiffParallel(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* tiff, const void* tiffOld, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    int32_t numWorkers = omp_get_max_threads();
    
    if(numWorkers == 1){
        return writeTiffSingle(x, y, z, fileName, tiff, tiffOld, bits, startSlice, stripSize, mode, transpose, compression, sampleFormat, tileSize);
    }

    // Tiled output geometry. tileSize == 0 keeps the strip path unchanged.
    const bool tiled = tileSize > 0;
    const uint64_t px = bits/8;
    const uint64_t tw = tileSize, tl = tileSize;
    const uint64_t tileBytes = tiled ? tw*tl*px : 0;
    const uint64_t tilesAcross = tiled ? (x + tw - 1)/tw : 0;
    const uint64_t tilesDown = tiled ? (y + tl - 1)/tl : 0;
    const uint64_t tilesPerDir = tilesAcross*tilesDown;

    uint64_t stripsPerDir = (uint64_t)ceil((double)y/(double)stripSize);
    uint64_t chunksPerDir = tiled ? tilesPerDir : stripsPerDir;
    uint64_t totalChunks = chunksPerDir*z;
    uint64_t extraBytes = 2000;
    uint8_t** comprA = NULL;
    uint64_t* cSizes = (uint64_t*)calloc(totalChunks, sizeof(uint64_t));

    std::future<uint8_t> writerThreadResult = std::async(std::launch::async, writeTiffThread, x, y, z, fileName, tiff, bits, startSlice, stripSize, mode, chunksPerDir, std::ref(comprA), cSizes, std::ref(compression), sampleFormat, tileSize);
    const bool useZstd = (compression == "zstd");
    const bool compress = (compression != "none" && !compression.empty());

    // Tiled: gather (and compress) every tile in parallel so the serial writer
    // thread only emits them. Uncompressed tiles gather straight into that buffer.
    if(tiled){
        comprA = (uint8_t**)malloc(totalChunks*sizeof(uint8_t*));

        #pragma omp parallel for
        for(uint64_t dir = startSlice; dir < z; dir++){
            uint8_t* tileBuf = compress ? (uint8_t*)malloc(tileBytes) : NULL;
            const uint8_t* slice = (const uint8_t*)tiff + ((dir-startSlice)*(x*y))*px;
            for(uint64_t ty = 0; ty < tilesDown; ty++){
                for(uint64_t tx = 0; tx < tilesAcross; tx++){
                    const uint64_t idx = (ty*tilesAcross + tx) + (dir*tilesPerDir);
                    const int64_t tileY0 = (int64_t)(ty*tl), tileX0 = (int64_t)(tx*tw);
                    int64_t validH = (int64_t)y - tileY0; if(validH > (int64_t)tl) validH = (int64_t)tl;
                    int64_t validW = (int64_t)x - tileX0; if(validW > (int64_t)tw) validW = (int64_t)tw;
                    if(compress){
                        const uint64_t cbound = (tileBytes+(extraBytes*px))*2+1;
                        comprA[idx] = (uint8_t*)malloc(cbound);
                        gatherTile(tileBuf, slice, (int64_t)x, tileX0, tileY0, validW, validH,
                                   (int64_t)tw, (int64_t)tl, (int64_t)px);
                        if(useZstd) cSizes[idx] = ZSTD_compress(comprA[idx], cbound, tileBuf, tileBytes, kZstdLevel);
                        else cSizes[idx] = lzwEncode(tileBuf, comprA[idx], tileBytes);
                    }
                    else{
                        comprA[idx] = (uint8_t*)malloc(tileBytes);
                        gatherTile(comprA[idx], slice, (int64_t)x, tileX0, tileY0, validW, validH,
                                   (int64_t)tw, (int64_t)tl, (int64_t)px);
                        cSizes[idx] = tileBytes;
                    }
                }
            }
            free(tileBuf);
        }
    }
    else if(compress){
        comprA = (uint8_t**)malloc(totalChunks*sizeof(uint8_t*));

        #pragma omp parallel for
        for(uint64_t dir = startSlice; dir < z; dir++){
            uint64_t len = 0;
            for (uint64_t i = 0; i*stripSize < y; i++)
            {
                uint64_t cbound = (((x*stripSize)*(bits/8))+(extraBytes*(bits/8)))*2+1;
                comprA[i+(dir*stripsPerDir)] = (uint8_t*)malloc(cbound);
                if (stripSize*(i+1) > y){
                    len = (y-(stripSize*i))*x*(bits/8);
                }
                else{
                    len = stripSize*x*(bits/8);
                }
                uint8_t* src = (uint8_t*)tiff+((((i*stripSize)*x)+((dir-startSlice)*(x*y)))*(bits/8));

                if(useZstd){
                    // zstd compresses exactly len bytes (no over-read, no padding).
                    cSizes[i+(dir*stripsPerDir)] = ZSTD_compress(comprA[i+(dir*stripsPerDir)], cbound, src, len, kZstdLevel);
                    continue;
                }
                // Compress exactly the strip's len bytes. An earlier version passed
                // len+extraBytes here, reading past the strip -- a benign over-read on
                // Linux but an out-of-bounds fault under concurrent allocation on Windows
                // (crashed only on multithreaded LZW writes). cbound above keeps the
                // output buffer margin; the source length must be exactly len.
                cSizes[i+(dir*stripsPerDir)] = lzwEncode(src,comprA[i+(dir*stripsPerDir)],len);
            }
        }
    }
    return writerThreadResult.get();
}

uint8_t writeTiffParallelWrapper(const uint64_t x, const uint64_t y, const uint64_t z, const char* fileName, const void* data, const uint64_t bits, const uint64_t startSlice, const uint64_t stripSize, const char* mode, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize){
    int32_t numWorkers = omp_get_max_threads();
    void* tiff = nullptr;

    // The TIFF spec requires tile dimensions to be a multiple of 16.
    if(tileSize && tileSize % 16){
        printf("Error: tile size %llu is not a multiple of 16\n",(unsigned long long)tileSize);
        return 1;
    }

    if(transpose){
        tiff = (void*)malloc(x*y*z*(bits/8));
        // Only use omp if there is more than one thread
        if(numWorkers > 1){
            if(bits == 8){
                #pragma omp parallel for collapse(3)
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((uint8_t*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((uint8_t*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }
            }
            else if(bits == 16){
                #pragma omp parallel for collapse(3)
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((uint16_t*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((uint16_t*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }        
            }
            else if(bits == 32){
                #pragma omp parallel for collapse(3)
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((float*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((float*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }        
            }
            else if(bits == 64){
                #pragma omp parallel for collapse(3)
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((double*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((double*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }        
            }
            else{
                free(tiff);
                return 1;
            }
        }
        else{
            if(bits == 8){
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((uint8_t*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((uint8_t*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }
            }
            else if(bits == 16){
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((uint16_t*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((uint16_t*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }        
            }
            else if(bits == 32){
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((float*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((float*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }        
            }
            else if(bits == 64){
                for(uint64_t dir = 0; dir < z; dir++){
                    for(uint64_t j = 0; j < y; j++){
                        for(uint64_t i = 0; i < x; i++){
                            ((double*)tiff)[i+(j*x)+((dir-startSlice)*(x*y))] = ((double*)data)[j+(i*y)+((dir-startSlice)*(x*y))];
                        }
                    }
                }        
            }
            else{
                free(tiff);
                return 1;
            }
        }
    }
    else{
        return writeTiffParallel(x, y, z, fileName, data, data, bits, startSlice, stripSize, mode, transpose, compression, sampleFormat, tileSize);
    }

    uint8_t ret = writeTiffParallel(x, y, z, fileName, tiff, data, bits, startSlice, stripSize, mode, transpose, compression, sampleFormat, tileSize);
    free(tiff);
    return ret;
}

void writeTiffParallelHelper(const char* fileName, const void* tiffOld, uint64_t bits, const char* mode, uint64_t x, uint64_t y, uint64_t z, uint64_t startSlice, const bool transpose, const std::string &compression, const uint16_t sampleFormat, const uint64_t tileSize)
{
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

	if(!mode) mode = "w";

	// For 2D images MATLAB passes in the 3rd dim as 0 so we set it to 1;
	if(!z){
		z = 1;
	}

	uint64_t stripSize = 512;

    writeTiffParallelWrapper(x, y, z, fileName, tiffOld, bits, startSlice, stripSize, mode, transpose, compression, sampleFormat, tileSize);
}
