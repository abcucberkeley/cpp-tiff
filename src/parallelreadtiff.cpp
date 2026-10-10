#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <atomic>
#include <memory>
#include <algorithm>
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

// RGB/multi-sample variant: de-interleaves a chunky strip (input
// buf[(k*x + j)*samples + c]) into the column-major (y, x, c, z) MATLAB output
// while transposing x/y. Only used when samples>1; the grayscale path keeps
// using flipTransposeStrip above, so its hot loop is unchanged.
template <typename T>
static inline void flipTransposeStripRGB(T* out, const T* buf, int64_t x, int64_t y,
                                         int64_t stripRows, int64_t rowOff, int64_t sliceOff,
                                         int64_t samples) {
    constexpr int64_t TILE = 128;
    const int64_t chanStride = x * y;     // output channel-plane stride
    const int64_t inRow = x * samples;    // input (chunky) row stride, in elements
    for (int64_t c = 0; c < samples; c++) {
        T* outc = out + c * chanStride + sliceOff;
        const T* bufc = buf + c;
        for (int64_t jt = 0; jt < x; jt += TILE) {
            const int64_t jEnd = (jt + TILE < x) ? jt + TILE : x;
            for (int64_t kt = 0; kt < stripRows; kt += TILE) {
                const int64_t kEnd = (kt + TILE < stripRows) ? kt + TILE : stripRows;
                for (int64_t j = jt; j < jEnd; j++) {
                    T* o = outc + (j * y) + rowOff;   // contiguous in k
                    const T* b = bufc + j * samples;  // strided in k (step inRow)
                    for (int64_t k = kt; k < kEnd; k++) o[k] = b[k * inRow];
                }
            }
        }
    }
}

// ---- Tiled TIFF support -------------------------------------------------
// A tile is a sub-rectangle rather than a full-width strip, so its rows are not
// contiguous in the output (the no-flip path cannot decode straight into the
// output like it does for strips) and libtiff pads edge tiles out to a full
// tile. validW/validH bound every copy to the part inside the image.

// Copy one decoded tile into the row-major (no-flip) output, row by row.
static inline void copyTileNoFlip(uint8_t* out, const uint8_t* buf, int64_t x,
                                  int64_t tileX0, int64_t tileY0,
                                  int64_t validW, int64_t validH,
                                  int64_t tw, int64_t sliceOff,
                                  int64_t samples, int64_t bytes) {
    const int64_t px = samples * bytes;      // bytes per pixel
    const int64_t inRow = tw * px;           // padded tile row stride
    const int64_t outRow = x * px;           // image row stride
    const int64_t copyBytes = validW * px;
    uint8_t* o = out + sliceOff + ((tileY0 * x) + tileX0) * px;
    const uint8_t* b = buf;
    for (int64_t r = 0; r < validH; r++) {
        memcpy(o, b, (size_t)copyBytes);
        o += outRow;
        b += inRow;
    }
}

// Tile equivalent of flipTransposeStrip: transpose one tile into the
// column-major (y, x, z) MATLAB output. j = output column (input x),
// k = output row within the tile (input y).
template <typename T>
static inline void flipTransposeTile(T* out, const T* buf, int64_t y,
                                     int64_t tileX0, int64_t tileY0,
                                     int64_t validW, int64_t validH,
                                     int64_t tw, int64_t sliceOff) {
    constexpr int64_t TILE = 128;
    for (int64_t jt = 0; jt < validW; jt += TILE) {
        const int64_t jEnd = (jt + TILE < validW) ? jt + TILE : validW;
        for (int64_t kt = 0; kt < validH; kt += TILE) {
            const int64_t kEnd = (kt + TILE < validH) ? kt + TILE : validH;
            for (int64_t j = jt; j < jEnd; j++) {
                T* o = out + ((tileX0 + j) * y) + tileY0 + sliceOff;  // contiguous in k
                const T* b = buf + j;                                 // strided in k (step tw)
                for (int64_t k = kt; k < kEnd; k++) o[k] = b[k * tw];
            }
        }
    }
}

// Tile equivalent of flipTransposeStripRGB (chunky multi-sample tiles).
template <typename T>
static inline void flipTransposeTileRGB(T* out, const T* buf, int64_t x, int64_t y,
                                        int64_t tileX0, int64_t tileY0,
                                        int64_t validW, int64_t validH,
                                        int64_t tw, int64_t sliceOff, int64_t samples) {
    constexpr int64_t TILE = 128;
    const int64_t chanStride = x * y;        // output channel-plane stride
    const int64_t inRow = tw * samples;      // input (chunky) tile row stride
    for (int64_t c = 0; c < samples; c++) {
        T* outc = out + c * chanStride + sliceOff;
        const T* bufc = buf + c;
        for (int64_t jt = 0; jt < validW; jt += TILE) {
            const int64_t jEnd = (jt + TILE < validW) ? jt + TILE : validW;
            for (int64_t kt = 0; kt < validH; kt += TILE) {
                const int64_t kEnd = (kt + TILE < validH) ? kt + TILE : validH;
                for (int64_t j = jt; j < jEnd; j++) {
                    T* o = outc + ((tileX0 + j) * y) + tileY0;
                    const T* b = bufc + j * samples;
                    for (int64_t k = kt; k < kEnd; k++) o[k] = b[k * inRow];
                }
            }
        }
    }
}

// Decode tiles [tileStart, tileEnd) of the current directory into the output.
// Shared by the 2D and 3D readers (they differ only in how tiles are split
// across workers). Returns 1 and sets failedTile if a tile cannot be read.
static uint8_t readTilesInto(TIFF* tif, uint8_t* buffer, int64_t tileBufBytes,
                             void* out, int64_t x, int64_t y, int64_t bits,
                             int64_t samples, int64_t tw, int64_t tl,
                             int64_t tilesAcross, int64_t tileStart, int64_t tileEnd,
                             int64_t sliceIdx, uint8_t flipXY, int64_t* failedTile) {
    const int64_t bytes = bits / 8;
    for (int64_t t = tileStart; t < tileEnd; t++) {
        const int64_t tileY0 = (t / tilesAcross) * tl;
        const int64_t tileX0 = (t % tilesAcross) * tw;
        int64_t validH = y - tileY0; if (validH > tl) validH = tl;
        int64_t validW = x - tileX0; if (validW > tw) validW = tw;
        if (validW <= 0 || validH <= 0) continue;

        if (TIFFReadEncodedTile(tif, (uint32_t)t, buffer, tileBufBytes) < 0) {
            *failedTile = t;
            return 1;
        }
        if (!flipXY) {
            copyTileNoFlip((uint8_t*)out, buffer, x, tileX0, tileY0, validW, validH, tw,
                           sliceIdx * y * x * samples * bytes, samples, bytes);
            continue;
        }
        if (samples == 1) {
            const int64_t sliceOff = sliceIdx * x * y;
            switch (bits) {
                case 8:
                    flipTransposeTile<uint8_t>((uint8_t*)out, (const uint8_t*)buffer, y, tileX0, tileY0, validW, validH, tw, sliceOff);
                    break;
                case 16:
                    flipTransposeTile<uint16_t>((uint16_t*)out, (const uint16_t*)buffer, y, tileX0, tileY0, validW, validH, tw, sliceOff);
                    break;
                case 32:
                    flipTransposeTile<float>((float*)out, (const float*)buffer, y, tileX0, tileY0, validW, validH, tw, sliceOff);
                    break;
                case 64:
                    flipTransposeTile<double>((double*)out, (const double*)buffer, y, tileX0, tileY0, validW, validH, tw, sliceOff);
                    break;
            }
        }
        else {
            const int64_t sliceOff = sliceIdx * x * y * samples;
            switch (bits) {
                case 8:
                    flipTransposeTileRGB<uint8_t>((uint8_t*)out, (const uint8_t*)buffer, x, y, tileX0, tileY0, validW, validH, tw, sliceOff, samples);
                    break;
                case 16:
                    flipTransposeTileRGB<uint16_t>((uint16_t*)out, (const uint16_t*)buffer, x, y, tileX0, tileY0, validW, validH, tw, sliceOff, samples);
                    break;
                case 32:
                    flipTransposeTileRGB<float>((float*)out, (const float*)buffer, x, y, tileX0, tileY0, validW, validH, tw, sliceOff, samples);
                    break;
                case 64:
                    flipTransposeTileRGB<double>((double*)out, (const double*)buffer, x, y, tileX0, tileY0, validW, validH, tw, sliceOff, samples);
                    break;
            }
        }
    }
    return 0;
}

// Open the file for one reader thread. On Linux this skips libtiff's memory
// map. With every thread mapping the whole file, a read ended with each thread
// unmapping it; the kernel serializes those unmaps and they stall the other
// threads' page faults, which cost up to ~0.4 s at the end of a 2 s read.
// Reading through read() only adds a copy of the compressed strips. macOS
// measured no difference and Windows is unmeasured, so they keep the mapping.
static TIFF* openReaderTiff(const char* fileName){
#ifdef __linux__
    return TIFFOpen(fileName, "rm");
#else
    return TIFFOpen(fileName, "r");
#endif
}

// Place one decoded scanline (row `row` of output slice `slice`) in the output.
// No-flip keeps the chunky row-major (z, y, x, c) layout the strip path writes;
// flip writes the column-major MATLAB (y, x, c, z) layout, matching
// flipTransposeStrip and flipTransposeStripRGB. Bit-exact copies, so the
// sample format does not matter.
template <typename T>
static void placeScanlineFlipped(T* out, const T* line, uint64_t x, uint64_t y,
                                 uint64_t samples, uint64_t row, uint64_t slice){
    T* base = out + slice*x*y*samples + row;
    for(uint64_t j = 0; j < x; j++)
        for(uint64_t c = 0; c < samples; c++)
            base[j*y + c*x*y] = line[j*samples + c];
}

static void placeScanline(void* out, const void* line, uint64_t x, uint64_t y, uint64_t samples,
                          uint64_t bits, uint64_t row, uint64_t slice, uint8_t flipXY){
    const uint64_t rowBytes = x*samples*(bits/8);
    if(!flipXY){
        memcpy(static_cast<uint8_t*>(out) + (slice*y + row)*rowBytes, line, rowBytes);
        return;
    }
    switch(bits){
        case 8:  placeScanlineFlipped(static_cast<uint8_t*>(out),  static_cast<const uint8_t*>(line),  x, y, samples, row, slice); break;
        case 16: placeScanlineFlipped(static_cast<uint16_t*>(out), static_cast<const uint16_t*>(line), x, y, samples, row, slice); break;
        case 32: placeScanlineFlipped(static_cast<uint32_t*>(out), static_cast<const uint32_t*>(line), x, y, samples, row, slice); break;
        case 64: placeScanlineFlipped(static_cast<uint64_t*>(out), static_cast<const uint64_t*>(line), x, y, samples, row, slice); break;
    }
}

static bool setDirectoryWithRetry(TIFF* tif, uint64_t dir){
    for(int attempt = 0; attempt < 3; attempt++)
        if(TIFFSetDirectory(tif, (tdir_t)dir)) return true;
    return false;
}

// Backup method in case there are errors reading strips: re-read every slice a
// scanline at a time, each worker taking a block of whole slices. Reports an
// error if a row still cannot be read rather than leaving it unwritten.
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
        const int64_t first = (int64_t)startSlice + (int64_t)w*batchSize;
        const int64_t last = std::min<int64_t>(first + batchSize, (int64_t)(startSlice + z));
        if(first >= last) continue;

        TiffPtr tif(openReaderTiff(fileName));
        if(!tif){
            #pragma omp critical
            {
                err = 1;
                sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
            }
            continue;
        }
        uint16_t spp = 1;
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
        std::unique_ptr<uint8_t[]> line(new uint8_t[x*spp*bytes]);
        for(int64_t dir = first; dir < last && !err; dir++){
            if(!setDirectoryWithRetry(tif.get(), (uint64_t)dir)){
                #pragma omp critical
                {
                    err = 1;
                    sprintf(errString,"Thread %d: TIFFSetDirectory(%ld) failed\n",w,(long)dir);
                }
                break;
            }
            for(uint64_t i = 0; i < y; i++){
                if(TIFFReadScanline(tif.get(), line.get(), (uint32_t)i, 0) < 0){
                    #pragma omp critical
                    {
                        err = 1;
                        sprintf(errString,"Thread %d: Row %lu of directory %ld cannot be read\n",w,(unsigned long)i,(long)dir);
                    }
                    break;
                }
                placeScanline(tiff, line.get(), x, y, spp, bits, i, (uint64_t)(dir - (int64_t)startSlice), flipXY);
            }
        }
    }
    if(err){
        printf("%s", errString);
    }
    return err;
}

uint8_t readTiffParallel(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint64_t stripSize, uint8_t flipXY){
    uint64_t bytes = bits/8;
    // At most one thread per slice, so a short stack does not open the file
    // once for every core.
    const int32_t team = (int32_t)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)omp_get_max_threads(), z));

    uint8_t errBak = 0;
    std::atomic<uint8_t> err{0};
    char errString[10000];
    // Each thread opens the file once and reads whichever slices it is handed.
    #pragma omp parallel num_threads(team)
    {
        const int32_t w = omp_get_thread_num();
        uint8_t outCounter = 0;
        TiffPtr tif(openReaderTiff(fileName));   // RAII: TIFFClose on every exit path
        while(!tif){
            tif.reset(openReaderTiff(fileName));
            if(outCounter == 3){
                #pragma omp critical
                {
                    err = 1;
                    sprintf(errString,"Thread %d: TIFFOpen failed for \"%s\"\n",w,fileName);
                }
                break;
            }
            outCounter++;
        }

        // Samples per pixel (1=grayscale, 3=RGB, 4=RGBA). xs is the interleaved
        // row width in samples; xs==x when grayscale, so those paths are unchanged.
        uint16_t spp = 1;
        if(tif) TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
        const uint64_t samples = spp;
        const uint64_t xs = x * samples;

        // Tiled TIFFs store sub-rectangles instead of full-width strips.
        const bool tiled = tif && TIFFIsTiled(tif.get());
        uint32_t tw = 0, tl = 0;
        uint64_t tileBufBytes = 0, tilesAcross = 0, tilesDown = 0;
        if(tiled){
            TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tw);
            TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &tl);
            const int64_t tileBytes = (int64_t)TIFFTileSize(tif.get());
            // Bail out rather than divide by zero below or read nothing at all.
            if(!tw || !tl || tileBytes <= 0){
                #pragma omp critical
                {
                    err = 1;
                    sprintf(errString,"Thread %d: tiled image has invalid tile geometry\n",w);
                }
            }
            else{
                tileBufBytes = (uint64_t)tileBytes;
                tilesAcross = (x + tw - 1) / tw;
                tilesDown = (y + tl - 1) / tl;
            }
        }

        // One scratch buffer per thread, reused for every slice it reads. Tiles
        // always decode into it; strips only on the flip path, since the no-flip
        // path decodes each strip straight into the output. new[] (not vector or
        // make_unique) keeps it RAII-owned without zero-filling before the read.
        std::unique_ptr<uint8_t[]> buffer;
        if(tiled && tileBufBytes) buffer.reset(new uint8_t[tileBufBytes]);
        else if(!tiled && flipXY) buffer.reset(new uint8_t[xs*stripSize*bytes]);

        // Read one slice into its place in the output.
        auto readOne = [&](int64_t dir){
            if(err || !tif) return;

            uint8_t counter = 0;
            while(!TIFFSetDirectory(tif.get(), (uint64_t)dir)){
                if(++counter == 3){
                    #pragma omp critical
                    {
                        err = 1;
                        sprintf(errString,"Thread %d: TIFFSetDirectory(%ld) failed\n",w,(long)dir);
                    }
                    return;
                }
            }

            if(tiled){
                int64_t failedTile = -1;
                if(readTilesInto(tif.get(), buffer.get(), (int64_t)tileBufBytes, tiff,
                                 (int64_t)x, (int64_t)y, (int64_t)bits, (int64_t)samples,
                                 (int64_t)tw, (int64_t)tl, (int64_t)tilesAcross,
                                 0, (int64_t)(tilesAcross*tilesDown),
                                 dir - (int64_t)startSlice, flipXY, &failedTile)){
                    #pragma omp critical
                    {
                        // No errBak: the scanline fallback cannot read tiled images.
                        err = 1;
                        sprintf(errString,"Thread %d: Tile %ld cannot be read\n",w,(long)failedTile);
                    }
                }
                return;
            }

            for (int64_t i = 0; i*stripSize < y; i++)
            {
                // No-flip: decode the strip straight into the output array (no
                // bounce buffer, no full-image memcpy). Flip: decode into the
                // buffer and transpose below.
                void* stripDst = flipXY
                    ? static_cast<void*>(buffer.get())
                    : static_cast<void*>(static_cast<uint8_t*>(tiff) + (((i*stripSize*xs)+((dir-startSlice)*(y*xs)))*bytes));
                int64_t cBytes = TIFFReadEncodedStrip(tif.get(), i, stripDst, stripSize*xs*bytes);
                if(cBytes < 0){
                    #pragma omp critical
                    {
                        errBak = 1;
                        err = 1;
                        sprintf(errString,"Thread %d: Strip %ld cannot be read\n",w,(long)i);
                    }
                    return;
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
                if(samples == 1){
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
                } else {
                    int64_t sliceOffC = (dir-startSlice)*(x*y*samples);
                    switch(bits){
                        case 8:
                            flipTransposeStripRGB<uint8_t>(static_cast<uint8_t*>(tiff), buffer.get(), x,y,stripRows,rowOff,sliceOffC,samples);
                            break;
                        case 16:
                            flipTransposeStripRGB<uint16_t>(static_cast<uint16_t*>(tiff), reinterpret_cast<uint16_t*>(buffer.get()), x,y,stripRows,rowOff,sliceOffC,samples);
                            break;
                        case 32:
                            flipTransposeStripRGB<float>(static_cast<float*>(tiff), reinterpret_cast<float*>(buffer.get()), x,y,stripRows,rowOff,sliceOffC,samples);
                            break;
                        case 64:
                            flipTransposeStripRGB<double>(static_cast<double*>(tiff), reinterpret_cast<double*>(buffer.get()), x,y,stripRows,rowOff,sliceOffC,samples);
                            break;
                    }
                }
            }
        };

        // On Linux, slices are handed out one at a time once there are more of
        // them than threads, so a thread that runs slow does not hold up the
        // whole read. Together with the unmapped reads (openReaderTiff), that
        // cut read time by 21-34% on our benchmarks and by up to ~70% on stacks
        // of small slices. With no more slices than threads each thread simply
        // takes one. Elsewhere each thread keeps one contiguous
        // block, as before: on macOS the one-at-a-time split was no faster, and
        // with a memory-mapped file each thread's walk past the slices it skips
        // costs more than the balancing saves.
        if(z <= (uint64_t)team){
            #pragma omp for schedule(static)
            for(int64_t dir = (int64_t)startSlice; dir < (int64_t)(startSlice+z); dir++) readOne(dir);
        }
        else{
#ifdef __linux__
            #pragma omp for schedule(dynamic, 1)
#else
            #pragma omp for schedule(static)
#endif
            for(int64_t dir = (int64_t)startSlice; dir < (int64_t)(startSlice+z); dir++) readOne(dir);
        }
    }
    if(err){
        if(errBak) return readTiffParallelBak(x, y, z, fileName, tiff, bits, startSlice, flipXY);
        else {
			printf("%s", errString);
		}
    }
	return err;
}

// Backup method in case there are errors reading strips of a single image:
// re-read it a scanline at a time. Compressed strips can only be decoded from
// their first row, so workers split the image by whole strips and read each
// one in row order. Reports an error if a row still cannot be read rather than
// leaving it unwritten.
uint8_t readTiffParallel2DBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY){
    (void)z;   // a single image
    uint64_t bytes = bits/8;
    uint32_t rowsPerStrip = 0;
    {
        TiffPtr probe(openReaderTiff(fileName));
        if(probe && setDirectoryWithRetry(probe.get(), startSlice))
            TIFFGetFieldDefaulted(probe.get(), TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
    }
    if(!rowsPerStrip || rowsPerStrip > y) rowsPerStrip = (uint32_t)y;
    const uint64_t numStrips = (y + rowsPerStrip - 1) / rowsPerStrip;
    const int32_t numWorkers = (int32_t)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)omp_get_max_threads(), numStrips));
    const uint64_t stripsPerWorker = (numStrips + numWorkers - 1) / numWorkers;

    int32_t w;
    std::atomic<uint8_t> err{0};
    char errString[10000];
    #pragma omp parallel for num_threads(numWorkers)
    for(w = 0; w < numWorkers; w++){
        if(err) continue;
        const uint64_t rowBegin = std::min<uint64_t>((uint64_t)w*stripsPerWorker*rowsPerStrip, y);
        const uint64_t rowEnd = std::min<uint64_t>(((uint64_t)w + 1)*stripsPerWorker*rowsPerStrip, y);
        if(rowBegin >= rowEnd) continue;

        TiffPtr tif(openReaderTiff(fileName));
        if(!tif){
            #pragma omp critical
            {
                err = 1;
                sprintf(errString,"Thread %d: File \"%s\" cannot be opened\n",w,fileName);
            }
            continue;
        }
        if(!setDirectoryWithRetry(tif.get(), startSlice)){
            #pragma omp critical
            {
                err = 1;
                sprintf(errString,"Thread %d: TIFFSetDirectory(%lu) failed\n",w,(unsigned long)startSlice);
            }
            continue;
        }
        uint16_t spp = 1;
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
        std::unique_ptr<uint8_t[]> line(new uint8_t[x*spp*bytes]);
        for(uint64_t i = rowBegin; i < rowEnd; i++){
            if(TIFFReadScanline(tif.get(), line.get(), (uint32_t)i, 0) < 0){
                #pragma omp critical
                {
                    err = 1;
                    sprintf(errString,"Thread %d: Row %lu cannot be read\n",w,(unsigned long)i);
                }
                break;
            }
            placeScanline(tiff, line.get(), x, y, spp, bits, i, 0, flipXY);
        }
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
    TIFF* tif = TIFFOpen(fileName, "r");
    if(!tif){
        printf("File \"%s\" cannot be opened\n",fileName);
        return 1;
    }
    uint16_t spp = 1;
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    const uint64_t samples = spp;          // 1=grayscale, 3=RGB, 4=RGBA
    const uint64_t xs = x * samples;       // interleaved row width (xs==x when grayscale)

    // Tiled TIFFs store sub-rectangles instead of full-width strips. A single 2D
    // image is one directory, so here the workers are split across tiles.
    const bool tiled = TIFFIsTiled(tif);
    uint32_t tw = 0, tl = 0;
    uint64_t tileBufBytes = 0, tilesAcross = 0, tilesDown = 0, numTiles = 0;
    if(tiled){
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tl);
        const int64_t tileBytes = (int64_t)TIFFTileSize(tif);
        // Bail out rather than divide by zero below or read nothing at all.
        if(!tw || !tl || tileBytes <= 0){
            printf("Tiled image \"%s\" has invalid tile geometry\n",fileName);
            TIFFClose(tif);
            return 1;
        }
        tileBufBytes = (uint64_t)tileBytes;
        tilesAcross = (x + tw - 1) / tw;
        tilesDown = (y + tl - 1) / tl;
        numTiles = tilesAcross * tilesDown;
    }
    const int32_t tileBatch = tiled ? (int32_t)((numTiles-1)/numWorkers+1) : 0;

    // Only start as many workers as there are strips or tiles to read. A
    // small image has just a few, and every extra worker opened the file
    // and selected its directory for nothing.
    const uint64_t workItems = tiled ? numTiles : stripsPerDir;
    const uint64_t perWorker = tiled ? (uint64_t)tileBatch : (uint64_t)batchSize;
    const int32_t activeWorkers = (int32_t)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)numWorkers, (workItems + perWorker - 1) / perWorker));
    #pragma omp parallel for num_threads(activeWorkers)
    for(w = 0; w < activeWorkers; w++){

        uint8_t outCounter = 0;
        TIFF* tif = openReaderTiff(fileName);
        while(!tif){
            tif = openReaderTiff(fileName);
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

        if(!tif) continue;   // err is already set; never hand libtiff a null handle
        // Tiles always decode into the buffer; strips only on the flip path,
        // since the no-flip path decodes each strip straight into the output.
        void* buffer = (tiled || flipXY) ? malloc(tiled ? tileBufBytes : xs*stripSize*bytes) : NULL;
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
        if(tiled){
            int64_t tStart = (int64_t)w*tileBatch;
            int64_t tEnd = tStart + tileBatch;
            if(tEnd > (int64_t)numTiles) tEnd = (int64_t)numTiles;
            int64_t failedTile = -1;
            if(!err && tStart < tEnd &&
               readTilesInto(tif, (uint8_t*)buffer, (int64_t)tileBufBytes, tiff,
                             (int64_t)x, (int64_t)y, (int64_t)bits, (int64_t)samples,
                             (int64_t)tw, (int64_t)tl, (int64_t)tilesAcross,
                             tStart, tEnd, 0, flipXY, &failedTile)){
                #pragma omp critical
                {
                    // No errBak: the scanline fallback cannot read tiled images.
                    err = 1;
                    sprintf(errString,"Thread %d: Tile %ld cannot be read\n",w,(long)failedTile);
                }
            }
        }
        else for (int64_t i = (w*batchSize); i < (w+1)*batchSize; i++)
        {
            if(i*stripSize >= y || err) break;
            // No-flip: decode the strip straight into the output, as the 3D
            // path does. Flip: decode into the buffer and transpose below.
            void* stripDst = flipXY ? buffer : static_cast<void*>(static_cast<uint8_t*>(tiff) + i*stripSize*xs*bytes);
            int64_t cBytes = TIFFReadEncodedStrip(tif, i, stripDst, stripSize*xs*bytes);
            if(cBytes < 0){
                #pragma omp critical
                {
                    errBak = 1;
                    err = 1;
                    sprintf(errString,"Thread %d: Strip %ld cannot be read\n",w,(long)i);
                }
                break;
            }
            if(!flipXY){
                continue;
            }
            // Same cache-blocked transpose as the 3D path, single image so
            // the slice offset is 0. samples==1 keeps the grayscale path.
            int64_t stripRows = y - (i*stripSize);
            if(stripRows > (int64_t)stripSize) stripRows = (int64_t)stripSize;
            int64_t rowOff = i*stripSize;
            if(samples == 1){
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
            } else {
                switch(bits){
                    case 8:
                        flipTransposeStripRGB<uint8_t>((uint8_t*)tiff,(uint8_t*)buffer,x,y,stripRows,rowOff,0,samples);
                        break;
                    case 16:
                        flipTransposeStripRGB<uint16_t>((uint16_t*)tiff,(uint16_t*)buffer,x,y,stripRows,rowOff,0,samples);
                        break;
                    case 32:
                        flipTransposeStripRGB<float>((float*)tiff,(float*)buffer,x,y,stripRows,rowOff,0,samples);
                        break;
                    case 64:
                        flipTransposeStripRGB<double>((double*)tiff,(double*)buffer,x,y,stripRows,rowOff,0,samples);
                        break;
                }
            }
        }
        free(buffer);
        TIFFClose(tif);
    }
    // Close the outer handle as well. It was only needed for the tag reads
    // above (each worker opens its own), and leaving it open leaked a handle
    // per 2D read -- harmless on Linux but it keeps the file locked on Windows.
    TIFFClose(tif);

    if(err) {
        if(errBak) return readTiffParallel2DBak(x, y, z, fileName, tiff, bits, startSlice, flipXY);
        else printf("%s", errString);
    }
	return err;
}


// Seek to an absolute byte offset; 0 on success. fseek takes a long, which is
// 32 bits on Windows, so offsets past 2 GB need the 64-bit variants.
static int seekFile(FILE* fp, uint64_t offset){
#ifdef _WIN32
    return _fseeki64(fp, (long long)offset, SEEK_SET);
#else
    return fseeko(fp, (off_t)offset, SEEK_SET);
#endif
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
		fclose(fp);
		err = 1;
		return err;
	}
    uint64_t* offsets = NULL;
    TIFFGetField(tif, TIFFTAG_STRIPOFFSETS, &offsets);
    if(!offsets){
        printf("Could not get offsets from the tiff file\n");
        TIFFClose(tif);
        fclose(fp);
        err = 1;
        return err;
    }
    const uint64_t offset = offsets[0];

    // ImageJ RGB stacks are chunky interleaved (spp=3), so every size below
    // scales by samples. Grayscale (samples==1) is unchanged.
    uint16_t spp = 1;
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    const uint64_t samples = spp;

    // The raw fread below bypasses libtiff, so swap bytes only when the file's
    // byte order differs from the host's (ImageJ itself writes big-endian, but
    // e.g. tifffile writes ImageJ-format files little-endian by default).
    const int needSwap = TIFFIsByteSwapped(tif);

    TIFFClose(tif);

    uint64_t bytes = bits/8;
    // The data is one contiguous block, so a z-range is just an offset into it.
    if(seekFile(fp, offset + startSlice*x*y*samples*bytes)){
        printf("File \"%s\": cannot seek to the image data\n",fileName);
        fclose(fp);
        err = 1;
        return err;
    }
    //#pragma omp parallel for
    /*
    for(uint64_t i = 0; i < z; i++){
    uint64_t cOffset = x*y*bytes*i;
    //pread(fd,tiff+cOffset,x*y*bytes,offset+cOffset);
    read(fd,tiff+cOffset,x*y*bytes);
    }*/
    // fread at most INT_MAX bytes at a time. A short read means the file ends
    // before its image data does (e.g. an interrupted copy), so report that
    // instead of retrying forever or returning the unread part of the buffer.
    const uint64_t tBytes = x*y*z*samples*bytes;
    uint64_t chunk = 0;
    while(chunk < tBytes){
        const uint64_t want = std::min<uint64_t>(tBytes - chunk, INT_MAX);
        const uint64_t got = fread(static_cast<uint8_t*>(tiff)+chunk,1,want,fp);
        chunk += got;
        if(got < want) break;
    }
    fclose(fp);
    if(chunk < tBytes){
        printf("File \"%s\" ends before its image data does\n",fileName);
        err = 1;
        return err;
    }
    // Swap endianess for types greater than 8 bits when the file's byte order
    // differs from the host's.
    if(bits > 8 && needSwap){
        #pragma omp parallel for
        for(uint64_t i = 0; i < x*y*z*samples; i++){
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
        uint64_t size = x*y*z*samples*(bits/8);
        void* tiffC = malloc(size);
        memcpy(tiffC,tiff,size);
        if(samples == 1){
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
        }
        else{
            // De-interleave each chunky slice into the column-major (y,x,c,z)
            // MATLAB layout, same mapping as the strip readers use.
            #pragma omp parallel for
            for(uint64_t k = 0; k < z; k++){
                const uint64_t off = k*x*y*samples;
                switch(bits){
                    case 8:
                        flipTransposeStripRGB<uint8_t>((uint8_t*)tiff, (uint8_t*)tiffC + off, x, y, y, 0, off, samples);
                        break;
                    case 16:
                        flipTransposeStripRGB<uint16_t>((uint16_t*)tiff, (uint16_t*)tiffC + off, x, y, y, 0, off, samples);
                        break;
                    case 32:
                        flipTransposeStripRGB<float>((float*)tiff, (float*)tiffC + off, x, y, y, 0, off, samples);
                        break;
                    case 64:
                        flipTransposeStripRGB<double>((double*)tiff, (double*)tiffC + off, x, y, y, 0, off, samples);
                        break;
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
	uint16_t spp = 1, planar = 1;
	TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
	TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);
	uint64_t samples = spp;   // 1=grayscale, 3=RGB, 4=RGBA
	const bool tiled = TIFFIsTiled(tif);
	TIFFClose(tif);

	// Planar (separate-plane) multi-sample TIFFs are not supported yet; reject
	// clearly rather than silently mis-reading. Chunky RGB/RGBA are handled.
	if(samples > 1 && planar == 2){
		printf("Planar (PLANARCONFIG=2) RGB TIFFs are not supported by cpp-tiff\n");
		return NULL;
	}

	// Check if image is an imagej image with imagej metadata. The ImageJ reader
	// pulls the pixels off disk as one contiguous block, which is not how tiled
	// data is laid out, so tiled files go through the normal readers instead.
	uint8_t imageJIm = 0;
	if(!tiled && isImageJIm(fileName)){
		imageJIm = 1;
		uint64_t tempZ = imageJImGetZ(fileName);
		if(tempZ) z = tempZ;
	}

    if(zRange.size()){
        if(zRange.size() == 2){
            startSlice = zRange[0];
            z = zRange[1] - zRange[0];   // number of slices to read
        }
        else{
            startSlice = zRange[0];
            z = 1;                        // single slice
        }
    }


	// Bytes are moved purely by bit width, so one allocation covers every
	// sample format.
	if(bits != 8 && bits != 16 && bits != 32 && bits != 64) return NULL;
	const bool allocated = !tiff;
	if(allocated){
		tiff = malloc(x*y*z*samples*(bits/8));
		if(!tiff) return NULL;
	}
	uint8_t err;
	if(imageJIm)    err = readTiffParallelImageJ(x,y,z,fileName, tiff, bits, startSlice, stripSize, flipXY);
	else if(z <= 1) err = readTiffParallel2D(x,y,z,fileName, tiff, bits, startSlice, stripSize, flipXY);
	else            err = readTiffParallel(x,y,z,fileName, tiff, bits, startSlice, stripSize, flipXY);
	// A failed read leaves part of the buffer unwritten, so report it instead of
	// returning partial data. The reader has already printed why.
	if(err){
		if(allocated) free(tiff);
		return NULL;
	}
	return tiff;
}

void* readTiffParallelWrapper(const char* fileName, const std::vector<uint64_t> &zRange)
{
	return readTiffParallelWrapperHelper(fileName,NULL,1,zRange);
}

void* readTiffParallelWrapperNoXYFlip(const char* fileName, const std::vector<uint64_t> &zRange)
{
	return readTiffParallelWrapperHelper(fileName,NULL,0,zRange);
}

// tTiff doesn't matter as tiff is set in the function
void readTiffParallelWrapperSet(const char* fileName, void* tiff){
	void* tTiff = readTiffParallelWrapperHelper(fileName,tiff,0);
}
