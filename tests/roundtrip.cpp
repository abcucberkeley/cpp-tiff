// Minimal smoke test for the cpp-tiff library: write small random volumes, read
// them back, and check the bytes survive the round trip. Covers every bit depth /
// sample format and each compression, plus the XY-flip reader (MATLAB layout).
//
// Exits 0 if every case passes, 1 otherwise, so CTest reports pass/fail.
// Usage: roundtripTest [output_dir]   (defaults to the current directory)
//
// Progress is written (flushed) to stderr before each operation so that if a
// build segfaults, the CTest log shows exactly which step/dtype/compression died.
#include <algorithm>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <omp.h>
#ifndef _WIN32
#include <chrono>
#include <csignal>
#include <pwd.h>
#include <spawn.h>
#include <thread>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/wait.h>
extern char** environ;
#endif

#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"
#include "tiffio.h"
#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif

// The test's own files, by UTF-8 path, as the library takes paths: on Windows through the wide-character API
// (converted here, independently of the library), so that a file the library wrote under a name in another
// script is found only where that name says
static FILE* openUtf8(const std::string& path, const char* mode){
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> utf16;
    return _wfopen(utf16.from_bytes(path).c_str(), utf16.from_bytes(mode).c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

struct Case { const char* name; uint64_t bits; uint16_t sampleFormat; };

static void step(const char* name, const char* comp, const char* what){
    std::fprintf(stderr, "  [%s/%s] %s\n", name, comp, what);
    std::fflush(stderr);
}

// ---- ImageJ-format fixture writer -----------------------------------------
// A single-IFD TIFF describing slice 0, every slice's raw data contiguous after
// it, and an ImageJ description carrying images=Z -- the layout ImageJ's fast
// save produces, which the library reads through its ImageJ path. The library's
// own writer cannot produce this layout, so the test writes it by hand. Fields
// are emitted byte-by-byte so both byte orders can be produced on any host.
static void ijPut16(std::vector<uint8_t>& v, uint16_t s, bool be){
    if(be){ v.push_back((uint8_t)(s >> 8)); v.push_back((uint8_t)s); }
    else  { v.push_back((uint8_t)s); v.push_back((uint8_t)(s >> 8)); }
}
static void ijPut32(std::vector<uint8_t>& v, uint32_t l, bool be){
    if(be) for(int i = 3; i >= 0; i--) v.push_back((uint8_t)(l >> (8*i)));
    else   for(int i = 0; i <  4; i++) v.push_back((uint8_t)(l >> (8*i)));
}
static void ijEntry(std::vector<uint8_t>& v, uint16_t tag, uint16_t typ, uint32_t cnt, bool be){
    ijPut16(v, tag, be); ijPut16(v, typ, be); ijPut32(v, cnt, be);
}
static bool writeImageJTiff(const std::string& path, const uint8_t* data,
                            uint64_t x, uint64_t y, uint64_t z,
                            uint64_t spp, uint64_t bits, bool bigEndian){
    const uint64_t es = bits/8;
    std::string desc = "ImageJ=1.11a\nimages=" + std::to_string(z) +
                       "\nslices=" + std::to_string(z) + "\n";
    desc.push_back('\0');
    if(desc.size() % 2) desc.push_back('\0');

    const uint32_t nEntries = 11;
    const uint32_t auxOff = 8 + 2 + nEntries*12 + 4;
    const uint32_t bpsLen = (spp > 1) ? (uint32_t)(spp*2) : 0;
    const uint32_t descOff = auxOff + bpsLen;
    uint32_t pixOff = descOff + (uint32_t)desc.size();
    pixOff += pixOff % 2;

    std::vector<uint8_t> v;
    v.push_back(bigEndian ? 'M' : 'I'); v.push_back(bigEndian ? 'M' : 'I');
    ijPut16(v, 42, bigEndian); ijPut32(v, 8, bigEndian);
    ijPut16(v, nEntries, bigEndian);
    // A SHORT value sits in the first two bytes of the 4-byte value field.
    auto shortVal = [&](uint16_t s){ ijPut16(v, s, bigEndian); v.push_back(0); v.push_back(0); };
    auto longVal  = [&](uint32_t l){ ijPut32(v, l, bigEndian); };
    ijEntry(v,256,3,1,bigEndian); shortVal((uint16_t)x);            // ImageWidth
    ijEntry(v,257,3,1,bigEndian); shortVal((uint16_t)y);            // ImageLength
    ijEntry(v,258,3,(uint32_t)spp,bigEndian);                       // BitsPerSample
    if(spp == 1) shortVal((uint16_t)bits); else longVal(auxOff);
    ijEntry(v,259,3,1,bigEndian); shortVal(1);                      // Compression: none
    ijEntry(v,262,3,1,bigEndian); shortVal(spp == 3 ? 2 : 1);       // Photometric
    ijEntry(v,270,2,(uint32_t)desc.size(),bigEndian); longVal(descOff);
    ijEntry(v,273,4,1,bigEndian); longVal(pixOff);                  // StripOffsets
    ijEntry(v,277,3,1,bigEndian); shortVal((uint16_t)spp);          // SamplesPerPixel
    ijEntry(v,278,3,1,bigEndian); shortVal((uint16_t)y);            // RowsPerStrip
    ijEntry(v,279,4,1,bigEndian); longVal((uint32_t)(x*y*spp*es));  // StripByteCounts
    ijEntry(v,284,3,1,bigEndian); shortVal(1);                      // PlanarConfig: chunky
    ijPut32(v, 0, bigEndian);                                       // next IFD
    if(spp > 1) for(uint64_t i = 0; i < spp; i++) ijPut16(v, (uint16_t)bits, bigEndian);
    v.insert(v.end(), desc.begin(), desc.end());
    while(v.size() < pixOff) v.push_back(0);

    const uint64_t nElems = x*y*z*spp;
    if(es == 1 || !bigEndian){
        v.insert(v.end(), data, data + nElems*es);
    } else {
        for(uint64_t i = 0; i < nElems; i++)
            for(int b = (int)es - 1; b >= 0; b--) v.push_back(data[i*es + b]);
    }
    FILE* f = openUtf8(path, "wb");
    if(!f) return false;
    const size_t w = std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return w == v.size();
}

// ---- Plain multi-page fixture writer -----------------------------------------
// Classic TIFF, uncompressed, one strip per page, chunky samples; little-endian
// unless bigEndian (data is in host order and swapped as written). The library's
// writer picks its own strips and only writes little-endian, so these fixtures
// are written here.
static bool writePlainTiff(const std::string& path, const uint8_t* data,
                           uint64_t x, uint64_t y, uint64_t z, uint64_t spp, uint64_t bits,
                           bool bigEndian = false){
    const bool be = bigEndian;
    const uint64_t es = bits/8;
    const uint64_t pageBytes = x*y*spp*(bits/8);
    const uint32_t nEntries = 10;
    const uint32_t ifdSize = 2 + nEntries*12 + 4;
    const uint32_t bpsLen = (spp > 1) ? (uint32_t)(spp*2) : 0;
    std::vector<uint8_t> v;
    v.push_back(be ? 'M' : 'I'); v.push_back(be ? 'M' : 'I'); ijPut16(v, 42, be); ijPut32(v, 8, be);
    for(uint64_t k = 0; k < z; k++){
        const uint32_t ifdOff = (uint32_t)v.size();
        const uint32_t bpsOff = ifdOff + ifdSize;
        const uint32_t pixOff = bpsOff + bpsLen;
        const uint32_t next = (k + 1 < z) ? (uint32_t)(pixOff + pageBytes) : 0;
        auto shortVal = [&](uint16_t sv){ ijPut16(v, sv, be); v.push_back(0); v.push_back(0); };
        auto longVal  = [&](uint32_t lv){ ijPut32(v, lv, be); };
        ijPut16(v, nEntries, be);
        ijEntry(v,256,3,1,be); shortVal((uint16_t)x);                  // ImageWidth
        ijEntry(v,257,3,1,be); shortVal((uint16_t)y);                  // ImageLength
        ijEntry(v,258,3,(uint32_t)spp,be);                             // BitsPerSample
        if(spp == 1) shortVal((uint16_t)bits); else longVal(bpsOff);
        ijEntry(v,259,3,1,be); shortVal(1);                            // Compression: none
        ijEntry(v,262,3,1,be); shortVal(spp == 3 ? 2 : 1);             // Photometric
        ijEntry(v,273,4,1,be); longVal(pixOff);                        // StripOffsets
        ijEntry(v,277,3,1,be); shortVal((uint16_t)spp);                // SamplesPerPixel
        ijEntry(v,278,3,1,be); shortVal((uint16_t)y);                  // RowsPerStrip
        ijEntry(v,279,4,1,be); longVal((uint32_t)pageBytes);           // StripByteCounts
        ijEntry(v,284,3,1,be); shortVal(1);                            // PlanarConfig: chunky
        ijPut32(v, next, be);                                          // next IFD
        if(spp > 1) for(uint64_t i = 0; i < spp; i++) ijPut16(v, (uint16_t)bits, be);
        if(!be || es == 1) v.insert(v.end(), data + k*pageBytes, data + (k + 1)*pageBytes);
        else for(uint64_t i = 0; i < pageBytes; i += es)
            for(int b = (int)es - 1; b >= 0; b--) v.push_back(data[k*pageBytes + i + b]);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t w = std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return w == v.size();
}

// Little-endian uncompressed grayscale with several strips per page stored in
// reverse order with gaps between them, a legal layout that cannot be read as
// one contiguous block, so the readers must fall back to libtiff per strip.
static bool writeReversedStripsTiff(const std::string& path, const uint8_t* data,
                                    uint64_t x, uint64_t y, uint64_t z, uint64_t bits, uint64_t rps){
    const uint64_t es = bits/8, rowBytes = x*es, nStrips = (y + rps - 1)/rps, gap = 16;
    const uint32_t nEntries = 10;
    std::vector<uint8_t> v;
    v.push_back('I'); v.push_back('I'); ijPut16(v, 42, false); ijPut32(v, 8, false);
    for(uint64_t k = 0; k < z; k++){
        const uint32_t ifdOff = (uint32_t)v.size();
        const uint32_t offsOff = ifdOff + 2 + nEntries*12 + 4, cntsOff = offsOff + 4*(uint32_t)nStrips;
        std::vector<uint32_t> offs(nStrips), cnts(nStrips);
        uint32_t pos = cntsOff + 4*(uint32_t)nStrips;
        for(uint64_t i = nStrips; i-- > 0;){
            cnts[i] = (uint32_t)(std::min<uint64_t>(rps, y - i*rps)*rowBytes);
            offs[i] = pos; pos += cnts[i] + gap;
        }
        auto shortVal = [&](uint16_t sv){ ijPut16(v, sv, false); v.push_back(0); v.push_back(0); };
        auto longVal  = [&](uint32_t lv){ ijPut32(v, lv, false); };
        ijPut16(v, nEntries, false);
        ijEntry(v,256,3,1,false); shortVal((uint16_t)x);                  // ImageWidth
        ijEntry(v,257,3,1,false); shortVal((uint16_t)y);                  // ImageLength
        ijEntry(v,258,3,1,false); shortVal((uint16_t)bits);               // BitsPerSample
        ijEntry(v,259,3,1,false); shortVal(1);                            // Compression: none
        ijEntry(v,262,3,1,false); shortVal(1);                            // Photometric
        ijEntry(v,273,4,(uint32_t)nStrips,false); longVal(offsOff);       // StripOffsets
        ijEntry(v,277,3,1,false); shortVal(1);                            // SamplesPerPixel
        ijEntry(v,278,3,1,false); shortVal((uint16_t)rps);                // RowsPerStrip
        ijEntry(v,279,4,(uint32_t)nStrips,false); longVal(cntsOff);       // StripByteCounts
        ijEntry(v,284,3,1,false); shortVal(1);                            // PlanarConfig: chunky
        ijPut32(v, (k + 1 < z) ? pos : 0, false);                         // next IFD
        for(uint32_t o : offs) ijPut32(v, o, false);
        for(uint32_t c : cnts) ijPut32(v, c, false);
        for(uint64_t i = nStrips; i-- > 0;){
            const uint8_t* src = data + k*y*rowBytes + i*rps*rowBytes;
            v.insert(v.end(), src, src + cnts[i]);
            v.insert(v.end(), (size_t)gap, (uint8_t)0);
        }
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t w = std::fwrite(v.data(), 1, v.size(), f);
    std::fclose(f);
    return w == v.size();
}

// ---- libtiff fixture writer ---------------------------------------------------
// Files the library's writer cannot produce, written by the libtiff built into
// it: planar color (PLANARCONFIG=2: every strip or tile of the first color,
// then of the next, one sample per pixel), any libtiff codec, any RowsPerStrip
// (rps) or tile size, big-endian, a predictor, the other bit order (fillOrder),
// and another RowsPerStrip for the directories after the first (laterRps, or
// rps when 0). data is chunky (z, y, x, c) in host order.
static bool writeLibtiff(const std::string& path, const uint8_t* data, uint64_t x, uint64_t y, uint64_t z,
                         uint64_t spp, uint64_t bits, uint16_t fmt, uint16_t comp, uint32_t rps, uint32_t tile,
                         bool planar, bool bigEndian = false, uint16_t predictor = PREDICTOR_NONE,
                         uint16_t fillOrder = FILLORDER_MSB2LSB, uint32_t laterRps = 0){
    TIFF* tif = TIFFOpen(path.c_str(), bigEndian ? "wb" : "wl");
    if(!tif) return false;
    const uint64_t es = bits/8, planes = planar ? spp : 1, px = planar ? es : spp*es;   // px: bytes per pixel of one plane
    std::vector<uint8_t> img(x*y*px), chunk;
    bool ok = true;
    for(uint64_t k = 0; k < z && ok; k++){
        const uint32_t pageRps = (k && laterRps) ? laterRps : rps;
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (uint32_t)x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (uint32_t)y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (uint16_t)bits);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)spp);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, fmt);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        if(spp == 2 || spp == 4){
            const uint16_t extra = spp == 4 ? EXTRASAMPLE_UNASSALPHA : EXTRASAMPLE_UNSPECIFIED;
            TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, (uint16_t)1, &extra);
        }
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, planar ? PLANARCONFIG_SEPARATE : PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, comp);
        if(predictor != PREDICTOR_NONE) TIFFSetField(tif, TIFFTAG_PREDICTOR, predictor);
        if(fillOrder != FILLORDER_MSB2LSB) TIFFSetField(tif, TIFFTAG_FILLORDER, fillOrder);
        if(tile){ TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile); TIFFSetField(tif, TIFFTAG_TILELENGTH, tile); }
        else TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, pageRps);
        const uint8_t* slice = data + k*x*y*spp*es;
        for(uint64_t p = 0; p < planes && ok; p++){
            if(planar) for(uint64_t i = 0; i < x*y; i++) std::memcpy(&img[i*es], slice + (i*spp + p)*es, es);
            else std::memcpy(img.data(), slice, img.size());
            // libtiff swaps big-endian data in place, so it always gets a copy.
            if(tile){
                for(uint64_t ty = 0; ty < y && ok; ty += tile)
                    for(uint64_t tx = 0; tx < x && ok; tx += tile){
                        chunk.assign((size_t)tile*tile*px, 0);
                        for(uint64_t r = 0; r < tile && ty + r < y; r++)
                            std::memcpy(&chunk[r*tile*px], &img[((ty + r)*x + tx)*px], std::min<uint64_t>(tile, x - tx)*px);
                        ok = TIFFWriteEncodedTile(tif, TIFFComputeTile(tif, (uint32_t)tx, (uint32_t)ty, 0, (uint16_t)p),
                                                  chunk.data(), (tmsize_t)chunk.size()) >= 0;
                    }
            }
            else{
                const uint64_t rows = std::min<uint64_t>(pageRps, y), strips = (y + rows - 1)/rows;
                for(uint64_t s = 0; s < strips && ok; s++){
                    const uint64_t n = std::min<uint64_t>(rows, y - s*rows)*x*px;
                    chunk.assign(img.begin() + s*rows*x*px, img.begin() + s*rows*x*px + n);
                    ok = TIFFWriteEncodedStrip(tif, (uint32_t)(p*strips + s), chunk.data(), (tmsize_t)n) == (tmsize_t)n;
                }
            }
        }
        ok = ok && TIFFWriteDirectory(tif);
    }
    TIFFClose(tif);
    return ok;
}

// ---- Stacks of directories unlike each other ------------------------------
// One directory a slice, each as its MixedDir says (size, pixels, rows per
// strip or tile size, codec, planar), its data chunky (y, x, c) in host order.
struct MixedDir { uint32_t x, y, bits, spp, rps, tile; uint16_t comp; bool planar; };
static bool writeMixedTiff(const std::string& path, const std::vector<MixedDir>& dirs, const std::vector<std::vector<uint8_t>>& data){
    TIFF* tif = TIFFOpen(path.c_str(), "w8");
    if(!tif) return false;
    bool ok = true;
    for(size_t k = 0; k < dirs.size() && ok; k++){
        const MixedDir& d = dirs[k];
        const uint64_t es = d.bits/8, planes = d.planar ? d.spp : 1, px = d.planar ? es : d.spp*es;
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, d.x);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, d.y);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, (uint16_t)d.bits);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)d.spp);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, d.spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, d.planar ? PLANARCONFIG_SEPARATE : PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, d.comp);
        if(d.tile){ TIFFSetField(tif, TIFFTAG_TILEWIDTH, d.tile); TIFFSetField(tif, TIFFTAG_TILELENGTH, d.tile); }
        else TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, d.rps);
        std::vector<uint8_t> img((uint64_t)d.x*d.y*px), chunk;
        for(uint64_t p = 0; p < planes && ok; p++){
            if(d.planar) for(uint64_t i = 0; i < (uint64_t)d.x*d.y; i++) std::memcpy(&img[i*es], &data[k][(i*d.spp + p)*es], es);
            else img = data[k];
            if(d.tile){
                for(uint32_t ty = 0; ty < d.y && ok; ty += d.tile)
                    for(uint32_t tx = 0; tx < d.x && ok; tx += d.tile){
                        chunk.assign((uint64_t)d.tile*d.tile*px, 0);
                        for(uint32_t r = 0; r < d.tile && ty + r < d.y; r++)
                            std::memcpy(&chunk[(uint64_t)r*d.tile*px], &img[((uint64_t)(ty + r)*d.x + tx)*px], std::min(d.tile, d.x - tx)*px);
                        ok = TIFFWriteEncodedTile(tif, TIFFComputeTile(tif, tx, ty, 0, (uint16_t)p), chunk.data(), (tmsize_t)chunk.size()) >= 0;
                    }
            }
            else{
                const uint32_t rows = std::min(d.rps, d.y), strips = (d.y + rows - 1)/rows;
                for(uint32_t st = 0; st < strips && ok; st++){
                    const uint64_t n = (uint64_t)std::min(rows, d.y - st*rows)*d.x*px;
                    ok = TIFFWriteEncodedStrip(tif, (uint32_t)(p*strips + st), &img[(uint64_t)st*rows*d.x*px], (tmsize_t)n) == (tmsize_t)n;
                }
            }
        }
        ok = ok && TIFFWriteDirectory(tif);
    }
    TIFFClose(tif);
    return ok;
}

// Offset and byte count of one strip, for damaging it on purpose. Handles the
// little-endian classic and BigTIFF files this test writes.
static bool stripLocation(const std::string& path, uint64_t page, uint64_t strip,
                          uint64_t* off, uint64_t* cnt){
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) return false;
    auto rd = [&](uint64_t pos, uint8_t* dst, size_t n){
        return std::fseek(f, (long)pos, SEEK_SET) == 0 && std::fread(dst, 1, n, f) == n; };
    auto le = [](const uint8_t* b, int n){ uint64_t val = 0; for(int i = n - 1; i >= 0; i--) val = (val << 8) | b[i]; return val; };
    uint8_t h[16];
    bool ok = rd(0, h, 16) && h[0] == 'I' && h[1] == 'I';
    const bool big = ok && h[2] == 43;
    const int cntSz = big ? 8 : 2, entSz = big ? 20 : 12, offSz = big ? 8 : 4;
    uint64_t ifd = big ? le(h + 8, 8) : le(h + 4, 4);
    uint8_t b[20];
    for(uint64_t pg = 0; ok && pg < page; pg++){
        ok = rd(ifd, b, cntSz) && rd(ifd + cntSz + le(b, cntSz)*entSz, b, offSz);
        ifd = le(b, offSz);
    }
    const uint64_t n = (ok && rd(ifd, b, cntSz)) ? le(b, cntSz) : 0;
    bool haveOff = false, haveCnt = false;
    for(uint64_t e = 0; ok && e < n; e++){
        ok = rd(ifd + cntSz + e*entSz, b, entSz);
        const uint64_t tag = le(b, 2), typ = le(b + 2, 2), count = big ? le(b + 4, 8) : le(b + 4, 4);
        if(!ok || (tag != 273 && tag != 279)) continue;
        const int sz = typ == 16 ? 8 : typ == 4 ? 4 : typ == 3 ? 2 : 0;
        if(!sz || strip >= count){ ok = false; break; }
        const uint8_t* field = b + (big ? 12 : 8);
        uint8_t val[8];
        if(count*sz <= (uint64_t)offSz) std::memcpy(val, field + strip*sz, sz);
        else ok = rd(le(field, offSz) + strip*sz, val, sz);
        if(tag == 273){ *off = le(val, sz); haveOff = true; } else { *cnt = le(val, sz); haveCnt = true; }
    }
    std::fclose(f);
    return ok && haveOff && haveCnt;
}

// Drop the last `cut` bytes of a file, as an interrupted copy would.
static bool truncateFile(const std::string& path, uint64_t cut){
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) return false;
    std::vector<uint8_t> v;
    uint8_t buf[65536];
    size_t n;
    while((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    if(cut >= v.size()) return false;
    f = std::fopen(path.c_str(), "wb");
    if(!f) return false;
    const size_t keep = v.size() - cut;
    const size_t w = std::fwrite(v.data(), 1, keep, f);
    std::fclose(f);
    return w == keep;
}

static std::vector<uint8_t> readFileBytes(const std::string& path){
    std::vector<uint8_t> v;
    FILE* f = std::fopen(path.c_str(), "rb");
    if(!f) return v;
    uint8_t buf[65536];
    size_t n;
    while((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    return v;
}

// The first value of a SHORT, LONG or LONG8 tag in a file's first directory,
// read from the bytes so it checks what is on disk rather than what libtiff
// makes of it. False if the tag is absent. Little-endian TIFF or BigTIFF.
static bool firstTag(const std::vector<uint8_t>& f, uint64_t tag, uint64_t* value){
    auto u = [&](uint64_t off, int n){
        uint64_t v = 0;
        for(int i = n - 1; i >= 0; i--) v = (v << 8) | f[off + i];
        return v;
    };
    if(f.size() < 16 || f[0] != 'I' || f[1] != 'I') return false;
    const bool big = u(2, 2) == 43;
    const uint64_t ifd = big ? u(8, 8) : u(4, 4);
    if(ifd + 8 > f.size()) return false;
    const uint64_t n = big ? u(ifd, 8) : u(ifd, 2), size = big ? 20 : 12, first = ifd + (big ? 8 : 2);
    for(uint64_t e = 0; e < n && first + (e + 1)*size <= f.size(); e++){
        const uint64_t p = first + e*size;
        if(u(p, 2) != tag) continue;
        const uint64_t type = u(p + 2, 2), count = big ? u(p + 4, 8) : u(p + 4, 4), val = p + (big ? 12 : 8);
        const int width = type == 3 ? 2 : type == 4 ? 4 : type == 16 ? 8 : 0;
        if(!width || !count) return false;
        // Values that do not fit in the entry are stored elsewhere, at the offset it holds.
        const uint64_t at = count*width <= (big ? 8u : 4u) ? val : (big ? u(val, 8) : u(val, 4));
        if(at + width > f.size()) return false;
        *value = u(at, width);
        return true;
    }
    return false;
}

// RowsPerStrip of a file's first directory (0 if absent).
static uint64_t firstRowsPerStrip(const std::vector<uint8_t>& f){
    uint64_t v = 0;
    return firstTag(f, 278, &v) ? v : 0;
}

// Column-major MATLAB layout (y, x[, c], z) of a chunky row-major source.
static std::vector<uint8_t> flipped(const uint8_t* src, uint64_t x, uint64_t y, uint64_t z,
                                    uint64_t spp, uint64_t es){
    std::vector<uint8_t> out(x*y*z*spp*es);
    for(uint64_t k = 0; k < z; k++)
        for(uint64_t i = 0; i < y; i++)
            for(uint64_t j = 0; j < x; j++)
                for(uint64_t c = 0; c < spp; c++)
                    std::memcpy(&out[(i + j*y + c*x*y + k*x*y*spp)*es],
                                src + (((k*y + i)*x + j)*spp + c)*es, es);
    return out;
}

// The scanline fallbacks are internal: the readers only use them when a strip
// fails to decode, so valid files never reach them. Call them directly.
uint8_t readTiffParallelBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY);
uint8_t readTiffParallel2DBak(uint64_t x, uint64_t y, uint64_t z, const char* fileName, void* tiff, uint64_t bits, uint64_t startSlice, uint8_t flipXY);

// Run one fallback read into a poisoned, guarded buffer: unwritten bytes keep
// the poison and any write past the end changes the guard.
static bool fallbackMatches(bool twoD, const std::string& path, uint64_t x, uint64_t y, uint64_t z,
                            uint64_t spp, uint64_t bits, uint64_t startSlice, bool flip,
                            const std::vector<uint8_t>& expected){
    const size_t guard = 4096;
    std::vector<uint8_t> out(expected.size() + guard, 0xA5);
    const uint8_t err = twoD ? readTiffParallel2DBak(x, y, z, path.c_str(), out.data(), bits, startSlice, flip)
                             : readTiffParallelBak(x, y, z, path.c_str(), out.data(), bits, startSlice, flip);
    bool guardOK = true;
    for(size_t i = expected.size(); i < out.size(); i++) guardOK = guardOK && out[i] == 0xA5;
    (void)spp;
    return !err && guardOK && std::memcmp(out.data(), expected.data(), expected.size()) == 0;
}

#ifndef _WIN32
// ---- Writes cut short (see main): a stack whose slice k is cutSlice(k), a
// check that a file holds only whole slices of it, and the test run as a
// child that writes it (--write-child), to be killed
static const char* selfPath = "";
static std::vector<uint16_t> cutSlice(uint64_t x, uint64_t y, uint64_t k){
    std::vector<uint16_t> d(x*y);
    for(uint64_t r = 0; r < y; r++)
        for(uint64_t c = 0; c < x; c++) d[r*x + c] = (uint16_t)((k*1009 + r*31 + c*7 + ((r*c) % 13) + 13) & 0xFFFF);
    return d;
}
static std::vector<uint16_t> cutStack(uint64_t x, uint64_t y, uint64_t z){
    std::vector<uint16_t> d(x*y*z);
    for(uint64_t k = 0; k < z; k++){ const std::vector<uint16_t> s = cutSlice(x, y, k); std::memcpy(&d[k*x*y], s.data(), s.size()*2); }
    return d;
}
// Whether file f holds only whole slices of cutStack's stack: libtiff walks
// directories whose every strip is in the file, getTiffInfo finds as many
// slices, and each reads back as written (found: how many; no file, or a file
// with no directory yet, holds 0)
static bool onlyWholeSlices(const std::string& f, uint64_t x, uint64_t y, uint64_t maxZ, uint64_t& found, std::string& why){
    found = 0;
    FILE* fp = std::fopen(f.c_str(), "rb");
    if(!fp) return true;
    std::fseek(fp, 0, SEEK_END);
    const uint64_t size = (uint64_t)std::ftell(fp);
    std::fclose(fp);
    const TIFFErrorHandler eh = TIFFSetErrorHandler(NULL), wh = TIFFSetWarningHandler(NULL);
    uint64_t dirs = 0;
    bool inFile = true;
    TIFF* t = TIFFOpen(f.c_str(), "r");
    if(t){
        do{
            for(uint32_t k = 0; k < TIFFNumberOfStrips(t) && inFile; k++){
                const uint64_t o = TIFFGetStrileOffset(t, k), n = TIFFGetStrileByteCount(t, k);
                inFile = n && o + n <= size;
            }
            if(!inFile) break;
            dirs++;
        } while(TIFFReadDirectory(t));
        TIFFClose(t);
    }
    TIFFSetErrorHandler(eh);
    TIFFSetWarningHandler(wh);
    if(!inFile){ why = "directory " + std::to_string(dirs) + " has a strip that is not in the file"; return false; }
    if(!dirs) return true;
    TiffInfo info;
    if(getTiffInfo(f.c_str(), info)){ why = "getTiffInfo failed, where libtiff walks " + std::to_string(dirs) + " directories"; return false; }
    if(info.z != dirs || info.z > maxZ || info.x != x || info.y != y){
        why = "getTiffInfo finds " + std::to_string(info.z) + " slices, libtiff " + std::to_string(dirs);
        return false;
    }
    std::vector<uint16_t> got(x*y);
    for(uint64_t k = 0; k < info.z; k++)
        if(!readTiffParallelInfo(f.c_str(), info, got.data(), 0, {k}) || got != cutSlice(x, y, k)){
            why = "slice " + std::to_string(k) + " does not read back as written";
            return false;
        }
    found = info.z;
    return true;
}
// Runs this test as a child writing cutStack's stack to f in mode, killed
// (SIGKILL) after killAfter seconds unless that is negative; returns the
// seconds it ran (negative if it could not run, or ran to the end and failed)
static double runChild(const char* self, const std::string& f, const char* mode, uint64_t x, uint64_t y, uint64_t z, double killAfter){
    const std::string sx = std::to_string(x), sy = std::to_string(y), sz = std::to_string(z);
    const char* args[] = {self, "--write-child", f.c_str(), mode, sx.c_str(), sy.c_str(), sz.c_str(), NULL};
    pid_t pid = 0;
    const auto t0 = std::chrono::steady_clock::now();
    if(posix_spawn(&pid, self, NULL, NULL, (char* const*)args, environ) != 0) return -1;
    if(killAfter >= 0){
        std::this_thread::sleep_for(std::chrono::duration<double>(killAfter));
        kill(pid, SIGKILL);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if(killAfter < 0 && !(WIFEXITED(status) && WEXITSTATUS(status) == 0)) return -1;
    return t;
}
#endif

int main(int argc, char** argv){
#ifndef _WIN32
    // (the writer of a test of writes cut short: roundtripTest --write-child <file> <mode> <x> <y> <z>)
    if(argc >= 7 && !std::strcmp(argv[1], "--write-child")){
        const uint64_t cx = std::strtoull(argv[4], 0, 10), cy = std::strtoull(argv[5], 0, 10), cz = std::strtoull(argv[6], 0, 10);
        const std::vector<uint16_t> d = cutStack(cx, cy, cz);
        return writeTiffParallelWrapper(cx, cy, cz, argv[2], d.data(), 16, 0, 0, argv[3], false, "lzw", 1, 0) ? 1 : 0;
    }
    selfPath = argv[0];
#endif
    const std::string dir = (argc > 1) ? argv[1] : ".";

    const uint64_t x = 40, y = 24, z = 3;
    const uint64_t stripSize = 8;
    const uint64_t n = x * y * z;

    const Case cases[] = {
        {"uint8",  8,  1}, {"uint16", 16, 1}, {"int16",  16, 2},
        {"int32",  32, 2}, {"float",  32, 3}, {"double", 64, 3},
    };
    const char* comps[] = {"none", "lzw", "zstd"};
    // 0 = striped. 16 gives a 3x2 grid over the 40x24 image, ragged in both
    // directions, so the padded edge tiles are exercised too.
    const uint64_t tiles[] = {0, 16};

    std::mt19937 rng(1234567u);
    std::uniform_int_distribution<int> byteDist(0, 255);

    int total = 0, failures = 0;
    for (const Case& c : cases){
        const uint64_t es = c.bits / 8;
        const uint64_t nbytes = n * es;

        std::vector<uint8_t> orig(nbytes);
        for (uint64_t i = 0; i < nbytes; i++) orig[i] = static_cast<uint8_t>(byteDist(rng));

        for (const char* comp : comps)
        for (const uint64_t tile : tiles){
            total++;
            char lbl[24];
            std::snprintf(lbl, sizeof lbl, "%s/%s", comp, tile ? "tile" : "strip");
            const std::string path = dir + "/rt_" + c.name + "_" + comp + "_t" + std::to_string(tile) + ".tif";

            step(c.name, lbl, "write");
            uint8_t werr = writeTiffParallelWrapper(x, y, z, path.c_str(), orig.data(), c.bits,
                                                    0, stripSize, "w", false,
                                                    std::string(comp), c.sampleFormat, tile);
            if (werr){ std::printf("FAIL  %-6s %-10s  (write error)\n", c.name, lbl); failures++; continue; }

            // Metadata should round trip.
            step(c.name, lbl, "read metadata");
            const uint64_t rbits = getDataType(path.c_str());
            const uint64_t rfmt  = getSampleFormat(path.c_str());
            uint64_t* dims = getImageSize(path.c_str());   // {y, x, z}
            const bool metaOK = rbits == c.bits && rfmt == c.sampleFormat &&
                                dims[0] == y && dims[1] == x && dims[2] == z;
            step(c.name, lbl, "free metadata buffer");
            free(dims);

            // No-flip read (the Python/C layout) must be byte-identical to what we wrote.
            step(c.name, lbl, "read no-flip");
            void* rbuf = readTiffParallelWrapperNoXYFlip(path.c_str(), {});
            const bool dataOK = rbuf && std::memcmp(rbuf, orig.data(), nbytes) == 0;
            step(c.name, lbl, "free no-flip buffer");
            free(rbuf);

            // Flip read (the MATLAB (y,x,z) column-major layout): compare against the
            // transpose of the original, exercising the flipTransposeStrip path.
            step(c.name, lbl, "read flip");
            std::vector<uint8_t> expFlip(nbytes);
            for (uint64_t zi = 0; zi < z; zi++)
                for (uint64_t yi = 0; yi < y; yi++)
                    for (uint64_t xi = 0; xi < x; xi++)
                        std::memcpy(&expFlip[(yi + xi*y + zi*x*y) * es],
                                    &orig[(zi*y*x + yi*x + xi) * es], es);
            void* fbuf = readTiffParallelWrapper(path.c_str());
            const bool flipOK = fbuf && std::memcmp(fbuf, expFlip.data(), nbytes) == 0;
            step(c.name, lbl, "free flip buffer");
            free(fbuf);

            std::remove(path.c_str());

            if (metaOK && dataOK && flipOK){
                std::printf("PASS  %-6s %-10s\n", c.name, lbl);
            } else {
                std::printf("FAIL  %-6s %-10s  (meta=%d noflip=%d flip=%d)\n",
                            c.name, lbl, (int)metaOK, (int)dataOK, (int)flipOK);
                failures++;
            }
        }
    }

    // ---- ImageJ-format files: RGB sample scaling, z-range offset, byte order.
    {
        const uint64_t ix = 40, iy = 24, iz = 3;

        // Chunky RGB uint8 with distinct per-channel patterns.
        std::vector<uint8_t> rgb(ix*iy*iz*3);
        for(uint64_t k = 0; k < iz; k++)
            for(uint64_t i = 0; i < iy; i++)
                for(uint64_t j = 0; j < ix; j++){
                    uint8_t* p = &rgb[((k*iy + i)*ix + j)*3];
                    p[0] = (uint8_t)((j*3 + k*7) % 250);
                    p[1] = (uint8_t)((i*5 + k*11) % 250);
                    p[2] = (uint8_t)((j + i*2 + k*13) % 250);
                }
        std::vector<uint16_t> g16(ix*iy*iz);
        for(uint64_t i = 0; i < g16.size(); i++) g16[i] = (uint16_t)((i*7 + 13) % 6000);

        struct IJCase { const char* name; const uint8_t* data; uint64_t spp, bits; bool be; };
        const IJCase ijCases[] = {
            {"ij-rgb",    rgb.data(),                 3, 8,  false},
            {"ij-g16-le", (const uint8_t*)g16.data(), 1, 16, false},
            {"ij-g16-be", (const uint8_t*)g16.data(), 1, 16, true},
        };
        for(const IJCase& c : ijCases){
            total++;
            const std::string path = dir + "/" + c.name + ".tif";
            const uint64_t es = c.bits/8;
            const uint64_t sliceBytes = ix*iy*c.spp*es;

            step(c.name, "imagej", "write fixture");
            if(!writeImageJTiff(path, c.data, ix, iy, iz, c.spp, c.bits, c.be)){
                std::printf("FAIL  %-9s (fixture write)\n", c.name); failures++; continue;
            }

            // No-flip read must reproduce the host-order source exactly.
            step(c.name, "imagej", "read no-flip");
            void* nf = readTiffParallelWrapperNoXYFlip(path.c_str(), {});
            const bool nfOK = nf && std::memcmp(nf, c.data, iz*sliceBytes) == 0;
            free(nf);

            // z-range [1,3) must start at slice 1, not slice 0.
            step(c.name, "imagej", "read z-range");
            void* zr = readTiffParallelWrapperNoXYFlip(path.c_str(), {1, 3});
            const bool zrOK = zr && std::memcmp(zr, c.data + sliceBytes, 2*sliceBytes) == 0;
            free(zr);

            // Flip read: column-major (y, x[, c], z), the layout the mex returns.
            step(c.name, "imagej", "read flip");
            std::vector<uint8_t> expFlip(iz*sliceBytes);
            for(uint64_t k = 0; k < iz; k++)
                for(uint64_t i = 0; i < iy; i++)
                    for(uint64_t j = 0; j < ix; j++)
                        for(uint64_t ch = 0; ch < c.spp; ch++)
                            std::memcpy(&expFlip[(i + j*iy + ch*ix*iy + k*ix*iy*c.spp)*es],
                                        c.data + (((k*iy + i)*ix + j)*c.spp + ch)*es, es);
            void* fl = readTiffParallelWrapper(path.c_str());
            const bool flOK = fl && std::memcmp(fl, expFlip.data(), expFlip.size()) == 0;
            free(fl);

            std::remove(path.c_str());
            if(nfOK && zrOK && flOK){
                std::printf("PASS  %-9s imagej\n", c.name);
            } else {
                std::printf("FAIL  %-9s imagej  (noflip=%d zrange=%d flip=%d)\n",
                            c.name, (int)nfOK, (int)zrOK, (int)flOK);
                failures++;
            }
        }
    }

    // ---- ImageJ stacks read by several threads, each its own run of 4 MB
    // blocks of whole rows: slices of more than one block (a run then starts
    // and ends inside a slice), runs of whole slices, a single slice, every
    // sample size in both byte orders, RGB and odd sizes; each read whole and
    // as a slice range, in both layouts, with 1, 3 and every thread.
    {
        struct IJP { const char* name; uint64_t x, y, z, spp, bits; bool be; };
        const IJP ijpCases[] = {
            {"ijp-g16-be-cut", 1500, 1500,  7, 1, 16, true},    // 4.5 MB slices, two blocks each
            {"ijp-rgb8-cut",   1500, 1000,  3, 3,  8, false},   // the same in RGB
            {"ijp-g64-be-one",  900,  800,  1, 1, 64, true},    // one 5.8 MB slice, two blocks
            {"ijp-g8-odd",       37,   29, 11, 1,  8, false},
            {"ijp-g32-be",       40,   24,  9, 1, 32, true},
            {"ijp-g64-be",       33,   21,  9, 1, 64, true},
            {"ijp-g64-le",       33,   21,  9, 1, 64, false},
            {"ijp-rgb16-be",     40,   24,  9, 3, 16, true},
        };
        const int prevThreads = omp_get_max_threads();
        for(const IJP& c : ijpCases){
            total++;
            const uint64_t es = c.bits/8, sliceBytes = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(sliceBytes*c.z);
            uint64_t st = 0x9E3779B97F4A7C15ull ^ (c.x*131 + c.y*7 + c.bits);
            for(uint8_t& b : src){ st = st*6364136223846793005ull + 1442695040888963407ull; b = (uint8_t)(st >> 56); }
            // The flip layout: column-major (y, x[, c], z)
            std::vector<uint8_t> flipped(src.size());
            for(uint64_t k = 0; k < c.z; k++)
                for(uint64_t i = 0; i < c.y; i++)
                    for(uint64_t j = 0; j < c.x; j++)
                        for(uint64_t ch = 0; ch < c.spp; ch++)
                            std::memcpy(&flipped[(i + j*c.y + ch*c.x*c.y + k*c.x*c.y*c.spp)*es],
                                        &src[(((k*c.y + i)*c.x + j)*c.spp + ch)*es], es);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "imagej-par", "write fixture");
            const bool made = writeImageJTiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, c.be);
            // A slice range inside the stack (the whole of a single slice)
            const uint64_t a = c.z > 2 ? 1 : 0, e = c.z > 2 ? c.z - 1 : c.z;
            std::string bad = made ? "" : " fixture";
            for(int t : {1, 3, prevThreads}){
                omp_set_num_threads(t);
                step(c.name, "imagej-par", "reads");
                void* nf = made ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
                void* nr = made ? readTiffParallelWrapperNoXYFlip(path.c_str(), {a, e}) : nullptr;
                void* fl = made ? readTiffParallelWrapper(path.c_str()) : nullptr;
                void* fr = made ? readTiffParallelWrapper(path.c_str(), {a, e}) : nullptr;
                const bool good = nf && std::memcmp(nf, src.data(), src.size()) == 0
                    && nr && std::memcmp(nr, src.data() + a*sliceBytes, (e - a)*sliceBytes) == 0
                    && fl && std::memcmp(fl, flipped.data(), flipped.size()) == 0
                    && fr && std::memcmp(fr, flipped.data() + a*sliceBytes, (e - a)*sliceBytes) == 0;
                free(nf); free(nr); free(fl); free(fr);
                if(made && !good) bad += " " + std::to_string(t) + " threads";
            }
            omp_set_num_threads(prevThreads);
            std::remove(path.c_str());
            if(bad.empty()) std::printf("PASS  %-14s imagej-par\n", c.name);
            else { std::printf("FAIL  %-14s imagej-par (%s)\n", c.name, bad.c_str() + 1); failures++; }
        }
    }

    // ---- Scanline fallbacks, called directly on valid files. Wide and tall
    // images (the old row offset only ran past the buffer when x < y), 3D, 2D
    // and an offset z-range, both layouts, grayscale from the library's writer
    // and hand-written RGB.
    {
        struct FB { const char* name; uint64_t x, y, spp, bits; uint16_t fmt; const char* comp; };
        const FB fbs[] = {
            {"u16-lzw-wide",  40, 24, 1, 16, 1, "lzw"},
            {"u16-lzw-tall",  24, 40, 1, 16, 1, "lzw"},
            {"u8-none-tall",  24, 40, 1,  8, 1, "none"},
            {"f32-zstd-wide", 40, 24, 1, 32, 3, "zstd"},
            {"rgb8-wide",     40, 24, 3,  8, 1, nullptr},
            {"rgb8-tall",     24, 40, 3,  8, 1, nullptr},
        };
        std::mt19937 frng(7654321u);
        std::uniform_int_distribution<int> fbyte(0, 255);
        for(const FB& c : fbs){
            const uint64_t fz = 3, es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*fz);
            for(auto& v8 : src) v8 = (uint8_t)fbyte(frng);
            if(c.bits == 32) for(uint64_t i = 0; i < src.size(); i += 4) src[i + 3] &= 0x3F;   // keep floats finite
            const std::string p3 = dir + "/fb3_" + c.name + ".tif", p2 = dir + "/fb2_" + c.name + ".tif";
            step(c.name, "fallback", "write fixtures");
            bool wrote;
            if(c.comp) wrote = !writeTiffParallelWrapper(c.x, c.y, fz, p3.c_str(), src.data(), c.bits, 0, 8, "w", false, std::string(c.comp), c.fmt, 0)
                            && !writeTiffParallelWrapper(c.x, c.y, 1, p2.c_str(), src.data(), c.bits, 0, 8, "w", false, std::string(c.comp), c.fmt, 0);
            else       wrote = writePlainTiff(p3, src.data(), c.x, c.y, fz, c.spp, c.bits)
                            && writePlainTiff(p2, src.data(), c.x, c.y, 1, c.spp, c.bits);
            total++;
            if(!wrote){ std::printf("FAIL  %-14s fallback (fixture write)\n", c.name); failures++; continue; }

            const std::vector<uint8_t> all(src.begin(), src.end());
            const std::vector<uint8_t> tail(src.begin() + slice, src.end());          // slices 1..2
            const std::vector<uint8_t> first(src.begin(), src.begin() + slice);       // slice 0
            step(c.name, "fallback", "3D no-flip / flip / z-range");
            const bool b3  = fallbackMatches(false, p3, c.x, c.y, fz, c.spp, c.bits, 0, false, all)
                          && fallbackMatches(false, p3, c.x, c.y, fz, c.spp, c.bits, 0, true, flipped(src.data(), c.x, c.y, fz, c.spp, es))
                          && fallbackMatches(false, p3, c.x, c.y, 2, c.spp, c.bits, 1, false, tail)
                          && fallbackMatches(false, p3, c.x, c.y, 2, c.spp, c.bits, 1, true, flipped(src.data() + slice, c.x, c.y, 2, c.spp, es));
            step(c.name, "fallback", "2D no-flip / flip");
            const bool b2  = fallbackMatches(true, p2, c.x, c.y, 1, c.spp, c.bits, 0, false, first)
                          && fallbackMatches(true, p2, c.x, c.y, 1, c.spp, c.bits, 0, true, flipped(src.data(), c.x, c.y, 1, c.spp, es));
            // The normal readers must agree on the same files (RGB fixtures included).
            step(c.name, "fallback", "normal readers");
            void* nf = readTiffParallelWrapperNoXYFlip(p3.c_str(), {});
            void* fl = readTiffParallelWrapper(p3.c_str());
            const std::vector<uint8_t> f3 = flipped(src.data(), c.x, c.y, fz, c.spp, es);
            const bool rd = nf && fl && std::memcmp(nf, all.data(), all.size()) == 0 && std::memcmp(fl, f3.data(), f3.size()) == 0;
            free(nf); free(fl);
            std::remove(p3.c_str()); std::remove(p2.c_str());
            if(b3 && b2 && rd) std::printf("PASS  %-14s fallback\n", c.name);
            else { std::printf("FAIL  %-14s fallback  (3D=%d 2D=%d readers=%d)\n", c.name, (int)b3, (int)b2, (int)rd); failures++; }
        }
    }

    // ---- Short strips. Flip reads decode runs of short strips and transpose
    // each run at once (stripsPerTranspose). Odd heights leave a partial last
    // strip and a partial last run. Wide uncompressed single-strip images are
    // split by libtiff into one-row strips on read, including RGB.
    {
        struct SS { const char* name; uint64_t x, y, z, spp, bits; uint16_t fmt; const char* comp; };
        const SS sss[] = {
            {"u8-none-rps3",    40,   41, 3, 1,  8, 1, "none"},
            {"u16-lzw-rps3",    40,   41, 3, 1, 16, 1, "lzw"},
            {"f32-none-rps3",   40,   41, 3, 1, 32, 3, "none"},
            {"f64-zstd-rps3",   40,   41, 3, 1, 64, 3, "zstd"},
            {"u16-onerow",    4100,   37, 2, 1, 16, 1, nullptr},
            {"rgb8-onerow",   1500,   37, 2, 3,  8, 1, nullptr},
        };
        std::mt19937 srng(24680u);
        std::uniform_int_distribution<int> sbyte(0, 255);
        for(const SS& c : sss){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*c.z);
            for(auto& v8 : src) v8 = (uint8_t)sbyte(srng);
            if(c.bits == 32) for(uint64_t i = 0; i < src.size(); i += 4) src[i + 3] &= 0x3F;   // keep floats finite
            if(c.bits == 64) for(uint64_t i = 0; i < src.size(); i += 8) src[i + 7] &= 0x3F;
            const std::string p3 = dir + "/ss3_" + c.name + ".tif", p2 = dir + "/ss2_" + c.name + ".tif";
            step(c.name, "short strips", "write fixtures");
            bool ok = c.comp ? !writeTiffParallelWrapper(c.x, c.y, c.z, p3.c_str(), src.data(), c.bits, 0, 3, "w", false, std::string(c.comp), c.fmt, 0)
                               && !writeTiffParallelWrapper(c.x, c.y, 1, p2.c_str(), src.data(), c.bits, 0, 3, "w", false, std::string(c.comp), c.fmt, 0)
                             : writePlainTiff(p3, src.data(), c.x, c.y, c.z, c.spp, c.bits)
                               && writePlainTiff(p2, src.data(), c.x, c.y, 1, c.spp, c.bits);
            step(c.name, "short strips", "3D, 2D and z-range reads, both layouts");
            struct Read { const std::string* path; std::vector<uint64_t> range; uint64_t first, count; };
            const Read reads[] = {{&p3, {}, 0, c.z}, {&p3, {1, c.z}, 1, c.z - 1}, {&p3, {1}, 1, 1}, {&p2, {}, 0, 1}};
            for(const Read& r : reads){
                if(!ok) break;
                const uint8_t* want = src.data() + r.first*slice;
                const std::vector<uint8_t> wantFlip = flipped(want, c.x, c.y, r.count, c.spp, es);
                void* nf = readTiffParallelWrapperNoXYFlip(r.path->c_str(), r.range);
                void* fl = readTiffParallelWrapper(r.path->c_str(), r.range);
                ok = nf && fl && std::memcmp(nf, want, r.count*slice) == 0
                     && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
                free(nf); free(fl);
            }
            std::remove(p3.c_str()); std::remove(p2.c_str());
            if(ok) std::printf("PASS  %-14s short strips\n", c.name);
            else { std::printf("FAIL  %-14s short strips\n", c.name); failures++; }
        }
    }

    // ---- Many slices, few threads. With more slices than threads each thread
    // reads its own block and then takes slices from the back of the others
    // (Linux), so every slice must still be read exactly once, in its place.
    {
        struct MS { const char* name; const char* comp; };
        const MS mss[] = {{"u16-none-37z", "none"}, {"u16-lzw-37z", "lzw"}};
        const uint64_t mx = 24, my = 17, mz = 37, slice = mx*my;
        std::vector<uint16_t> ms(slice*mz);
        for(uint64_t i = 0; i < ms.size(); i++) ms[i] = (uint16_t)((i*31 + 7) % 65000);
        const std::vector<uint8_t> wantFlip = flipped((const uint8_t*)ms.data(), mx, my, mz, 1, 2);
        const std::vector<uint8_t> wantRangeFlip = flipped((const uint8_t*)(ms.data() + 3*slice), mx, my, 27, 1, 2);
        const int prevThreads = omp_get_max_threads();
        for(const MS& c : mss){
            total++;
            const std::string mp = dir + "/many_" + c.name + ".tif";
            step(c.name, "many slices", "write fixture");
            bool ok = !writeTiffParallelWrapper(mx, my, mz, mp.c_str(), ms.data(), 16, 0, 5, "w", false, std::string(c.comp), 1, 0);
            const int threads[] = {2, 3, 5, prevThreads};
            for(int t : threads){
                if(!ok) break;
                omp_set_num_threads(t);
                step(c.name, "many slices", "full and z-range reads, both layouts");
                void* nf = readTiffParallelWrapperNoXYFlip(mp.c_str(), {});
                void* fl = readTiffParallelWrapper(mp.c_str());
                void* rg = readTiffParallelWrapperNoXYFlip(mp.c_str(), {3, 30});
                void* rf = readTiffParallelWrapper(mp.c_str(), {3, 30});
                ok = nf && fl && rg && rf
                     && std::memcmp(nf, ms.data(), ms.size()*2) == 0 && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0
                     && std::memcmp(rg, ms.data() + 3*slice, 27*slice*2) == 0 && std::memcmp(rf, wantRangeFlip.data(), wantRangeFlip.size()) == 0;
                free(nf); free(fl); free(rg); free(rf);
            }
            omp_set_num_threads(prevThreads);
            std::remove(mp.c_str());
            if(ok) std::printf("PASS  %-14s many slices\n", c.name);
            else { std::printf("FAIL  %-14s many slices\n", c.name); failures++; }
        }
    }

    // ---- Big-endian uncompressed stacks. Each page is larger than libtiff's
    // ~8 KB strip size, so libtiff splits it into short strips and the readers
    // copy them straight from the file (readRawStrips), swapping bytes the way
    // libtiff would.
    {
        struct BE { const char* name; uint64_t spp, bits; };
        const BE bes[] = {{"be-u8", 1, 8}, {"be-u16", 1, 16}, {"be-32bit", 1, 32}, {"be-64bit", 1, 64}, {"be-rgb16", 3, 16}};
        const uint64_t bx = 300, by = 41, bz = 3;
        std::mt19937 brng(13579u);
        std::uniform_int_distribution<int> bbyte(0, 255);
        for(const BE& c : bes){
            total++;
            const uint64_t es = c.bits/8, slice = bx*by*c.spp*es;
            std::vector<uint8_t> src(slice*bz);
            for(auto& v8 : src) v8 = (uint8_t)bbyte(brng);
            const std::string bp = dir + "/" + c.name + ".tif";
            step(c.name, "big-endian", "write fixture");
            bool ok = writePlainTiff(bp, src.data(), bx, by, bz, c.spp, c.bits, true);
            step(c.name, "big-endian", "full, z-range and single-slice reads, both layouts");
            struct Read { std::vector<uint64_t> range; uint64_t first, count; };
            const Read reads[] = {{{}, 0, bz}, {{1, bz}, 1, bz - 1}, {{2}, 2, 1}};
            for(const Read& r : reads){
                if(!ok) break;
                const uint8_t* want = src.data() + r.first*slice;
                const std::vector<uint8_t> wantFlip = flipped(want, bx, by, r.count, c.spp, es);
                void* nf = readTiffParallelWrapperNoXYFlip(bp.c_str(), r.range);
                void* fl = readTiffParallelWrapper(bp.c_str(), r.range);
                ok = nf && fl && std::memcmp(nf, want, r.count*slice) == 0
                     && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
                free(nf); free(fl);
            }
            std::remove(bp.c_str());
            if(ok) std::printf("PASS  %-14s big-endian\n", c.name);
            else { std::printf("FAIL  %-14s big-endian\n", c.name); failures++; }
        }

        // Strips stored out of order cannot be copied as one block.
        total++;
        const uint64_t rx = 300, ry = 37, rz = 3, rslice = rx*ry*2;
        std::vector<uint8_t> rsrc(rslice*rz);
        for(auto& v8 : rsrc) v8 = (uint8_t)bbyte(brng);
        const std::string rp = dir + "/reversed_strips.tif";
        step("reversed", "strip order", "write + reads");
        bool rok = writeReversedStripsTiff(rp, rsrc.data(), rx, ry, rz, 16, 4);
        for(int flip = 0; rok && flip < 2; flip++){
            const std::vector<uint8_t> want = flip ? flipped(rsrc.data(), rx, ry, rz, 1, 2) : rsrc;
            const std::vector<uint8_t> want1 = flip ? flipped(rsrc.data() + rslice, rx, ry, 1, 1, 2)
                                                    : std::vector<uint8_t>(rsrc.begin() + rslice, rsrc.begin() + 2*rslice);
            void* r = flip ? readTiffParallelWrapper(rp.c_str()) : readTiffParallelWrapperNoXYFlip(rp.c_str(), {});
            void* r1 = flip ? readTiffParallelWrapper(rp.c_str(), {1}) : readTiffParallelWrapperNoXYFlip(rp.c_str(), {1});
            rok = r && r1 && std::memcmp(r, want.data(), want.size()) == 0 && std::memcmp(r1, want1.data(), want1.size()) == 0;
            free(r); free(r1);
        }
        std::remove(rp.c_str());
        if(rok) std::printf("PASS  %-14s strip order\n", "reversed");
        else { std::printf("FAIL  %-14s strip order\n", "reversed"); failures++; }
    }

    // ---- One OpenMP thread. Single-image reads used to take a separate raw-read
    // path there, which always returned slice 0 of an uncompressed stack.
    {
        total++;
        const uint64_t sx = 40, sy = 24, sz = 4, slice = sx*sy;
        std::vector<uint16_t> st(slice*sz);
        for(uint64_t i = 0; i < st.size(); i++) st[i] = (uint16_t)((i*7 + 3) % 50000);
        const std::string sp = dir + "/one_thread.tif";
        step("one-thread", "none", "write fixture");
        bool ok = writePlainTiff(sp, (const uint8_t*)st.data(), sx, sy, sz, 1, 16);
        const int prevThreads = omp_get_max_threads();
        omp_set_num_threads(1);
        step("one-thread", "none", "single-slice reads, both layouts");
        for(uint64_t k = 0; ok && k < sz; k++){
            const uint16_t* want = st.data() + k*slice;
            const std::vector<uint8_t> wantFlip = flipped((const uint8_t*)want, sx, sy, 1, 1, 2);
            void* one = readTiffParallelWrapperNoXYFlip(sp.c_str(), {k});
            void* range = readTiffParallelWrapperNoXYFlip(sp.c_str(), {k, k + 1});
            void* flip = readTiffParallelWrapper(sp.c_str(), {k, k + 1});
            ok = one && range && flip
                 && std::memcmp(one, want, slice*2) == 0 && std::memcmp(range, want, slice*2) == 0
                 && std::memcmp(flip, wantFlip.data(), wantFlip.size()) == 0;
            free(one); free(range); free(flip);
        }
        step("one-thread", "none", "full read");
        void* all = ok ? readTiffParallelWrapperNoXYFlip(sp.c_str(), {}) : nullptr;
        ok = all && std::memcmp(all, st.data(), st.size()*2) == 0;
        free(all);
        omp_set_num_threads(prevThreads);
        std::remove(sp.c_str());
        if(ok) std::printf("PASS  %-14s one thread\n", "u16-none");
        else { std::printf("FAIL  %-14s one thread  (single-slice or full read wrong)\n", "u16-none"); failures++; }
    }

    // ---- Writer layouts. MATLAB-layout input (column-major, transposed strip by
    // strip as it is written) must give the same file as row-major input, a
    // single-threaded write must give the same file as a parallel one, and the
    // file must read back to the source.
    {
        struct WL { const char* name; uint64_t x, y, z, bits; uint16_t fmt; const char* comp; uint64_t strip, tile; };
        const WL wls[] = {
            {"u8-none-s7",    40, 41, 3,  8, 1, "none",   7,  0},
            {"u16-lzw-s512", 333, 77, 4, 16, 1, "lzw",  512,  0},
            {"f32-zstd-s5",   61, 29, 3, 32, 3, "zstd",   5,  0},
            {"f64-lzw-t16",   50, 37, 2, 64, 3, "lzw",  512, 16},
            {"u16-none-t32",  70, 45, 3, 16, 1, "none", 512, 32},
            {"u8-zstd-t16",   33, 70, 2,  8, 1, "zstd", 512, 16},
        };
        std::mt19937 wrng(424242u);
        std::uniform_int_distribution<int> wbyte(0, 255);
        const int prevThreads = omp_get_max_threads();
        for(const WL& c : wls){
            total++;
            const uint64_t es = c.bits/8;
            std::vector<uint8_t> rm(c.x*c.y*c.z*es);
            for(auto& v8 : rm) v8 = (uint8_t)(wbyte(wrng) & 0x3F);   // compressible, and floats stay finite
            const std::vector<uint8_t> cm = flipped(rm.data(), c.x, c.y, c.z, 1, es);
            const std::string pa = dir + "/wl_row_" + c.name + ".tif", pb = dir + "/wl_matlab_" + c.name + ".tif",
                              pc = dir + "/wl_one_" + c.name + ".tif";
            step(c.name, "writer layouts", "row-major, MATLAB-layout and one-thread writes");
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, pa.c_str(), rm.data(), c.bits, 0, c.strip, "w", false, std::string(c.comp), c.fmt, c.tile)
                      && !writeTiffParallelWrapper(c.x, c.y, c.z, pb.c_str(), cm.data(), c.bits, 0, c.strip, "w", true, std::string(c.comp), c.fmt, c.tile);
            omp_set_num_threads(1);
            ok = ok && !writeTiffParallelWrapper(c.x, c.y, c.z, pc.c_str(), cm.data(), c.bits, 0, c.strip, "w", true, std::string(c.comp), c.fmt, c.tile);
            omp_set_num_threads(prevThreads);
            const std::vector<uint8_t> fa = readFileBytes(pa), fb = readFileBytes(pb), fc = readFileBytes(pc);
            ok = ok && !fa.empty() && fa == fb && fa == fc;
            void* back = ok ? readTiffParallelWrapperNoXYFlip(pa.c_str(), {}) : nullptr;
            ok = ok && back && std::memcmp(back, rm.data(), rm.size()) == 0;
            free(back);
            std::remove(pa.c_str()); std::remove(pb.c_str()); std::remove(pc.c_str());
            if(ok) std::printf("PASS  %-14s writer layouts\n", c.name);
            else { std::printf("FAIL  %-14s writer layouts  (files differ or read back wrong)\n", c.name); failures++; }
        }
    }

    // ---- Automatic strip size (strip size 0, what the Python and MATLAB writers
    // pass). 2D images and short stacks get smaller strips so more threads can
    // compress them; tall stacks keep 512 rows. Only the image size decides, so
    // one thread and many must give the same file, and it must read back.
    {
        struct AS { const char* name; uint64_t x, y, z, bits; const char* comp; uint64_t rows; };
        const AS cases[] = {
            {"auto-2d",    2048, 2048,   1, 16, "lzw",   16},   // 8 rows would give 256 strips; 64 KB minimum
            {"auto-short", 2000, 1500,   3, 16, "zstd",  18},   // 86 strips per slice
            {"auto-tall",   256,  700, 260,  8, "none", 512},   // already enough strips
        };
        std::mt19937 arng(77u);
        const int prevThreads = omp_get_max_threads();
        for(const AS& c : cases){
            total++;
            const uint64_t es = c.bits/8;
            std::vector<uint8_t> src(c.x*c.y*c.z*es);
            for(auto& v8 : src) v8 = (uint8_t)(arng() & 0x3F);
            const std::string pa = dir + "/" + c.name + "_n.tif", pb = dir + "/" + c.name + "_1.tif";
            step(c.name, "automatic strips", "strip size 0 at full and one thread, read back both ways");
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, pa.c_str(), src.data(), c.bits, 0, 0, "w", false, std::string(c.comp), 1, 0);
            omp_set_num_threads(1);
            ok = ok && !writeTiffParallelWrapper(c.x, c.y, c.z, pb.c_str(), src.data(), c.bits, 0, 0, "w", false, std::string(c.comp), 1, 0);
            omp_set_num_threads(prevThreads);
            const std::vector<uint8_t> fa = readFileBytes(pa), fb = readFileBytes(pb);
            const uint64_t rows = firstRowsPerStrip(fa);
            ok = ok && !fa.empty() && fa == fb && rows == c.rows;
            void* back = ok ? readTiffParallelWrapperNoXYFlip(pa.c_str(), {}) : nullptr;
            ok = ok && back && std::memcmp(back, src.data(), src.size()) == 0;
            free(back);
            if(ok){
                const std::vector<uint8_t> cm = flipped(src.data(), c.x, c.y, c.z, 1, es);
                void* fl = readTiffParallelWrapper(pa.c_str());
                ok = fl && std::memcmp(fl, cm.data(), cm.size()) == 0;
                free(fl);
            }
            std::remove(pa.c_str()); std::remove(pb.c_str());
            if(ok) std::printf("PASS  %-14s automatic strips (%llu rows)\n", c.name, (unsigned long long)rows);
            else { std::printf("FAIL  %-14s automatic strips  (%llu rows per strip, expected %llu; or the files differ or read back wrong)\n",
                               c.name, (unsigned long long)rows, (unsigned long long)c.rows); failures++; }
        }
    }

    // ---- Short stacks (a few slices with many strips each) are read slice by
    // slice with every thread instead of one thread per slice. Grayscale stacks
    // written by the library, an RGB fixture, and a z-range must match the
    // source in both layouts.
    {
        const int prevThreads = omp_get_max_threads();
        omp_set_num_threads(8);   // 2 x slices <= threads, so the slice-by-slice path is taken
        struct SS { const char* name; uint64_t x, y, z, spp, bits; const char* comp; };
        // Slices must be at least 4 MB (16 MB uncompressed) to be read this way.
        const SS cases[] = {
            {"short-u16-lzw",  2048, 1024, 3, 1, 16, "lzw"},
            {"short-u8-none",  4096, 4096, 3, 1,  8, "none"},
            {"short-rgb",      2400, 2400, 2, 3,  8, nullptr},   // fixture, one strip per page split by libtiff
        };
        std::mt19937 srng(31u);
        for(const SS& c : cases){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*c.z);
            for(auto& v8 : src) v8 = (uint8_t)srng();
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "short stacks", "read slice by slice in both layouts and a z-range");
            bool ok = c.comp ? !writeTiffParallelWrapper(c.x, c.y, c.z, path.c_str(), src.data(), c.bits, 0, 64, "w", false, std::string(c.comp), 1, 0)
                             : writePlainTiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits);
            void* rm = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
            ok = ok && rm && std::memcmp(rm, src.data(), src.size()) == 0;
            free(rm);
            void* fl = ok ? readTiffParallelWrapper(path.c_str()) : nullptr;
            if(fl){
                const std::vector<uint8_t> cm = flipped(src.data(), c.x, c.y, c.z, c.spp, es);
                ok = std::memcmp(fl, cm.data(), cm.size()) == 0;
            }
            else ok = false;
            free(fl);
            void* part = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {1, c.z}) : nullptr;
            ok = ok && part && std::memcmp(part, src.data() + slice, slice*(c.z - 1)) == 0;
            free(part);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-14s short stacks\n", c.name);
            else { std::printf("FAIL  %-14s short stacks  (read back wrong)\n", c.name); failures++; }
        }
        omp_set_num_threads(prevThreads);
    }

    // ---- Metadata in one pass: getTiffInfo must give the values the
    // per-value helpers give (each opening the file), for strips, tiles, chunky
    // and planar color, ImageJ stacks in both byte orders and a deep stack.
    {
        struct MD { const char* name; bool imageJ; uint64_t x, y, z, spp, bits; uint16_t fmt, comp; uint32_t rps, tile; bool planar, be; };
        const MD mds[] = {
            {"md-strips", false, 40, 30,   6, 1, 16, 1, COMPRESSION_LZW,            7,  0, false, false},
            {"md-tiles",  false, 40, 30,   5, 1, 32, 3, COMPRESSION_ZSTD,           0, 16, false, false},
            {"md-planar", false, 40, 30,   3, 3,  8, 1, COMPRESSION_ADOBE_DEFLATE,  9,  0, true,  false},
            {"md-rgba",   false, 40, 30,   4, 4,  8, 1, COMPRESSION_NONE,          30,  0, false, true},
            {"md-signed", false, 40, 30,   1, 1, 16, 2, COMPRESSION_NONE,          30,  0, false, false},
            {"md-deep",   false,  8,  8, 300, 1, 16, 1, COMPRESSION_NONE,           8,  0, false, false},
            {"md-ij-le",  true,  40, 30,   6, 1, 16, 1, COMPRESSION_NONE,           0,  0, false, false},
            {"md-ij-be",  true,  40, 30,   4, 3,  8, 1, COMPRESSION_NONE,           0,  0, false, true},
        };
        for(const MD& c : mds){
            total++;
            const std::string path = dir + "/" + c.name + ".tif";
            std::vector<uint8_t> src(c.x*c.y*c.z*c.spp*(c.bits/8));
            for(size_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(i*131 + 7);
            step(c.name, "metadata", "write fixture");
            const bool made = c.imageJ ? writeImageJTiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, c.be)
                                       : writeLibtiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, c.fmt, c.comp, c.rps, c.tile, c.planar, c.be);
            TiffInfo info;
            bool ok = made && !getTiffInfo(path.c_str(), info);
            step(c.name, "metadata", "compare with the helpers");
            uint64_t* dims = ok ? getImageSize(path.c_str()) : NULL;
            TIFF* tif = ok ? TIFFOpen(path.c_str(), "r") : NULL;
            ok = ok && dims && tif;
            if(ok){
                uint32_t rps = 0;
                uint16_t comp = COMPRESSION_NONE, pc = PLANARCONFIG_CONTIG;
                TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps);
                TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &comp);
                TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &pc);
                const bool tiled = TIFFIsTiled(tif);
                const uint64_t spp = getSamplesPerPixel(path.c_str());
                const bool planar = spp > 1 && pc == PLANARCONFIG_SEPARATE;
                const bool ij = !tiled && !planar && isImageJIm(path.c_str());
                const uint64_t ijImages = imageJImGetZ(path.c_str());
                ok = info.x == dims[1] && info.y == dims[0] && info.directories == dims[2]
                  && info.z == (ij && ijImages ? ijImages : dims[2]) && info.z == c.z
                  && info.bits == getDataType(path.c_str()) && info.samples == spp
                  && info.sampleFormat == getSampleFormat(path.c_str())
                  && info.stripSize == rps && info.compression == comp
                  && info.tiled == tiled && info.planar == planar
                  && info.imageJ == ij && info.imageJ == c.imageJ && info.imageJImages == ijImages;
                // Where each directory starts, as libtiff finds it walking there
                ok = ok && info.offsets.size() == info.directories;
                for(uint64_t d = 0; ok && d < info.directories; d++)
                    ok = TIFFSetDirectory(tif, (tdir_t)d) && TIFFCurrentDirOffset(tif) == info.offsets[d];
            }
            if(tif) TIFFClose(tif);
            free(dims);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-16s metadata in one pass\n", c.name);
            else { std::printf("FAIL  %-16s metadata in one pass  (differs from the helpers)\n", c.name); failures++; }
        }
    }

    // ---- Where the directories start (TiffInfo::offsets). Reads given them read
    // what reads without them do: whole stacks, slice ranges and single slices
    // near the end, in both layouts, with one thread and every thread. And they
    // are what the readers go by: with two directories' starts swapped, a read
    // of one of the slices returns the other.
    {
        total++;
        const uint64_t ox = 64, oy = 48, oz = 150, sliceBytes = ox*oy*2;
        std::vector<uint16_t> src(ox*oy*oz);
        for(uint64_t i = 0; i < src.size(); i++) src[i] = (uint16_t)((i*2654435761u) >> 7);
        const std::string path = dir + "/offsets.tif";
        step("offsets", "dirs", "write + reads with and without the starts");
        TiffInfo info;
        bool ok = writeLibtiff(path, (const uint8_t*)src.data(), ox, oy, oz, 1, 16, 1, COMPRESSION_LZW, 24, 0, false, false)
                  && !getTiffInfo(path.c_str(), info) && info.offsets.size() == oz;
        TiffInfo walk = info;
        walk.offsets.clear();
        const int prevThreads = omp_get_max_threads();
        const std::vector<std::vector<uint64_t>> ranges = {{}, {140, 150}, {149}, {75}};
        for(int t : {1, prevThreads})
            for(const std::vector<uint64_t>& r : ranges)
                for(uint8_t flip = 0; flip < 2 && ok; flip++){
                    omp_set_num_threads(t);
                    void* a = readTiffParallelInfo(path.c_str(), info, NULL, flip, r);
                    void* b = readTiffParallelInfo(path.c_str(), walk, NULL, flip, r);
                    const uint64_t n = (r.empty() ? oz : r.size() == 2 ? r[1] - r[0] : 1)*sliceBytes;
                    ok = a && b && std::memcmp(a, b, n) == 0
                         && (flip || std::memcmp(a, (const uint8_t*)src.data() + (r.empty() ? 0 : r[0])*sliceBytes, n) == 0);
                    free(a); free(b);
                }
        omp_set_num_threads(prevThreads);
        step("offsets", "dirs", "swapped starts");
        bool used = false;
        if(ok){
            TiffInfo swapped = info;
            std::swap(swapped.offsets[100], swapped.offsets[101]);
            void* s = readTiffParallelInfo(path.c_str(), swapped, NULL, 0, {100});
            used = s && std::memcmp(s, (const uint8_t*)src.data() + 101*sliceBytes, sliceBytes) == 0;
            free(s);
        }
        std::remove(path.c_str());
        if(ok && used) std::printf("PASS  %-16s directory starts\n", "offsets");
        else { std::printf("FAIL  %-16s directory starts  (reads=%d used=%d)\n", "offsets", (int)ok, (int)used); failures++; }
    }

    // ---- Per-slice destinations: readTiffParallelInfo reading each slice into
    // an allocation of its own. The allocations are poisoned (values the data
    // never takes) and guarded on both sides, and given in the reverse of the
    // order they were made in, so that a slice written at another's address, as
    // if the slices were contiguous, past its end, or not at all, is caught.
    // The first slice's allocation also has room after it for the rest of the
    // read, which a reader taking the slices to follow the first writes over
    // (and fails the check) instead of over the heap.
    // Every reader path: strips decoded or copied raw, runs of one-row strips,
    // tiles, chunky and planar color, planar tiles, a single image, stacks of
    // a few large slices (read slice by slice, or in pieces), the ImageJ reader,
    // and the scanline fallbacks (a stack whose first directory has three
    // strips and the others one, which the strip readers cannot read); both
    // layouts, whole and as slice ranges, with 1, 3 and 8 threads and every
    // thread. Destinations that do not fit the read are refused.
    {
        // kind 0: written by libtiff; 1: an ImageJ stack (big-endian above 8
        // bits); 2: libtiff, with one strip in every directory after the first
        struct PS { const char* name; uint64_t x, y, z, spp, bits; uint16_t comp; uint32_t rps, tile; bool planar; int kind; };
        const PS psCases[] = {
            {"ps-u16-lzw",        40,   24, 9, 1, 16, COMPRESSION_LZW,   8,  0, false, 0},
            {"ps-u8-none",        40,   24, 9, 1,  8, COMPRESSION_NONE,  8,  0, false, 0},
            {"ps-f32-zstd",       40,   24, 9, 1, 32, COMPRESSION_ZSTD,  8,  0, false, 0},
            {"ps-f64-rows",       37,   29, 6, 1, 64, COMPRESSION_LZW,   1,  0, false, 0},
            {"ps-u16-tiled",      40,   24, 9, 1, 16, COMPRESSION_LZW,   0, 16, false, 0},
            {"ps-rgb16-zstd",     40,   24, 5, 3, 16, COMPRESSION_ZSTD,  8,  0, false, 0},
            {"ps-rgb8-tiled",     40,   24, 5, 3,  8, COMPRESSION_NONE,  0, 16, false, 0},
            {"ps-rgb8-planar",    40,   24, 5, 3,  8, COMPRESSION_LZW,   8,  0, true,  0},
            {"ps-rgba16-ptiled",  40,   24, 5, 4, 16, COMPRESSION_ZSTD,  0, 16, true,  0},
            {"ps-u16-image",      40,   24, 1, 1, 16, COMPRESSION_LZW,   8,  0, false, 0},
            {"ps-u32-two",      1024, 1024, 2, 1, 32, COMPRESSION_LZW,  16,  0, false, 0},
            {"ps-u32-five",     1024, 1024, 5, 1, 32, COMPRESSION_LZW,  16,  0, false, 0},
            {"ps-ij-g16",       1500, 1500, 4, 1, 16, COMPRESSION_NONE,  0,  0, false, 1},
            {"ps-ij-rgb8",        40,   24, 9, 3,  8, COMPRESSION_NONE,  0,  0, false, 1},
            {"ps-fallback-u16",   40,   24, 6, 1, 16, COMPRESSION_LZW,   8,  0, false, 2},
            {"ps-fallback-rgb8",  40,   24, 6, 3,  8, COMPRESSION_ZSTD,  8,  0, true,  2},
        };
        const int prevThreads = omp_get_max_threads();
        std::vector<int> threadCounts = {1, 3, 8};
        if(prevThreads != 1 && prevThreads != 3 && prevThreads != 8) threadCounts.push_back(prevThreads);
        const size_t guard = 64;
        const uint8_t poison = 0xA5;
        auto isPoison = [&](uint8_t v){ return v == poison; };
        for(const PS& c : psCases){
            total++;
            const uint64_t es = c.bits/8, sliceBytes = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(sliceBytes*c.z);
            uint64_t st = 0x2545F4914F6CDD1Dull ^ (c.x*131 + c.y*7 + c.z*5 + c.bits*3 + c.spp);
            for(uint8_t& b : src){ st = st*6364136223846793005ull + 1442695040888963407ull; b = (uint8_t)(st >> 60); }
            const std::vector<uint8_t> flip = flipped(src.data(), c.x, c.y, c.z, c.spp, es);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "per-slice", "write fixture");
            const uint16_t fmt = std::strncmp(c.name, "ps-f", 4) == 0 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT;
            bool made = c.kind == 1 ? writeImageJTiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, c.bits > 8)
                                    : writeLibtiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, fmt, c.comp, c.rps, c.tile, c.planar,
                                                   false, PREDICTOR_NONE, FILLORDER_MSB2LSB, c.kind == 2 ? (uint32_t)c.y : 0);
            TiffInfo info;
            made = made && !getTiffInfo(path.c_str(), info) && info.z == c.z && (info.imageJ != 0) == (c.kind == 1);
            std::vector<std::vector<uint64_t>> ranges = {{}, {c.z - 1}};
            if(c.z > 2) ranges.push_back({1, c.z - 1});
            if(c.z > 1) ranges.push_back({0});
            std::string bad = made ? "" : " fixture";
            for(int t : threadCounts)
                for(const std::vector<uint64_t>& r : ranges)
                    for(uint8_t fl = 0; fl < 2 && made; fl++){
                        omp_set_num_threads(t);
                        const uint64_t first = r.empty() ? 0 : r[0], count = r.empty() ? c.z : r.size() == 2 ? r[1] - r[0] : 1;
                        std::vector<std::vector<uint8_t>> bufs(count);
                        std::vector<void*> table(count);
                        for(uint64_t k = 0; k < count; k++){
                            bufs[count - 1 - k].assign(guard + sliceBytes + (k ? 0 : count*sliceBytes) + guard, poison);
                            table[k] = bufs[count - 1 - k].data() + guard;
                        }
                        step(c.name, "per-slice", "read");
                        bool good = readTiffParallelInfo(path.c_str(), info, table, fl, r);
                        const std::vector<uint8_t>& want = fl ? flip : src;
                        for(uint64_t k = 0; k < count && good; k++){
                            const std::vector<uint8_t>& b = bufs[count - 1 - k];
                            good = std::memcmp(b.data() + guard, want.data() + (first + k)*sliceBytes, sliceBytes) == 0
                                   && std::all_of(b.begin(), b.begin() + guard, isPoison) && std::all_of(b.begin() + guard + sliceBytes, b.end(), isPoison);
                        }
                        if(!good) bad += std::string(" ") + (fl ? "flip" : "no-flip") + "/" + (r.empty() ? "all" : r.size() == 2 ? "range" : "slice")
                                         + std::to_string(first) + "/" + std::to_string(t) + "t";
                    }
            omp_set_num_threads(prevThreads);
            // Refused, with nothing written: a destination too many or too few,
            // a NULL one, and one not aligned to the samples (above 8 bits)
            if(made && c.kind == 0 && c.z > 2){
                step(c.name, "per-slice", "refusals");
                std::vector<std::vector<uint8_t>> bufs(c.z + 1, std::vector<uint8_t>(guard + sliceBytes + guard, poison));
                std::vector<void*> table(c.z + 1);
                for(uint64_t k = 0; k <= c.z; k++) table[k] = bufs[k].data() + guard;
                std::vector<void*> few(table.begin(), table.begin() + (c.z - 1)), withNull(table.begin(), table.begin() + c.z), skewed = withNull;
                withNull[1] = nullptr;
                skewed[2] = static_cast<uint8_t*>(skewed[2]) + 1;
                bool refused = !readTiffParallelInfo(path.c_str(), info, table, 0, {})
                               && !readTiffParallelInfo(path.c_str(), info, few, 1, {})
                               && !readTiffParallelInfo(path.c_str(), info, withNull, 0, {})
                               && !readTiffParallelInfo(path.c_str(), info, std::vector<void*>(table.begin(), table.begin() + 2), 0, {0, 3})
                               && (es == 1 || !readTiffParallelInfo(path.c_str(), info, skewed, 0, {}));
                for(const std::vector<uint8_t>& b : bufs)
                    refused = refused && std::all_of(b.begin(), b.end(), isPoison);
                if(!refused) bad += " refusals";
            }
            std::remove(path.c_str());
            if(bad.empty()) std::printf("PASS  %-16s per-slice destinations\n", c.name);
            else { std::printf("FAIL  %-16s per-slice destinations (%s)\n", c.name, bad.c_str() + 1); failures++; }
        }
    }

    // ---- The two layouts: mode "w" writes every directory, and its strip or
    // tile arrays, before the first slice's data (DirectoriesFirst in
    // parallelwritetiff.cpp), mode "wc" each slice's directory after its data.
    // Both read back as the data, whole and as slice ranges, in both layouts,
    // with one thread and every thread, and getTiffInfo finds the same in both:
    // strips and tiles, LZW, zstd and none, gray and RGB, from row-major and
    // MATLAB-layout data, one slice and many; with "w"'s directories linked
    // slice by slice and 7 at a time (CPPTIFF_LINK_SLICES); "w"'s file the same
    // byte for byte with every directory written by libtiff instead of copied
    // (CPPTIFF_COPY_DIRECTORIES=0); and then with two slices added to each with
    // "a".
    for(const char* link : {"", "7"}){
#ifdef _WIN32
        _putenv_s("CPPTIFF_LINK_SLICES", link);
#else
        if(*link) setenv("CPPTIFF_LINK_SLICES", link, 1);
        else unsetenv("CPPTIFF_LINK_SLICES");
#endif
        struct DF { const char* name; uint64_t x, y, z, spp, bits; const char* comp; uint64_t tile; bool transpose; };
        const DF dfCases[] = {
            {"df-u16-lzw",   40, 24, 9, 1, 16, "lzw",  0,  false},
            {"df-u8-none",   40, 24, 9, 1,  8, "none", 0,  false},
            {"df-f32-zstd",  40, 24, 9, 1, 32, "zstd", 0,  true},
            {"df-u16-tiled", 40, 24, 5, 1, 16, "lzw",  16, false},
            {"df-rgb8-zstd", 40, 24, 5, 3,  8, "zstd", 0,  false},
            {"df-u16-one",   40, 24, 1, 1, 16, "lzw",  0,  false},
            {"df-u16-many", 33, 17, 600, 1, 16, "none", 0, false},
        };
        const int prevThreads = omp_get_max_threads();
        for(const DF& c : dfCases){
            total++;
            const uint64_t es = c.bits/8, sliceBytes = c.x*c.y*c.spp*es, zAll = c.z + 2;
            std::vector<uint8_t> src(sliceBytes*zAll);
            uint64_t st = 0x9E3779B97F4A7C15ull ^ (c.x*131 + c.y*7 + c.z*5 + c.bits*3 + c.spp);
            for(uint8_t& b : src){ st = st*6364136223846793005ull + 1442695040888963407ull; b = (uint8_t)(st >> 59); }
            // (the stack, then two more slices to add)
            const std::vector<uint8_t> first(src.begin(), src.begin() + sliceBytes*c.z), more(src.begin() + sliceBytes*c.z, src.end());
            const std::vector<uint8_t> flip = flipped(first.data(), c.x, c.y, c.z, c.spp, es);
            const std::vector<uint8_t> flipMore = flipped(more.data(), c.x, c.y, 2, c.spp, es);
            const std::vector<uint8_t> flipAll = flipped(src.data(), c.x, c.y, zAll, c.spp, es);
            const std::string pc = dir + "/" + c.name + "-wc.tif", pf = dir + "/" + c.name + "-w.tif";
            const uint16_t fmt = c.bits == 32 ? 3 : 1;
            step(c.name, "layouts", "write both ways");
            const std::vector<uint8_t>& in = c.transpose ? flip : first;
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, pc.c_str(), in.data(), c.bits, 0, 8, "wc", c.transpose, c.comp, fmt, c.tile, c.spp)
                      && !writeTiffParallelWrapper(c.x, c.y, c.z, pf.c_str(), in.data(), c.bits, 0, 8, "w", c.transpose, c.comp, fmt, c.tile, c.spp);
            const std::string pl = dir + "/" + c.name + "-wl.tif";
#ifdef _WIN32
            _putenv_s("CPPTIFF_COPY_DIRECTORIES", "0");
#else
            setenv("CPPTIFF_COPY_DIRECTORIES", "0", 1);
#endif
            bool same = ok && !writeTiffParallelWrapper(c.x, c.y, c.z, pl.c_str(), in.data(), c.bits, 0, 8, "w", c.transpose, c.comp, fmt, c.tile, c.spp);
#ifdef _WIN32
            _putenv_s("CPPTIFF_COPY_DIRECTORIES", "");
#else
            unsetenv("CPPTIFF_COPY_DIRECTORIES");
#endif
            same = same && readFileBytes(pl) == readFileBytes(pf) && !readFileBytes(pf).empty();
            std::remove(pl.c_str());
            // The layouts: "w" every directory before the first slice's data and
            // all the data after the last directory, "wc" each directory after its
            // slice's data
            auto layoutOf = [&](const std::string& f, bool& dirsFirst){
                TIFF* t = TIFFOpen(f.c_str(), "r");
                if(!t) return false;
                uint64_t lastDir = 0, firstDir = UINT64_MAX, minData = UINT64_MAX, dirs = 0;
                do{
                    lastDir = std::max<uint64_t>(lastDir, TIFFCurrentDirOffset(t));
                    firstDir = std::min<uint64_t>(firstDir, TIFFCurrentDirOffset(t));
                    for(uint32_t k = 0; k < TIFFNumberOfStrips(t); k++) minData = std::min<uint64_t>(minData, TIFFGetStrileOffset(t, k));
                    dirs++;
                } while(TIFFReadDirectory(t));
                TIFFClose(t);
                dirsFirst = lastDir < minData;
                return dirs == c.z && (dirsFirst || firstDir > minData);
            };
            bool firstF = false, firstC = true;
            const bool layout = ok && layoutOf(pf, firstF) && layoutOf(pc, firstC) && firstF && !firstC;
            TiffInfo ic, iff;
            ok = ok && !getTiffInfo(pc.c_str(), ic) && !getTiffInfo(pf.c_str(), iff) && iff.z == c.z && iff.x == ic.x && iff.y == ic.y
                 && iff.bits == ic.bits && iff.samples == ic.samples && iff.stripSize == ic.stripSize && iff.compression == ic.compression
                 && iff.tiled == ic.tiled && iff.offsets.size() == c.z && ic.z == c.z;
            const uint64_t a = c.z > 2 ? 1 : 0, e = c.z > 2 ? c.z - 1 : c.z;
            for(const std::string& f : {pf, pc})
                for(int t : {1, prevThreads}){
                    omp_set_num_threads(t);
                    step(c.name, "layouts", "reads");
                    void* nf = ok ? readTiffParallelWrapperNoXYFlip(f.c_str(), {}) : nullptr;
                    void* nr = ok ? readTiffParallelWrapperNoXYFlip(f.c_str(), {a, e}) : nullptr;
                    void* fl = ok ? readTiffParallelWrapper(f.c_str()) : nullptr;
                    void* fr = ok ? readTiffParallelWrapper(f.c_str(), {a, e}) : nullptr;
                    ok = nf && std::memcmp(nf, first.data(), first.size()) == 0
                         && nr && std::memcmp(nr, first.data() + a*sliceBytes, (e - a)*sliceBytes) == 0
                         && fl && std::memcmp(fl, flip.data(), flip.size()) == 0
                         && fr && std::memcmp(fr, flip.data() + a*sliceBytes, (e - a)*sliceBytes) == 0;
                    free(nf); free(nr); free(fl); free(fr);
                }
            omp_set_num_threads(prevThreads);
            // Two slices added to each with "a"
            step(c.name, "layouts", "append");
            const std::vector<uint8_t>& inMore = c.transpose ? flipMore : more;
            bool appended = ok;
            for(const std::string& f : {pf, pc}){
                appended = appended && !writeTiffParallelWrapper(c.x, c.y, 2, f.c_str(), inMore.data(), c.bits, 0, 8, "a", c.transpose, c.comp, fmt, c.tile, c.spp);
                void* nf = appended ? readTiffParallelWrapperNoXYFlip(f.c_str(), {}) : nullptr;
                void* fl = appended ? readTiffParallelWrapper(f.c_str()) : nullptr;
                TiffInfo ia;
                appended = nf && fl && !getTiffInfo(f.c_str(), ia) && ia.z == zAll && std::memcmp(nf, src.data(), src.size()) == 0
                           && std::memcmp(fl, flipAll.data(), flipAll.size()) == 0;
                free(nf); free(fl);
            }
            std::remove(pc.c_str()); std::remove(pf.c_str());
            const std::string how = *link ? std::string(", linked ") + link + " at a time" : "";
            if(ok && layout && same && appended) std::printf("PASS  %-16s both layouts%s\n", c.name, how.c_str());
            else { std::printf("FAIL  %-16s both layouts%s  (read back=%d layout=%d same as libtiff's=%d appended=%d)\n", c.name, how.c_str(), (int)ok, (int)layout, (int)same, (int)appended); failures++; }
        }
    }
#ifdef _WIN32
    _putenv_s("CPPTIFF_LINK_SLICES", "");
#else
    unsetenv("CPPTIFF_LINK_SLICES");

    // ---- Writes cut short, in both layouts: by a full disk (a file-size limit
    // in this process) and killed (SIGKILL: the test runs itself as the writer,
    // --write-child, and kills it partway). The file must then hold exactly the
    // slices that were finished: getTiffInfo's slices each read back as written,
    // libtiff walks no other directory, and every strip it finds is in the file.
    // Stacks of 2,000 slices of 64 x 64 (whose strips wait in the writer's
    // queue) and of 120 of 512 x 512, LZW; mode "w" with its slices linked as it
    // links them (1 MB at a time) and one at a time (CPPTIFF_LINK_SLICES=1).
    // Mode "w"'s files then take two more slices with "a", after those left.
    {
        struct CS { const char* name; uint64_t x, y, z; };
        const CS csCases[] = {{"cut-tiny", 64, 64, 2000}, {"cut-big", 512, 512, 120}};
        struct rlimit oldLimit;
        getrlimit(RLIMIT_FSIZE, &oldLimit);
        void (*oldHandler)(int) = std::signal(SIGXFSZ, SIG_IGN);
        for(const CS& c : csCases)
            for(const char* how : {"w", "w1", "wc"}){
                total++;
                const std::string mode = how[1] == 'c' ? "wc" : "w";
                if(how[1] == '1') setenv("CPPTIFF_LINK_SLICES", "1", 1);
                else unsetenv("CPPTIFF_LINK_SLICES");
                const std::string f = dir + "/" + c.name + "-" + how + ".tif";
                std::string why;
                uint64_t found = 0;
                bool ok = true;
                // (mode "w": two slices added with "a" to what is left, which
                // then holds them too, after the slices it held)
                auto addTwo = [&](const std::string& when){
                    if(mode != "w") return true;
                    std::vector<uint16_t> more(2*c.x*c.y);
                    for(uint64_t k = 0; k < 2; k++){
                        const std::vector<uint16_t> sl = cutSlice(c.x, c.y, found + k);
                        std::memcpy(&more[k*c.x*c.y], sl.data(), sl.size()*2);
                    }
                    uint64_t now = 0;
                    if(writeTiffParallelWrapper(c.x, c.y, 2, f.c_str(), more.data(), 16, 0, 0, "a", false, "lzw", 1, 0)){
                        why = when + ": two slices could not be added to the " + std::to_string(found) + " left";
                        return false;
                    }
                    if(!onlyWholeSlices(f, c.x, c.y, c.z + 2, now, why) || now != found + 2){
                        why = when + ": with two slices added to the " + std::to_string(found) + " left, " + (why.empty() ? std::to_string(now) + " slices" : why);
                        return false;
                    }
                    return true;
                };
                // whole: the file's size, and the time a child takes to make it
                step(c.name, how, "whole");
                const std::vector<uint16_t> data = cutStack(c.x, c.y, c.z);
                ok = !writeTiffParallelWrapper(c.x, c.y, c.z, f.c_str(), data.data(), 16, 0, 0, mode.c_str(), false, "lzw", 1, 0)
                     && onlyWholeSlices(f, c.x, c.y, c.z, found, why) && found == c.z;
                if(!ok) why = "the whole write: " + why;
                FILE* fp = ok ? std::fopen(f.c_str(), "rb") : NULL;
                uint64_t size = 0;
                if(fp){ std::fseek(fp, 0, SEEK_END); size = (uint64_t)std::ftell(fp); std::fclose(fp); }
                // a full disk at 10%, 45% and 80% of the file
                std::string seen;
                for(double frac : {0.10, 0.45, 0.80}){
                    if(!ok) break;
                    step(c.name, how, "disk full");
                    std::remove(f.c_str());
                    struct rlimit lim = oldLimit;
                    lim.rlim_cur = (rlim_t)(size*frac);
                    setrlimit(RLIMIT_FSIZE, &lim);
                    const bool failedAsItShould = writeTiffParallelWrapper(c.x, c.y, c.z, f.c_str(), data.data(), 16, 0, 0, mode.c_str(), false, "lzw", 1, 0) != 0;
                    setrlimit(RLIMIT_FSIZE, &oldLimit);
                    ok = failedAsItShould && onlyWholeSlices(f, c.x, c.y, c.z, found, why);
                    if(!failedAsItShould) why = "a write past the limit reported success";
                    else if(!ok) why = "disk full at " + std::to_string(frac) + ": " + why;
                    seen += " " + std::to_string(found);
                    ok = ok && addTwo("disk full at " + std::to_string(frac));
                }
                // killed at a quarter, half, three quarters and 95% of a child's run
                const double whole = ok ? runChild(selfPath, f, mode.c_str(), c.x, c.y, c.z, -1) : 0;
                ok = ok && whole > 0 && onlyWholeSlices(f, c.x, c.y, c.z, found, why) && found == c.z;
                for(double frac : {0.25, 0.5, 0.75, 0.95}){
                    if(!ok) break;
                    step(c.name, how, "killed");
                    std::remove(f.c_str());
                    runChild(selfPath, f, mode.c_str(), c.x, c.y, c.z, whole*frac);
                    ok = onlyWholeSlices(f, c.x, c.y, c.z, found, why);
                    if(!ok) why = "killed at " + std::to_string(frac) + ": " + why;
                    seen += " " + std::to_string(found);
                    ok = ok && addTwo("killed at " + std::to_string(frac));
                }
                std::remove(f.c_str());
                const std::string label = mode + (how[1] == '1' ? ", linked one at a time" : "");
                if(ok) std::printf("PASS  %-16s cut short, mode %s (slices left:%s)\n", c.name, label.c_str(), seen.c_str());
                else { std::printf("FAIL  %-16s cut short, mode %s  (%s)\n", c.name, label.c_str(), why.c_str()); failures++; }
            }
        unsetenv("CPPTIFF_LINK_SLICES");
        std::signal(SIGXFSZ, oldHandler);
    }
#endif

    // ---- Directories unlike the first. The readers plan a read from the first
    // directory, and a directory laid out otherwise read wrong data with no
    // error (more rows per strip in the first, planar after chunky) or failed
    // (tiles after strips, tiles of another size); one of another size or
    // pixel read as garbage or overflowed the scanline fallback. Now one laid
    // out otherwise reads right, by its own layout, and one that does not fit
    // fails the read, writing nothing past the output: whole stacks and slice
    // ranges, in both layouts, with 1 thread and 8, stacks of a few large
    // slices (read slice by slice, or in pieces) included, and the scanline
    // fallbacks called directly.
    {
        struct MX { const char* name; std::vector<MixedDir> dirs; bool fits; };
        const MixedDir g16{40, 24, 16, 1, 8, 0, COMPRESSION_LZW, false};
        auto with = [](MixedDir d, std::function<void(MixedDir&)> f){ f(d); return d; };
        const std::vector<MX> mxCases = {
            {"mx-rps-larger",    {with(g16, [](MixedDir& d){ d.rps = 24; }), g16, g16}, true},
            {"mx-rps-smaller",   {g16, with(g16, [](MixedDir& d){ d.rps = 24; }), with(g16, [](MixedDir& d){ d.rps = 3; })}, true},
            {"mx-rps-none",      {with(g16, [](MixedDir& d){ d.rps = 24; d.comp = COMPRESSION_NONE; }), with(g16, [](MixedDir& d){ d.comp = COMPRESSION_NONE; }),
                                  with(g16, [](MixedDir& d){ d.rps = 1; d.comp = COMPRESSION_NONE; })}, true},
            {"mx-codecs",        {g16, with(g16, [](MixedDir& d){ d.comp = COMPRESSION_NONE; }), with(g16, [](MixedDir& d){ d.comp = COMPRESSION_ZSTD; })}, true},
            {"mx-codecs-none",   {with(g16, [](MixedDir& d){ d.comp = COMPRESSION_NONE; }), g16, with(g16, [](MixedDir& d){ d.comp = COMPRESSION_ZSTD; })}, true},
            {"mx-tiles-strips",  {g16, with(g16, [](MixedDir& d){ d.tile = 16; }), g16}, true},
            {"mx-strips-tiles",  {with(g16, [](MixedDir& d){ d.tile = 16; }), g16, with(g16, [](MixedDir& d){ d.tile = 32; })}, true},
            {"mx-planar",        {with(g16, [](MixedDir& d){ d.spp = 3; }), with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; }),
                                  with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; d.tile = 16; })}, true},
            {"mx-chunky",        {with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; }), with(g16, [](MixedDir& d){ d.spp = 3; d.rps = 5; }),
                                  with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; d.rps = 24; })}, true},
            {"mx-planar-strips", {with(g16, [](MixedDir& d){ d.spp = 3; }), with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; }),
                                  with(g16, [](MixedDir& d){ d.spp = 3; })}, true},
            {"mx-chunky-strips", {with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; }), with(g16, [](MixedDir& d){ d.spp = 3; d.planar = true; }),
                                  with(g16, [](MixedDir& d){ d.spp = 3; })}, true},
            {"mx-pieces",        {MixedDir{1024, 1024, 16, 1, 64, 0, COMPRESSION_LZW, false}, MixedDir{1024, 1024, 16, 1, 16, 0, COMPRESSION_LZW, false},
                                  MixedDir{1024, 1024, 16, 1, 64, 0, COMPRESSION_ZSTD, false}}, true},
            {"mx-slice-by-slice",{MixedDir{1024, 1024, 32, 1, 64, 0, COMPRESSION_LZW, false}, MixedDir{1024, 1024, 32, 1, 16, 0, COMPRESSION_LZW, false}}, true},
            {"mx-smaller",       {g16, with(g16, [](MixedDir& d){ d.x = 20; d.y = 12; }), g16}, false},
            {"mx-larger",        {g16, with(g16, [](MixedDir& d){ d.x = 80; d.y = 48; }), g16}, false},
            {"mx-larger-raw",    {with(g16, [](MixedDir& d){ d.comp = COMPRESSION_NONE; d.rps = 4; }),
                                  with(g16, [](MixedDir& d){ d.comp = COMPRESSION_NONE; d.x = 80; d.y = 48; d.rps = 48; }),
                                  with(g16, [](MixedDir& d){ d.comp = COMPRESSION_NONE; })}, false},
            {"mx-bits",          {g16, with(g16, [](MixedDir& d){ d.bits = 32; }), g16}, false},
            {"mx-samples",       {g16, with(g16, [](MixedDir& d){ d.spp = 3; }), g16}, false},
        };
        const int prevThreads = omp_get_max_threads();
        for(const MX& c : mxCases){
            total++;
            const MixedDir& d0 = c.dirs[0];
            const uint64_t es = d0.bits/8, sliceBytes = (uint64_t)d0.x*d0.y*d0.spp*es, z = c.dirs.size();
            std::vector<std::vector<uint8_t>> data;
            uint64_t st = 0x51AF0DD5EEDull ^ (uint64_t)std::strlen(c.name);
            for(const MixedDir& d : c.dirs){
                data.emplace_back((uint64_t)d.x*d.y*d.spp*(d.bits/8));
                for(uint8_t& b : data.back()){ st = st*6364136223846793005ull + 1442695040888963407ull; b = (uint8_t)(st >> 59); }
            }
            // What the read of the slices that fit returns (dir 1 is the odd one out of those that do not)
            std::vector<uint8_t> src, flip;
            for(uint64_t k = 0; k < z; k++) src.insert(src.end(), data[k].begin(), data[k].end());
            if(c.fits) flip = flipped(src.data(), d0.x, d0.y, z, d0.spp, es);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "mixed dirs", "write fixture");
            const bool made = writeMixedTiff(path, c.dirs, data);
            TiffInfo info;
            std::string bad = made && !getTiffInfo(path.c_str(), info) && info.z == z ? "" : " fixture";
            std::vector<std::vector<uint64_t>> ranges = {{}, {1}, {1, z}, {z - 1}};
            for(int t : {1, 8})
                for(const std::vector<uint64_t>& r : ranges)
                    for(uint8_t fl = 0; fl < 2 && bad.empty(); fl++){
                        omp_set_num_threads(t);
                        const uint64_t first = r.empty() ? 0 : r[0], count = r.empty() ? z : r.size() == 2 ? r[1] - r[0] : 1;
                        // (the odd directory of those that do not fit is 1; a read without it reads right)
                        const bool shouldRead = c.fits || first > 1 || first + count <= 1;
                        std::vector<uint8_t> out(sliceBytes*count + 4096, 0xA5);
                        step(c.name, "mixed dirs", "read");
                        const bool read = readTiffParallelInfo(path.c_str(), info, out.data(), fl, r) != NULL;
                        bool good = read == shouldRead && std::all_of(out.begin() + sliceBytes*count, out.end(), [](uint8_t v){ return v == 0xA5; });
                        if(good && read){
                            if(c.fits) good = std::memcmp(out.data(), (fl ? flip : src).data() + first*sliceBytes, sliceBytes*count) == 0;
                            else{
                                const std::vector<uint8_t> f = flipped(data[first].data(), d0.x, d0.y, 1, d0.spp, es);
                                good = count == 1 && std::memcmp(out.data(), (fl ? f : data[first]).data(), sliceBytes) == 0;
                            }
                        }
                        if(!good) bad += std::string(" ") + (fl ? "flip" : "C") + "/" + (r.empty() ? "all" : r.size() == 2 ? "range" : "slice")
                                         + std::to_string(first) + "/" + std::to_string(t) + "t";
                    }
            // The per-slice form, and the scanline fallbacks, of the whole stack
            if(bad.empty()){
                omp_set_num_threads(prevThreads);
                step(c.name, "mixed dirs", "per-slice read and fallbacks");
                std::vector<std::vector<uint8_t>> bufs(z, std::vector<uint8_t>(sliceBytes + 64, 0xA5));
                std::vector<void*> table;
                for(auto& b : bufs) table.push_back(b.data());
                bool good = readTiffParallelInfo(path.c_str(), info, table, 0, {}) == c.fits;
                for(uint64_t k = 0; k < z && good; k++)
                    good = std::all_of(bufs[k].begin() + sliceBytes, bufs[k].end(), [](uint8_t v){ return v == 0xA5; })
                           && (!c.fits || std::memcmp(bufs[k].data(), data[k].data(), sliceBytes) == 0);
                for(uint8_t fl = 0; fl < 2 && good; fl++){
                    std::vector<uint8_t> out(sliceBytes*z + 4096, 0xA5);
                    const bool read = !readTiffParallelBak(d0.x, d0.y, z, path.c_str(), out.data(), d0.bits, 0, fl);
                    good = read == c.fits && std::all_of(out.begin() + sliceBytes*z, out.end(), [](uint8_t v){ return v == 0xA5; })
                           && (!c.fits || std::memcmp(out.data(), (fl ? flip : src).data(), sliceBytes*z) == 0);
                    // (the single-image fallback, of the odd directory)
                    std::vector<uint8_t> one(sliceBytes + 4096, 0xA5);
                    const bool read1 = !readTiffParallel2DBak(d0.x, d0.y, 1, path.c_str(), one.data(), d0.bits, 1, fl);
                    const bool strips1 = c.dirs[1].tile == 0;
                    good = good && std::all_of(one.begin() + sliceBytes, one.end(), [](uint8_t v){ return v == 0xA5; })
                           && (c.fits && strips1 ? read1 && std::memcmp(one.data(), (fl ? flip : src).data() + sliceBytes, sliceBytes) == 0 : !c.fits ? !read1 : true);
                }
                if(!good) bad += " per-slice/fallbacks";
            }
            omp_set_num_threads(prevThreads);
            std::remove(path.c_str());
            if(bad.empty()) std::printf("PASS  %-18s mixed directories\n", c.name);
            else { std::printf("FAIL  %-18s mixed directories (%s)\n", c.name, bad.c_str() + 1); failures++; }
        }
    }

    // ---- Pieces of slices. A compressed stack of large slices with fewer than
    // 8 slices per thread is read in pieces of whole strips (or tiles) handed
    // out to the threads; every piece must land in its place, for strips,
    // tiles, color and planar color, whole stacks and z-ranges. With 11 strips
    // per slice the pieces differ in size (6 + 5 strips on 5 threads).
    {
        struct PC { const char* name; uint64_t x, y, z, spp; const char* comp; uint64_t strip, tile; bool planar; };
        const PC pcs[] = {
            {"pc-lzw-strips", 1024, 1024, 7, 1, "lzw",   64,   0, false},
            {"pc-lzw-11",     1024, 1100, 7, 1, "lzw",  100,   0, false},   // unequal pieces
            {"pc-zstd-tiles", 1040, 1000, 7, 1, "zstd",   0, 128, false},   // ragged tiles
            {"pc-rgb-lzw",    1024,  600, 5, 3, "lzw",   50,   0, false},
            {"pc-planar-lzw", 1024,  600, 5, 3, "lzw",   40,   0, true},
        };
        std::mt19937 qrng(2468u);
        const int prevThreads = omp_get_max_threads();
        for(const PC& c : pcs){
            total++;
            const uint64_t slice = c.x*c.y*c.spp*2;   // 16-bit
            std::vector<uint8_t> src(slice*c.z);
            for(uint64_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(qrng() & 0x3F);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "slice pieces", "write fixture");
            bool ok = c.planar ? writeLibtiff(path, src.data(), c.x, c.y, c.z, c.spp, 16, 1, COMPRESSION_LZW, (uint32_t)c.strip, 0, true)
                               : !writeTiffParallelWrapper(c.x, c.y, c.z, path.c_str(), src.data(), 16, 0, c.strip, "w", false, std::string(c.comp), 1, c.tile, c.spp);
            struct Read { std::vector<uint64_t> range; uint64_t first, count; };
            const Read reads[] = {{{}, 0, c.z}, {{1, c.z - 1}, 1, c.z - 2}, {{2, c.z}, 2, c.z - 2}};
            for(int t : {2, 3, 5, 6, 8}){
                omp_set_num_threads(t);
                step(c.name, "slice pieces", "full and z-range reads, both layouts");
                for(const Read& r : reads){
                    if(!ok) break;
                    const uint8_t* want = src.data() + r.first*slice;
                    const std::vector<uint8_t> wantFlip = flipped(want, c.x, c.y, r.count, c.spp, 2);
                    void* nf = readTiffParallelWrapperNoXYFlip(path.c_str(), r.range);
                    void* fl = readTiffParallelWrapper(path.c_str(), r.range);
                    ok = nf && fl && std::memcmp(nf, want, r.count*slice) == 0 && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
                    free(nf); free(fl);
                }
            }
            omp_set_num_threads(prevThreads);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-16s slice pieces\n", c.name);
            else { std::printf("FAIL  %-16s slice pieces  (read back wrong)\n", c.name); failures++; }
        }
    }

    // ---- Color writes: RGB (3 samples) and RGBA (4), each pixel's samples
    // together (chunky). Row-major (z, y, x, c) and MATLAB-layout (y, x, c, z)
    // input must give the same file, as must one thread, and the file must read
    // back in both layouts. The tags are RGB, chunky, and unassociated alpha
    // for RGBA.
    {
        struct CW { const char* name; uint64_t x, y, z, spp, bits; uint16_t fmt; const char* comp; uint64_t strip, tile; };
        const CW cws[] = {
            {"rgb8-none-s7",      40, 41, 3, 3,  8, 1, "none",   7,  0},
            {"rgb8-lzw-auto",    333, 77, 1, 3,  8, 1, "lzw",    0,  0},   // 2D, automatic strips
            {"rgba16-zstd-t16",   50, 37, 2, 4, 16, 1, "zstd", 512, 16},   // ragged tiles
            {"rgbf32-lzw-t32",    70, 45, 3, 3, 32, 3, "lzw",  512, 32},
            {"rgbaf64-none-s5",   61, 29, 2, 4, 64, 3, "none",   5,  0},
            {"rgbi16-zstd-s512", 333, 77, 4, 3, 16, 2, "zstd", 512,  0},
            {"rgba8-none-t16",    33, 70, 2, 4,  8, 1, "none", 512, 16},
        };
        std::mt19937 crng(8642u);
        const int prevThreads = omp_get_max_threads();
        for(const CW& c : cws){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> rm(slice*c.z);
            for(auto& v8 : rm) v8 = (uint8_t)(crng() & 0x3F);   // compressible, and floats stay finite
            const std::vector<uint8_t> cm = flipped(rm.data(), c.x, c.y, c.z, c.spp, es);
            const std::string pa = dir + "/cw_row_" + c.name + ".tif", pb = dir + "/cw_matlab_" + c.name + ".tif",
                              pc = dir + "/cw_one_" + c.name + ".tif";
            step(c.name, "color writes", "row-major, MATLAB-layout and one-thread writes");
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, pa.c_str(), rm.data(), c.bits, 0, c.strip, "w", false, std::string(c.comp), c.fmt, c.tile, c.spp)
                      && !writeTiffParallelWrapper(c.x, c.y, c.z, pb.c_str(), cm.data(), c.bits, 0, c.strip, "w", true, std::string(c.comp), c.fmt, c.tile, c.spp);
            omp_set_num_threads(1);
            ok = ok && !writeTiffParallelWrapper(c.x, c.y, c.z, pc.c_str(), cm.data(), c.bits, 0, c.strip, "w", true, std::string(c.comp), c.fmt, c.tile, c.spp);
            omp_set_num_threads(prevThreads);
            const std::vector<uint8_t> fa = readFileBytes(pa), fb = readFileBytes(pb), fc = readFileBytes(pc);
            const bool same = ok && !fa.empty() && fa == fb && fa == fc;
            uint64_t spp = 0, photometric = 0, planarConfig = 0, extra = 0;
            const bool tags = firstTag(fa, 277, &spp) && spp == c.spp && firstTag(fa, 262, &photometric) && photometric == 2
                              && firstTag(fa, 284, &planarConfig) && planarConfig == 1
                              && (c.spp == 4 ? firstTag(fa, 338, &extra) && extra == 2 : !firstTag(fa, 338, &extra))
                              && getSamplesPerPixel(pa.c_str()) == c.spp;
            step(c.name, "color writes", "read back in both layouts and a z-range");
            void* nf = same ? readTiffParallelWrapperNoXYFlip(pa.c_str(), {}) : nullptr;
            void* fl = same ? readTiffParallelWrapper(pa.c_str()) : nullptr;
            bool back = nf && fl && std::memcmp(nf, rm.data(), rm.size()) == 0 && std::memcmp(fl, cm.data(), cm.size()) == 0;
            free(nf); free(fl);
            if(back && c.z > 1){
                void* part = readTiffParallelWrapperNoXYFlip(pa.c_str(), {1, c.z});
                back = part && std::memcmp(part, rm.data() + slice, slice*(c.z - 1)) == 0;
                free(part);
            }
            std::remove(pa.c_str()); std::remove(pb.c_str()); std::remove(pc.c_str());
            if(same && tags && back) std::printf("PASS  %-16s color writes\n", c.name);
            else { std::printf("FAIL  %-16s color writes  (files same=%d tags=%d read back=%d)\n", c.name, (int)same, (int)tags, (int)back); failures++; }
        }
        // Other sample counts are refused rather than written with RGB tags.
        total++;
        step("bad-samples", "color writes", "2 and 5 samples per pixel");
        std::vector<uint8_t> small(16*16*5);
        const std::string bp = dir + "/bad_samples.tif";
        const bool refused = writeTiffParallelWrapper(16, 16, 1, bp.c_str(), small.data(), 8, 0, 0, "w", false, "lzw", 1, 0, 2) != 0
                             && writeTiffParallelWrapper(16, 16, 1, bp.c_str(), small.data(), 8, 0, 0, "w", false, "lzw", 1, 0, 5) != 0;
        std::remove(bp.c_str());
        if(refused) std::printf("PASS  %-16s color writes\n", "bad-samples");
        else { std::printf("FAIL  %-16s color writes  (2 or 5 samples per pixel accepted)\n", "bad-samples"); failures++; }
    }

    // ---- Planar color (PLANARCONFIG=2), each color stored as its own image.
    // Every reader returns the same layouts as for chunky color: (z, y, x, c)
    // no-flip and (y, x, c, z) flip. libtiff writes the fixtures in several
    // codecs, as strips of one row, several rows or a whole plane and as tiles,
    // so the per-strip, back-to-back uncompressed (raw) and tiled paths all
    // run, along with 2D images, z-ranges and the scanline fallbacks.
    {
        struct PL { const char* name; uint64_t x, y, z, spp, bits; uint16_t fmt, comp; uint32_t rps, tile; bool be; };
        const PL pls[] = {
            {"pl-rgb8-lzw-r5",      40,  41, 3, 3,  8, 1, COMPRESSION_LZW,           5,           0, false},
            {"pl-rgba16-zstd-t16",  50,  37, 3, 4, 16, 1, COMPRESSION_ZSTD,          0,          16, false},
            {"pl-rgb16-none-r7",   300,  41, 3, 3, 16, 1, COMPRESSION_NONE,          7,           0, false},
            {"pl-rgb16-none-be",   300,  41, 3, 3, 16, 1, COMPRESSION_NONE,          3,           0, true},
            {"pl-rgbf32-zip-r1",    33,  29, 2, 3, 32, 3, COMPRESSION_ADOBE_DEFLATE, 1,           0, false},
            {"pl-rgbaf64-none-1s",  45,  23, 2, 4, 64, 3, COMPRESSION_NONE,          0xFFFFFFFFu, 0, false},   // one strip per color
            {"pl-2ch-u16-lzw-t32",  70,  45, 2, 2, 16, 1, COMPRESSION_LZW,           0,          32, false},
            {"pl-rgb8-pack-2d",    130, 200, 1, 3,  8, 1, COMPRESSION_PACKBITS,     64,           0, false},
            {"pl-rgba8-none-2d",   300, 100, 1, 4,  8, 1, COMPRESSION_NONE,         10,           0, false},
        };
        std::mt19937 prng(97531u);
        for(const PL& c : pls){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            std::vector<uint8_t> src(slice*c.z);
            for(auto& v8 : src) v8 = (uint8_t)prng();
            if(c.bits == 32) for(uint64_t i = 0; i < src.size(); i += 4) src[i + 3] &= 0x3F;   // keep floats finite
            if(c.bits == 64) for(uint64_t i = 0; i < src.size(); i += 8) src[i + 7] &= 0x3F;
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "planar", "write fixture");
            bool ok = writeLibtiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, c.fmt, c.comp, c.rps, c.tile, true, c.be);
            uint64_t* dims = ok ? getImageSize(path.c_str()) : nullptr;
            ok = ok && dims && dims[0] == c.y && dims[1] == c.x && dims[2] == c.z && getSamplesPerPixel(path.c_str()) == c.spp;
            free(dims);
            step(c.name, "planar", "full, z-range and single-slice reads, both layouts");
            struct Read { std::vector<uint64_t> range; uint64_t first, count; };
            std::vector<Read> reads = {{{}, 0, c.z}};
            if(c.z > 1){ reads.push_back({{1, c.z}, 1, c.z - 1}); reads.push_back({{c.z - 1}, c.z - 1, 1}); }
            for(const Read& r : reads){
                if(!ok) break;
                const uint8_t* want = src.data() + r.first*slice;
                const std::vector<uint8_t> wantFlip = flipped(want, c.x, c.y, r.count, c.spp, es);
                void* nf = readTiffParallelWrapperNoXYFlip(path.c_str(), r.range);
                void* fl = readTiffParallelWrapper(path.c_str(), r.range);
                ok = nf && fl && std::memcmp(nf, want, r.count*slice) == 0 && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
                free(nf); free(fl);
            }
            // The scanline fallbacks read strips only; libtiff has no scanlines for tiles.
            bool fb = true;
            if(ok && !c.tile){
                step(c.name, "planar", "scanline fallbacks");
                const uint64_t last = c.z - 1;
                const std::vector<uint8_t> lastSlice(src.begin() + last*slice, src.end());
                fb = fallbackMatches(false, path, c.x, c.y, c.z, c.spp, c.bits, 0, false, src)
                     && fallbackMatches(false, path, c.x, c.y, c.z, c.spp, c.bits, 0, true, flipped(src.data(), c.x, c.y, c.z, c.spp, es))
                     && fallbackMatches(true, path, c.x, c.y, 1, c.spp, c.bits, last, false, lastSlice)
                     && fallbackMatches(true, path, c.x, c.y, 1, c.spp, c.bits, last, true, flipped(lastSlice.data(), c.x, c.y, 1, c.spp, es));
            }
            std::remove(path.c_str());
            if(ok && fb) std::printf("PASS  %-16s planar\n", c.name);
            else { std::printf("FAIL  %-16s planar  (reads=%d fallbacks=%d)\n", c.name, (int)ok, (int)fb); failures++; }
        }

        // Many slices with few threads (on Linux each thread reads its own block
        // and then takes slices from the others), and short stacks of large
        // slices, which are read slice by slice with every thread.
        const int prevThreads = omp_get_max_threads();
        struct PT { const char* name; uint64_t x, y, z; uint16_t comp; uint32_t rps; std::vector<int> threads; };
        const PT pts[] = {
            {"pl-many-none",    24,   17, 37, COMPRESSION_NONE,    5, {2, 3, 5, prevThreads}},
            {"pl-many-lzw",     24,   17, 37, COMPRESSION_LZW,     5, {2, 3, 5, prevThreads}},
            {"pl-short-lzw",  1200, 1200,  2, COMPRESSION_LZW,    64, {8}},   // 4.3 MB slices
            {"pl-short-none", 2400, 2400,  2, COMPRESSION_NONE, 2400, {8}},   // 17 MB slices, one strip per color
        };
        for(const PT& c : pts){
            total++;
            const uint64_t slice = c.x*c.y*3;
            std::vector<uint8_t> src(slice*c.z);
            for(uint64_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(prng() & 0x7F);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "planar", "write fixture");
            bool ok = writeLibtiff(path, src.data(), c.x, c.y, c.z, 3, 8, 1, c.comp, c.rps, 0, true);
            const std::vector<uint8_t> wantFlip = flipped(src.data(), c.x, c.y, c.z, 3, 1);
            const std::vector<uint8_t> wantTailFlip = flipped(src.data() + slice, c.x, c.y, c.z - 1, 3, 1);
            for(int t : c.threads){
                if(!ok) break;
                omp_set_num_threads(t);
                step(c.name, "planar", "full and z-range reads, both layouts");
                void* nf = readTiffParallelWrapperNoXYFlip(path.c_str(), {});
                void* fl = readTiffParallelWrapper(path.c_str());
                void* rg = readTiffParallelWrapperNoXYFlip(path.c_str(), {1, c.z});
                void* rf = readTiffParallelWrapper(path.c_str(), {1, c.z});
                ok = nf && fl && rg && rf
                     && std::memcmp(nf, src.data(), src.size()) == 0 && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0
                     && std::memcmp(rg, src.data() + slice, slice*(c.z - 1)) == 0 && std::memcmp(rf, wantTailFlip.data(), wantTailFlip.size()) == 0;
                free(nf); free(fl); free(rg); free(rf);
            }
            omp_set_num_threads(prevThreads);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-16s planar\n", c.name);
            else { std::printf("FAIL  %-16s planar  (read back wrong)\n", c.name); failures++; }
        }
    }

    // ---- RowsPerStrip past the image height. With 2^32 - 1, the TIFF default
    // for a single strip, flip reads sized their buffers from it and a
    // compressed single-strip image crashed; planar files, which libtiff never
    // splits into smaller strips, hit it uncompressed too.
    {
        struct RP { const char* name; uint64_t x, y, z, spp, bits; uint16_t comp; uint32_t rps; bool planar; };
        const RP rps[] = {
            {"rps-max-lzw-2d",  300, 200, 1, 1, 16, COMPRESSION_LZW,  0xFFFFFFFFu, false},
            {"rps-max-zstd-3d", 120,  50, 3, 1,  8, COMPRESSION_ZSTD, 0xFFFFFFFFu, false},
            {"rps-big-rgb-lzw",  90,  60, 2, 3,  8, COMPRESSION_LZW,  1000,        false},
            {"rps-max-planar",   90,  60, 2, 3, 16, COMPRESSION_NONE, 0xFFFFFFFFu, true},
        };
        std::mt19937 rrng(1357u);
        for(const RP& c : rps){
            total++;
            const uint64_t es = c.bits/8;
            std::vector<uint8_t> src(c.x*c.y*c.z*c.spp*es);
            for(auto& v8 : src) v8 = (uint8_t)(rrng() & 0x7F);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "rows per strip", "write fixture + reads, both layouts");
            bool ok = writeLibtiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, 1, c.comp, c.rps, 0, c.planar);
            void* nf = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
            void* fl = ok ? readTiffParallelWrapper(path.c_str()) : nullptr;
            const std::vector<uint8_t> cm = flipped(src.data(), c.x, c.y, c.z, c.spp, es);
            ok = nf && fl && std::memcmp(nf, src.data(), src.size()) == 0 && std::memcmp(fl, cm.data(), cm.size()) == 0;
            free(nf); free(fl);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-16s rows per strip\n", c.name);
            else { std::printf("FAIL  %-16s rows per strip  (read back wrong)\n", c.name); failures++; }
        }
    }

    // ---- Write failures must be reported instead of leaving a broken file and
    // returning success.
    {
        const uint64_t fx = 128, fy = 128, fz = 8;
        std::vector<uint16_t> fv(fx*fy*fz);
        for(uint64_t i = 0; i < fv.size(); i++) fv[i] = (uint16_t)(i*2654435761u >> 7);
        // A folder that cannot exist: the path runs through a regular file.
        total++;
        const std::string blocker = dir + "/not_a_folder.txt";
        if(FILE* f = std::fopen(blocker.c_str(), "wb")){ std::fputs("x", f); std::fclose(f); }
        const std::string inside = blocker + "/sub/out.tif";
        step("bad-folder", "write failures", "write below a regular file");
        const bool badFolder = writeTiffParallelHelper(inside.c_str(), fv.data(), 16, "w", fx, fy, fz, 0, false, "lzw", 1, 0) != 0;
        std::remove(blocker.c_str());
        if(badFolder) std::printf("PASS  %-14s write failures\n", "bad-folder");
        else { std::printf("FAIL  %-14s write failures  (reported success)\n", "bad-folder"); failures++; }
#ifdef __linux__
        // A device that rejects every write.
        total++;
        step("dev-full", "write failures", "write to /dev/full");
        const bool devFull = writeTiffParallelWrapper(fx, fy, fz, "/dev/full", fv.data(), 16, 0, 16, "w", false, "none", 1, 0) != 0;
        if(devFull) std::printf("PASS  %-14s write failures\n", "dev-full");
        else { std::printf("FAIL  %-14s write failures  (reported success)\n", "dev-full"); failures++; }
#endif
#ifndef _WIN32
        // Running out of space partway through: lower the file-size limit so
        // writes past it fail (with SIGXFSZ ignored they return an error instead
        // of killing the process), in both the parallel and one-thread writers.
        const std::string partial = dir + "/partial_write.tif";
        struct rlimit oldLimit;
        getrlimit(RLIMIT_FSIZE, &oldLimit);
        void (*oldHandler)(int) = std::signal(SIGXFSZ, SIG_IGN);
        const int prevThreads = omp_get_max_threads();
        const int threadCounts[] = {prevThreads, 1};
        for(int t : threadCounts){
            for(const char* comp : {"none", "zstd"}){
                total++;
                struct rlimit lim = oldLimit;
                lim.rlim_cur = 64*1024;   // the volume is 256 KB uncompressed, well above this compressed
                step("partial", comp, "write past a 64 KB file-size limit");
                omp_set_num_threads(t);
                std::vector<uint16_t> noisy(fv.size());
                std::mt19937 nrng(7u);
                for(auto& v16 : noisy) v16 = (uint16_t)nrng();   // incompressible, so zstd output exceeds the limit too
                setrlimit(RLIMIT_FSIZE, &lim);
                const bool reported = writeTiffParallelWrapper(fx, fy, fz, partial.c_str(), noisy.data(), 16, 0, 16, "w", false, std::string(comp), 1, 0) != 0;
                setrlimit(RLIMIT_FSIZE, &oldLimit);
                omp_set_num_threads(prevThreads);
                std::remove(partial.c_str());
                if(reported) std::printf("PASS  partial-%-6s t%-3d write failures\n", comp, t);
                else { std::printf("FAIL  partial-%-6s t%-3d write failures  (reported success)\n", comp, t); failures++; }
            }
        }
        std::signal(SIGXFSZ, oldHandler);
#endif
    }

    // ---- LZW. The library decodes LZW itself unless a file needs something
    // only libtiff does (a predictor, the other bit order, big-endian samples),
    // so libtiff-written files of each kind are read back in both layouts. And
    // what the library writes must decode with libtiff alone. The data repeats
    // with some noise (long strings, full tables) and ends with a random slice
    // (codes longer than the bytes they stand for).
    {
        std::mt19937 lrng(8642u);
        auto lzwData = [&](uint64_t bytes, uint64_t es, uint64_t slice, uint64_t bits){
            std::vector<uint8_t> v(bytes);
            for(uint64_t i = 0; i < bytes; i++)
                v[i] = (bytes > slice && i >= bytes - slice) ? (uint8_t)lrng() : (uint8_t)(((i/es) % 97)/3 + (lrng() & 3));
            if(bits == 32) for(uint64_t i = 0; i < bytes; i += 4) v[i + 3] &= 0x3F;   // keep floats finite
            return v;
        };
        struct LZ { const char* name; uint64_t x, y, z, spp, bits; uint16_t fmt; uint32_t rps, tile; bool planar, be; uint16_t predictor, fillOrder; };
        const LZ lzs[] = {
            {"lzw-strips",     301, 203, 3, 1, 16, 1, 37,          0, false, false, PREDICTOR_NONE,          FILLORDER_MSB2LSB},
            {"lzw-one-strip",  301, 203, 2, 1,  8, 1, 0xFFFFFFFFu, 0, false, false, PREDICTOR_NONE,          FILLORDER_MSB2LSB},
            {"lzw-tiles",      301, 203, 2, 1, 16, 1, 0,          64, false, false, PREDICTOR_NONE,          FILLORDER_MSB2LSB},
            {"lzw-planar",     301, 203, 2, 3,  8, 1, 50,          0, true,  false, PREDICTOR_NONE,          FILLORDER_MSB2LSB},
            {"lzw-predictor",  301, 203, 3, 1, 16, 1, 37,          0, false, false, PREDICTOR_HORIZONTAL,    FILLORDER_MSB2LSB},
            {"lzw-float-pred", 301, 203, 2, 1, 32, 3, 37,          0, false, false, PREDICTOR_FLOATINGPOINT, FILLORDER_MSB2LSB},
            {"lzw-lsb2msb",    301, 203, 3, 1, 16, 1, 37,          0, false, false, PREDICTOR_NONE,          FILLORDER_LSB2MSB},
            {"lzw-big-endian", 301, 203, 3, 1, 16, 1, 37,          0, false, true,  PREDICTOR_NONE,          FILLORDER_MSB2LSB},
        };
        for(const LZ& c : lzs){
            total++;
            const uint64_t es = c.bits/8, slice = c.x*c.y*c.spp*es;
            const std::vector<uint8_t> src = lzwData(slice*c.z, es, slice, c.bits);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "lzw", "write fixture with libtiff");
            bool ok = writeLibtiff(path, src.data(), c.x, c.y, c.z, c.spp, c.bits, c.fmt, COMPRESSION_LZW, c.rps, c.tile,
                                   c.planar, c.be, c.predictor, c.fillOrder);
            step(c.name, "lzw", "read with the library, both layouts");
            void* nf = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
            void* fl = ok ? readTiffParallelWrapper(path.c_str(), {}) : nullptr;
            const std::vector<uint8_t> wantFlip = flipped(src.data(), c.x, c.y, c.z, c.spp, es);
            ok = nf && fl && std::memcmp(nf, src.data(), src.size()) == 0 && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
            free(nf); free(fl);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-16s lzw read\n", c.name);
            else { std::printf("FAIL  %-16s lzw read  (read back wrong)\n", c.name); failures++; }
        }
        struct LW { const char* name; uint64_t x, y, z, spp, bits; uint16_t fmt; uint64_t tile; };
        const LW lws[] = {
            {"lzw-lib-u16",   1000, 700, 3, 1, 16, 1,   0},
            {"lzw-lib-u8",     777, 333, 2, 1,  8, 1,   0},
            {"lzw-lib-f32",    512, 300, 2, 1, 32, 3,   0},
            {"lzw-lib-rgb",    300, 200, 2, 3,  8, 1,   0},
            {"lzw-lib-tiles",  600, 500, 2, 1, 16, 1, 128},
        };
        for(const LW& c : lws){
            total++;
            const uint64_t es = c.bits/8, px = c.spp*es, slice = c.x*c.y*px;
            const std::vector<uint8_t> src = lzwData(slice*c.z, es, slice, c.bits);
            const std::string path = dir + "/" + c.name + ".tif";
            step(c.name, "lzw", "write with the library");
            bool ok = !writeTiffParallelWrapper(c.x, c.y, c.z, path.c_str(), src.data(), c.bits, 0, 0, "w", false, "lzw", c.fmt, c.tile, c.spp);
            step(c.name, "lzw", "decode with libtiff");
            TIFF* tif = ok ? TIFFOpen(path.c_str(), "r") : NULL;
            ok = tif != NULL;
            std::vector<uint8_t> buf;
            for(uint64_t k = 0; ok && k < c.z; k++){
                ok = TIFFSetDirectory(tif, (tdir_t)k);
                const uint8_t* want = src.data() + k*slice;
                if(ok && c.tile){
                    buf.resize((size_t)TIFFTileSize(tif));
                    for(uint64_t ty = 0; ok && ty < c.y; ty += c.tile)
                        for(uint64_t tx = 0; ok && tx < c.x; tx += c.tile){
                            ok = TIFFReadEncodedTile(tif, TIFFComputeTile(tif, (uint32_t)tx, (uint32_t)ty, 0, 0), buf.data(), (tmsize_t)buf.size()) >= 0;
                            const uint64_t w = std::min<uint64_t>(c.tile, c.x - tx);
                            for(uint64_t r = 0; ok && r < c.tile && ty + r < c.y; r++)
                                ok = std::memcmp(&buf[r*c.tile*px], want + ((ty + r)*c.x + tx)*px, w*px) == 0;
                        }
                }
                else if(ok){
                    buf.resize((size_t)TIFFStripSize(tif));
                    uint64_t at = 0;
                    for(uint32_t st = 0; ok && st < TIFFNumberOfStrips(tif); st++){
                        const tmsize_t n = TIFFReadEncodedStrip(tif, st, buf.data(), (tmsize_t)buf.size());
                        ok = n > 0 && at + (uint64_t)n <= slice && std::memcmp(buf.data(), want + at, (size_t)n) == 0;
                        at += ok ? (uint64_t)n : 0;
                    }
                    ok = ok && at == slice;
                }
            }
            if(tif) TIFFClose(tif);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-16s lzw decoded by libtiff\n", c.name);
            else { std::printf("FAIL  %-16s lzw decoded by libtiff  (differs from what was written)\n", c.name); failures++; }
        }
    }

    // ---- A leading ~ or ~name is that home folder, and nothing else in a path
    // is expanded (expandTilde, which the MATLAB mex files call on Linux and
    // macOS): the wordexp it used expanded any path with a ~ anywhere as a shell
    // word, so it cut the path at spaces, crashed on parentheses, substituted
    // variables and ran commands in $(...) and backquotes. Folders of such names
    // work with the library itself, which expands nothing.
    {
        const std::string ran = dir + "/tilde_ran";
        std::remove(ran.c_str());
#ifndef _WIN32
        const char* h0 = getenv("HOME");
        const std::string home0 = h0 ? h0 : "";
        const bool hadHome = h0 != nullptr;
        auto expect = [&](const std::string& in, const std::string& want){
            total++;
            step("tilde", "expandTilde", in.c_str());
            const std::string got = expandTilde(in);
            if(got == want) std::printf("PASS  %-14s \"%s\"\n", "tilde", in.c_str());
            else { std::printf("FAIL  %-14s \"%s\" gave \"%s\", not \"%s\"\n", "tilde", in.c_str(), got.c_str(), want.c_str()); failures++; }
        };
        // (a user's home folder from the user database, the current user's for nullptr, without trailing slashes)
        auto userHome = [](const char* name, std::string& home, std::string& user){
            std::vector<char> buf(1 << 16);
            struct passwd pw, *found = nullptr;
            const int err = name ? getpwnam_r(name, &pw, buf.data(), buf.size(), &found)
                                 : getpwuid_r(getuid(), &pw, buf.data(), buf.size(), &found);
            if(err || !found || !found->pw_dir) return false;
            home = found->pw_dir;
            user = found->pw_name ? found->pw_name : "";
            while(!home.empty() && home.back() == '/') home.pop_back();
            return true;
        };
        setenv("HOME", "/h/one", 1);
        expect("~", "/h/one");
        expect("~/", "/h/one/");
        expect("~/a b (c)/x~y $HOME `id`.tif", "/h/one/a b (c)/x~y $HOME `id`.tif");
        const std::vector<std::string> kept = {"/data/run~2 (copy)/a.tif", "til~de (x)/a.tif", "a/~/b", "x~y", "", "$HOME/~",
            "~nosuchuser_cpptiff_rt/x", "~ /x", "a~b/$(touch " + ran + ")", "a~b/`touch " + ran + "`", "~$(touch " + ran + ")"};
        for(const std::string& kp : kept) expect(kp, kp);
        setenv("HOME", "/h/two//", 1);
        expect("~/x", "/h/two/x");
        expect("~", "/h/two");
        setenv("HOME", "/", 1);
        expect("~", "/");
        expect("~/x", "/x");
        // (~name is that user's home folder whatever HOME is; without HOME, ~ is the current user's)
        std::string uh, un;
        if(userHome("root", uh, un)){
            expect("~root", uh.empty() ? std::string("/") : uh);
            expect("~root/x y", uh + "/x y");
        }
        unsetenv("HOME");
        if(userHome(nullptr, uh, un)){
            expect("~/x", uh + "/x");
            if(!un.empty()) expect("~" + un + "/x", uh + "/x");
        }
        if(hadHome) setenv("HOME", home0.c_str(), 1);
        else unsetenv("HOME");
#endif
        // Folders of such names, written and read through the library
        const uint64_t tx = 40, ty = 24, tz = 3;
        std::vector<uint16_t> tv(tx*ty*tz);
        for(uint64_t i = 0; i < tv.size(); i++) tv[i] = (uint16_t)(i*977u + 13u);
        std::vector<std::string> names = {"til~de (x)", "run~2 copy", "a~b $HOME `id`", "~c"};
#ifndef _WIN32
        names.push_back("a~b $(touch " + ran + ")");
        names.push_back("x~y `touch " + ran + "`");
#endif
        for(const std::string& nm : names){
            total++;
            const std::string path = dir + "/" + nm + "/a.tif";
            step("tilde-dir", nm.c_str(), "write and read in a folder of such a name");
            bool ok = !writeTiffParallelHelper(path.c_str(), tv.data(), 16, "w", tx, ty, tz, 0, false, "lzw", 1, 0);
            TiffInfo ti;
            ok = ok && !getTiffInfo(path.c_str(), ti) && ti.z == tz;
            void* back = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
            ok = back && std::memcmp(back, tv.data(), tv.size()*2) == 0;
            free(back);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-14s \"%s\"\n", "tilde-dir", nm.c_str());
            else { std::printf("FAIL  %-14s \"%s\"\n", "tilde-dir", nm.c_str()); failures++; }
        }
        total++;
        FILE* rf = std::fopen(ran.c_str(), "rb");
        if(!rf) std::printf("PASS  %-14s no command in a path ran\n", "tilde");
        else { std::fclose(rf); std::printf("FAIL  %-14s a command in a path ran\n", "tilde"); failures++; }
    }

    // ---- Paths are UTF-8 on every platform: on Windows the library opens them with
    // the wide-character API (its narrow functions read a path in the ANSI code
    // page, so that a folder named with an accented or a CJK letter could not be
    // opened or written to). A stack, one slice of it, its size, and an ImageJ
    // stack, in folders named "café µm" and "日本", the file found where its name says.
    {
        const uint64_t ux = 40, uy = 24, uz = 3;
        std::vector<uint16_t> uv(ux*uy*uz);
        for(uint64_t i = 0; i < uv.size(); i++) uv[i] = (uint16_t)(i*2654435761u >> 5);
        const char* folders[] = {"caf\xc3\xa9 \xc2\xb5m", "\xe6\x97\xa5\xe6\x9c\xac"};
        for(const char* folder : folders){
            const std::string base = dir + "/" + folder;
            const std::string stack = base + "/stack.tif", ij = base + "/imagej.tif";
            total++;
            step("utf8-path", folder, "write, find and read a file in a folder named in another script");
            bool ok = !writeTiffParallelHelper(stack.c_str(), uv.data(), 16, "w", ux, uy, uz, 0, false, "lzw", 1, 0);
            FILE* f = ok ? openUtf8(stack, "rb") : nullptr;
            const bool found = f != nullptr;
            if(f) std::fclose(f);
            TiffInfo ti;
            ok = ok && found && !getTiffInfo(stack.c_str(), ti) && ti.z == uz && ti.x == ux && ti.y == uy;
            void* all = ok ? readTiffParallelWrapperNoXYFlip(stack.c_str(), {}) : nullptr;
            void* one = ok ? readTiffParallelWrapperNoXYFlip(stack.c_str(), {1}) : nullptr;
            ok = all && one && std::memcmp(all, uv.data(), uv.size()*2) == 0 && std::memcmp(one, uv.data() + ux*uy, ux*uy*2) == 0;
            free(all); free(one);
            uint64_t* dims = ok ? getImageSize(stack.c_str()) : nullptr;
            ok = dims && dims[0] == uy && dims[1] == ux && dims[2] == uz;
            free(dims);
            // (the ImageJ reader opens the file itself)
            ok = ok && writeImageJTiff(ij, reinterpret_cast<const uint8_t*>(uv.data()), ux, uy, uz, 1, 16, false);
            void* ijBack = ok ? readTiffParallelWrapperNoXYFlip(ij.c_str(), {}) : nullptr;
            ok = ijBack && std::memcmp(ijBack, uv.data(), uv.size()*2) == 0;
            free(ijBack);
            if(ok) std::printf("PASS  %-14s %s\n", "utf8-path", folder);
            else { std::printf("FAIL  %-14s %s  (written and found where named: %d)\n", "utf8-path", folder, (int)found); failures++; }
        }
    }

    // ---- Arguments the writer and the readers take, and refuse. Compression
    // names are read in any case ("ZSTD" and "NONE" were written as LZW, like
    // any unknown name, which now fails); an image without pixels or slices
    // fails, leaving no file (Python's writer had written one slice from the
    // memory past an empty array); and slices past the end fail with an error of
    // their own, not each reader thread's "failed to open".
    {
        const uint64_t ax = 40, ay = 30, az = 4;
        std::vector<uint16_t> av(ax*ay*az);
        for(uint64_t i = 0; i < av.size(); i++) av[i] = (uint16_t)(i*40503u >> 3);
        struct CN { const char* name; uint64_t tag; };
        const CN cns[] = {{"ZSTD", COMPRESSION_ZSTD}, {"Lzw", COMPRESSION_LZW}, {"NONE", COMPRESSION_NONE}, {"", COMPRESSION_NONE}};
        for(const CN& c : cns){
            total++;
            const std::string path = dir + "/compression_name.tif";
            step("comp-name", c.name, "write with the name in another case");
            bool ok = !writeTiffParallelWrapper(ax, ay, az, path.c_str(), av.data(), 16, 0, 0, "w", false, std::string(c.name), 1, 0);
            uint64_t tag = 0;
            ok = ok && firstTag(readFileBytes(path), 259, &tag) && tag == c.tag;
            void* back = ok ? readTiffParallelWrapperNoXYFlip(path.c_str(), {}) : nullptr;
            ok = back && std::memcmp(back, av.data(), av.size()*2) == 0;
            free(back);
            std::remove(path.c_str());
            if(ok) std::printf("PASS  %-14s compression \"%s\"\n", "comp-name", c.name);
            else { std::printf("FAIL  %-14s compression \"%s\"  (tag %llu)\n", "comp-name", c.name, (unsigned long long)tag); failures++; }
        }
        struct RF { const char* name; uint64_t x, y, z, start; const char* comp; bool single; };
        const RF rfs[] = {
            {"comp-jpeg",    ax, ay, az, 0,  "jpeg", false},
            {"comp-lzw2",    ax, ay, az, 0,  "lzw2", false},
            {"comp-single",  ax, ay, az, 0,  "zip",  true},    // the one-thread writer
            {"empty-x",       0, ay, az, 0,  "lzw",  false},
            {"empty-y",      ax,  0, az, 0,  "none", false},
            {"empty-z",      ax, ay,  0, 0,  "zstd", false},
            {"empty-start",  ax, ay, az, az, "lzw",  false},
            {"empty-single", ax,  0, az, 0,  "lzw",  true},
        };
        for(const RF& c : rfs){
            total++;
            const std::string path = dir + "/" + c.name + ".tif";
            std::remove(path.c_str());
            step(c.name, "refused writes", "a write that must fail");
            const bool failed = c.single
                ? writeTiffSingle(c.x, c.y, c.z, path.c_str(), av.data(), nullptr, 16, c.start, 0, "w", false, std::string(c.comp)) != 0
                : writeTiffParallelWrapper(c.x, c.y, c.z, path.c_str(), av.data(), 16, c.start, 0, "w", false, std::string(c.comp), 1, 0) != 0;
            FILE* f = std::fopen(path.c_str(), "rb");
            const bool noFile = !f;
            if(f) std::fclose(f);
            std::remove(path.c_str());
            if(failed && noFile) std::printf("PASS  %-14s refused write\n", c.name);
            else { std::printf("FAIL  %-14s refused write  (failed=%d, file left=%d)\n", c.name, (int)failed, (int)!noFile); failures++; }
        }
        // Slice ranges: the last slice, the last two and every slice read, in
        // both layouts; one past the end, an empty or reversed range, and three
        // numbers fail.
        const std::string path = dir + "/slice_ranges.tif";
        const bool made = !writeTiffParallelWrapper(ax, ay, az, path.c_str(), av.data(), 16, 0, 0, "w", false, "lzw", 1, 0);
        const std::vector<uint8_t> avBytes(reinterpret_cast<const uint8_t*>(av.data()), reinterpret_cast<const uint8_t*>(av.data() + av.size()));
        struct RG { std::vector<uint64_t> range; bool good; };
        const RG rgs[] = {{{az - 1}, true}, {{az - 2, az}, true}, {{0, az}, true}, {{az}, false}, {{0, az + 1}, false},
                          {{az - 1, az + 1}, false}, {{2, 2}, false}, {{3, 1}, false}, {{0, 1, 2}, false}};
        for(const RG& r : rgs){
            total++;
            std::string text;
            for(size_t k = 0; k < r.range.size(); k++) text += (k ? "," : "") + std::to_string(r.range[k]);
            step("slice-range", text.c_str(), r.good ? "a range that reads" : "a range that must fail");
            void* nf = made ? readTiffParallelWrapperNoXYFlip(path.c_str(), r.range) : nullptr;
            void* fl = made ? readTiffParallelWrapper(path.c_str(), r.range) : nullptr;
            bool ok = made;
            if(r.good){
                const uint64_t first = r.range[0], count = r.range.size() == 2 ? r.range[1] - r.range[0] : 1, slice = ax*ay*2;
                const std::vector<uint8_t> wantFlip = flipped(avBytes.data() + first*slice, ax, ay, count, 1, 2);
                ok = ok && nf && fl && std::memcmp(nf, avBytes.data() + first*slice, count*slice) == 0
                        && std::memcmp(fl, wantFlip.data(), wantFlip.size()) == 0;
            }
            else ok = ok && !nf && !fl;
            free(nf); free(fl);
            if(ok) std::printf("PASS  %-14s slices {%s}\n", "slice-range", text.c_str());
            else { std::printf("FAIL  %-14s slices {%s}  (%s)\n", "slice-range", text.c_str(), r.good ? "read back wrong" : "did not fail"); failures++; }
        }
        std::remove(path.c_str());
    }

    // ---- Unreadable and damaged files: report failure, never crash.
    {
        const std::string missing = dir + "/does_not_exist.tif";
        const std::string notTiff = dir + "/not_a_tiff.tif";
        if(FILE* f = std::fopen(notTiff.c_str(), "wb")){
            const char line[] = "this is not a tiff file\n";
            for(int i = 0; i < 100; i++) std::fwrite(line, 1, sizeof line - 1, f);
            std::fclose(f);
        }
        const std::string bad[] = {missing, notTiff};
        for(const std::string& p : bad){
            total++;
            const char* nm = (&p == &bad[0]) ? "missing" : "not-a-tiff";
            step(nm, "damaged", "metadata + reads");
            uint64_t* d = getImageSize(p.c_str());
            const bool ok = !d && getDataType(p.c_str()) == 0 && getImageSizeZ(p.c_str()) == 0
                            && !readTiffParallelWrapperNoXYFlip(p.c_str(), {})
                            && !readTiffParallelWrapper(p.c_str());
            free(d);
            if(ok) std::printf("PASS  %-14s damaged\n", nm);
            else { std::printf("FAIL  %-14s damaged  (expected NULL/0 everywhere)\n", nm); failures++; }
        }
        std::remove(notTiff.c_str());

        // Damage one LZW strip of page 2: reading that page must fail cleanly in
        // both layouts and through the single-image path, and the undamaged
        // pages must still read correctly.
        total++;
        const uint64_t cx = 64, cy = 48, cz = 4;
        std::vector<uint16_t> vol(cx*cy*cz);
        std::mt19937 crng(97531u);
        for(auto& v16 : vol) v16 = (uint16_t)(crng() & 0xFFFF);
        const std::string cp = dir + "/corrupt_strip.tif";
        step("corrupt", "damaged", "write + damage page 2 strip 1");
        uint64_t off = 0, cnt = 0;
        bool ok = !writeTiffParallelWrapper(cx, cy, cz, cp.c_str(), vol.data(), 16, 0, 16, "w", false, "lzw", 1, 0)
                  && stripLocation(cp, 2, 1, &off, &cnt) && cnt > 64;
        if(ok){
            FILE* f = std::fopen(cp.c_str(), "r+b");
            std::vector<uint8_t> junk(std::min<uint64_t>(256, cnt/2), 0xFF);
            ok = f && std::fseek(f, (long)(off + cnt/4), SEEK_SET) == 0 && std::fwrite(junk.data(), 1, junk.size(), f) == junk.size();
            if(f) std::fclose(f);
        }
        step("corrupt", "damaged", "reads");
        void* r3 = ok ? readTiffParallelWrapperNoXYFlip(cp.c_str(), {}) : nullptr;
        void* rf = ok ? readTiffParallelWrapper(cp.c_str()) : nullptr;
        void* r2 = ok ? readTiffParallelWrapperNoXYFlip(cp.c_str(), {2, 3}) : nullptr;
        void* rg = ok ? readTiffParallelWrapperNoXYFlip(cp.c_str(), {0, 2}) : nullptr;
        const bool good = rg && std::memcmp(rg, vol.data(), 2*cx*cy*sizeof(uint16_t)) == 0;
        const bool pass = ok && !r3 && !rf && !r2 && good;
        free(r3); free(rf); free(r2); free(rg);
        std::remove(cp.c_str());
        if(pass) std::printf("PASS  %-14s damaged\n", "corrupt-strip");
        else { std::printf("FAIL  %-14s damaged  (setup=%d full=%d flip=%d page=%d undamaged=%d)\n", "corrupt-strip",
                           (int)ok, (int)!r3, (int)!rf, (int)!r2, (int)good); failures++; }

        // An ImageJ stack cut short, as an interrupted copy leaves it. Reading
        // past the end must fail instead of returning unread memory (or, past
        // 2 GB, retrying forever), while the complete slices still read.
        {
            total++;
            const uint64_t tx = 40, ty = 24, tz = 6, slice = tx*ty, sliceBytes = slice*2;
            std::vector<uint16_t> ij(slice*tz);
            for(uint64_t i = 0; i < ij.size(); i++) ij[i] = (uint16_t)((i*13 + 5) % 60000);
            const std::string tp = dir + "/truncated_imagej.tif";
            step("truncated-ij", "damaged", "write + cut half of the last slice");
            const bool setup = writeImageJTiff(tp, (const uint8_t*)ij.data(), tx, ty, tz, 1, 16, false)
                               && truncateFile(tp, sliceBytes/2);
            step("truncated-ij", "damaged", "reads");
            void* full  = setup ? readTiffParallelWrapperNoXYFlip(tp.c_str(), {}) : nullptr;
            void* fullF = setup ? readTiffParallelWrapper(tp.c_str()) : nullptr;
            void* tail  = setup ? readTiffParallelWrapperNoXYFlip(tp.c_str(), {4, 6}) : nullptr;
            void* head  = setup ? readTiffParallelWrapperNoXYFlip(tp.c_str(), {0, 5}) : nullptr;
            const bool complete = head && std::memcmp(head, ij.data(), 5*sliceBytes) == 0;
            const bool tpass = setup && !full && !fullF && !tail && complete;
            free(full); free(fullF); free(tail); free(head);
            std::remove(tp.c_str());
            if(tpass) std::printf("PASS  %-14s damaged\n", "truncated-ij");
            else { std::printf("FAIL  %-14s damaged  (setup=%d full=%d flip=%d tail=%d complete=%d)\n", "truncated-ij",
                               (int)setup, (int)!full, (int)!fullF, (int)!tail, (int)complete); failures++; }
        }

        // A single uncompressed image cut short, read with every thread and
        // with one (which used to take a raw-read path that ignored the error).
        {
            total++;
            const uint64_t px = 64, py = 48, imgBytes = px*py*2;
            std::vector<uint16_t> img(px*py);
            for(uint64_t i = 0; i < img.size(); i++) img[i] = (uint16_t)((i*29 + 11) % 65000);
            const std::string pp = dir + "/truncated_page.tif";
            step("truncated-2d", "damaged", "write + cut half of the image");
            const bool setup = writePlainTiff(pp, (const uint8_t*)img.data(), px, py, 1, 1, 16)
                               && truncateFile(pp, imgBytes/2);
            step("truncated-2d", "damaged", "reads, all threads and one");
            void* a  = setup ? readTiffParallelWrapperNoXYFlip(pp.c_str(), {}) : nullptr;
            void* af = setup ? readTiffParallelWrapper(pp.c_str()) : nullptr;
            const int prevThreads = omp_get_max_threads();
            omp_set_num_threads(1);
            void* b  = setup ? readTiffParallelWrapperNoXYFlip(pp.c_str(), {}) : nullptr;
            void* bf = setup ? readTiffParallelWrapper(pp.c_str()) : nullptr;
            omp_set_num_threads(prevThreads);
            const bool ppass = setup && !a && !af && !b && !bf;
            free(a); free(af); free(b); free(bf);
            std::remove(pp.c_str());
            if(ppass) std::printf("PASS  %-14s damaged\n", "truncated-2d");
            else { std::printf("FAIL  %-14s damaged  (setup=%d all=%d allFlip=%d one=%d oneFlip=%d)\n", "truncated-2d",
                               (int)setup, (int)!a, (int)!af, (int)!b, (int)!bf); failures++; }
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
