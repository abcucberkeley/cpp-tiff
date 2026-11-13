#ifndef CPPTIFF_STRIPDECODE_H
#define CPPTIFF_STRIPDECODE_H
#include <cstdint>
#include <vector>
#include "tiffio.h"

// A strip's or a tile's decoded bytes, as TIFFReadEncodedStrip and TIFFReadEncodedTile give them (at most
// size of them; their count, or -1), LZW decoded by the library's own decoder where it can
// (parallelreadtiff.cpp). raw is the caller's buffer for the stored bytes, reused from one to the next.
// Internal to the library (not installed): the GPU reader decodes on the host with them what the GPU
// does not decode.
tmsize_t decodeStrip(TIFF* tif, uint32_t strip, void* buf, tmsize_t size, std::vector<uint8_t>& raw);
tmsize_t decodeTile(TIFF* tif, uint32_t tile, void* buf, tmsize_t size, std::vector<uint8_t>& raw);

#endif
