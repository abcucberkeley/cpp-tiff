#ifndef CPPTIFF_TIFFDIRS_H
#define CPPTIFF_TIFFDIRS_H
#include <cstdint>
#include "tiffio.h"

// libtiff's map from directory numbers to where the directories start: functions internal to the libtiff this
// library builds and links in (4.7.1), not part of libtiff's API. A handle's map holds every directory the handle
// has walked through, and TIFFSetDirectory reads a directory the map has straight away, where without it the
// handle walks every directory between the one it is on (or the first) and that one, a read per directory.
extern "C" int _TIFFGetOffsetFromDirNumber(TIFF* tif, tdir_t dirn, uint64_t* diroff);
extern "C" int _TIFFCheckDirNumberAndOffset(TIFF* tif, tdir_t dirn, uint64_t diroff);

#endif
