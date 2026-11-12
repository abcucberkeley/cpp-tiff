#ifndef CPPTIFF_PATHUTIL_H
#define CPPTIFF_PATHUTIL_H
#include <cstdio>
#include "tiffio.h"

// Opening files by their path, inside the library. Paths are UTF-8 on every platform. On Windows they are
// opened with Windows' wide-character functions, so that names in any script open: its narrow functions read
// a path's bytes in the ANSI code page, so that a folder named "café" or "日本" could not be opened from
// Python, Java or C++, which pass UTF-8. A path whose bytes are not valid UTF-8 is read in the ANSI code
// page, as callers that passed such paths before had them read (MATLAB's mxArrayToString, a narrow argv).
TIFF* tiffOpenPath(const char* path, const char* mode);
FILE* fopenPath(const char* path, const char* mode);
bool folderExistsPath(const char* path);
int mkdirPath(const char* path);

#endif
