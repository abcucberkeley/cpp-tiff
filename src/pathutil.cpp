#include "pathutil.h"
#include <sys/stat.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <direct.h>
#include <string>

// The path in UTF-16: read as UTF-8, or in the ANSI code page when its bytes are not valid UTF-8 (empty
// when it cannot be read either way, or is empty)
static std::wstring widePath(const char* path){
    UINT codePage = CP_UTF8;
    DWORD flags = MB_ERR_INVALID_CHARS;
    int n = MultiByteToWideChar(codePage, flags, path, -1, NULL, 0);
    if(n <= 0){
        codePage = CP_ACP;
        flags = 0;
        n = MultiByteToWideChar(codePage, flags, path, -1, NULL, 0);
    }
    if(n <= 1) return std::wstring();
    std::wstring wide((size_t)n, L'\0');
    if(MultiByteToWideChar(codePage, flags, path, -1, &wide[0], n) <= 0) return std::wstring();
    wide.resize((size_t)n - 1);
    return wide;
}

TIFF* tiffOpenPath(const char* path, const char* mode){
    const std::wstring wide = widePath(path);
    return wide.empty() ? TIFFOpen(path, mode) : TIFFOpenW(wide.c_str(), mode);
}

FILE* fopenPath(const char* path, const char* mode){
    const std::wstring wide = widePath(path), wideMode = widePath(mode);
    return wide.empty() ? fopen(path, mode) : _wfopen(wide.c_str(), wideMode.c_str());
}

bool folderExistsPath(const char* path){
    const std::wstring wide = widePath(path);
    struct _stat64 info;
    return !wide.empty() && _wstat64(wide.c_str(), &info) == 0 && (info.st_mode & _S_IFDIR);
}

int mkdirPath(const char* path){
    const std::wstring wide = widePath(path);
    return wide.empty() ? -1 : _wmkdir(wide.c_str());
}
#else
TIFF* tiffOpenPath(const char* path, const char* mode){
    return TIFFOpen(path, mode);
}

FILE* fopenPath(const char* path, const char* mode){
    return fopen(path, mode);
}

bool folderExistsPath(const char* path){
    struct stat info;
    return stat(path, &info) == 0 && (info.st_mode & S_IFDIR);
}

int mkdirPath(const char* path){
    return mkdir(path, 0777);
}
#endif
