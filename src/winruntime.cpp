// Windows only: once the library is loaded, the GCC runtime DLLs it loads (libgomp-1.dll, its OpenMP runtime, and
// libgcc_s_seh-1.dll) stay loaded until the process ends, while the library itself can still be unloaded.
// With a GCC that emulates thread-local storage, as the release's MinGW-w64 GCC 12.2 does (MSYS2's GCC 16 does not),
// every thread that runs OpenMP, or the library's std::call_once (std::future::get) or exceptions, holds thread-local
// values whose destructor is libgcc's, and libwinpthread-1.dll runs it when that thread exits. libwinpthread-1.dll
// stays loaded as long as libgomp's threads live, as they were started through it, but libgomp-1.dll and
// libgcc_s_seh-1.dll went with the library, by clear mex or as MATLAB exits, and a thread of MATLAB's that had called
// a mex file then crashed as it exited (at libgcc_s_seh-1.dll+0x12e50, its emutls_destroy, with 1.7.3's DLLs). The
// library links the shared libgcc on Windows (CMakeLists.txt), so it leaves no destructor of its own behind. The DLLs
// are found through the import tables, the library's and theirs, so that the names a Python wheel gives them are
// found too.
#if defined(_WIN32)
#include <windows.h>
#include <cstring>

namespace {

bool isRuntime(const char* name){
    return !_strnicmp(name, "libgomp", 7) || !_strnicmp(name, "libgcc_s", 8);
}

void pinImports(HMODULE module, int depth){
    const unsigned char* base = reinterpret_cast<const unsigned char*>(module);
    const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const IMAGE_NT_HEADERS* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if(!dir.VirtualAddress) return;
    for(const IMAGE_IMPORT_DESCRIPTOR* d = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); d->Name; d++){
        const char* name = reinterpret_cast<const char*>(base + d->Name);
        HMODULE dep;
        if(isRuntime(name) && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, name, &dep) && depth < 2)
            pinImports(dep, depth + 1);
    }
}

struct PinRuntime {
    PinRuntime(){
        HMODULE self;
        if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCWSTR>(this), &self))
            pinImports(self, 0);
    }
} pinRuntime;

}
#endif
