// Windows only. EXPERIMENT (branch win-runtime-unload): which DLLs the library keeps loaded.
// CPPTIFF_WIN_PIN 0: none (as released), 1: this library itself (and so everything it loads),
// 2: the GCC runtime DLLs it loads (libgomp, libgcc_s), found through the import tables.
#if defined(_WIN32)
#include <windows.h>
#include <cstring>

#ifndef CPPTIFF_WIN_PIN
#define CPPTIFF_WIN_PIN 0
#endif

namespace {

#if CPPTIFF_WIN_PIN == 2
bool isRuntime(const char* name){
    return !_strnicmp(name, "libgomp", 7) || !_strnicmp(name, "libgcc_s", 8);
}

// Pins the runtime DLLs that module imports, and the runtime DLLs those import
void pinImports(HMODULE module, int depth){
    const unsigned char* base = reinterpret_cast<const unsigned char*>(module);
    const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const IMAGE_NT_HEADERS* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if(!dir.VirtualAddress) return;
    for(const IMAGE_IMPORT_DESCRIPTOR* d = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); d->Name; d++){
        const char* name = reinterpret_cast<const char*>(base + d->Name);
        if(!isRuntime(name)) continue;
        HMODULE dep;
        if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, name, &dep) && depth < 2) pinImports(dep, depth + 1);
    }
}
#endif

struct Pin {
    Pin(){
#if CPPTIFF_WIN_PIN == 1
        HMODULE self;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                           reinterpret_cast<LPCWSTR>(this), &self);
#elif CPPTIFF_WIN_PIN == 2
        HMODULE self;
        if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCWSTR>(this), &self))
            pinImports(self, 0);
#endif
    }
} pin;

}
#endif
