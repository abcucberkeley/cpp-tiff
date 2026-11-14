// Windows only: once loaded, the library stays loaded until the process ends, and with it the GCC runtime DLLs it
// loads (libgomp-1.dll, libgcc_s_seh-1.dll and libwinpthread-1.dll); clear mex still unloads the mex files.
// With a GCC that emulates thread-local storage, as the release's MinGW-w64 GCC 12.2 does (MSYS2's GCC 16 does not),
// every thread that runs OpenMP, or the library's std::call_once (std::future::get) or exceptions, holds thread-local
// values whose destructors are libgcc's (libgcc_s_seh-1.dll's for libgomp's, the library's own static copy for its
// own), and libwinpthread-1.dll runs them when that thread exits. libwinpthread-1.dll stays loaded as long as
// libgomp's threads live, as they were started through it, but the library, libgomp-1.dll and libgcc_s_seh-1.dll were
// unloaded by clear mex or as MATLAB exited, and a thread of MATLAB's that had called a mex file then crashed as it
// exited (at libgcc_s_seh-1.dll+0x12e50, its emutls_destroy, with 1.7.3's DLLs). Kept loaded, nothing the library
// leaves behind (those destructors, libgomp's idle threads) outlives its code.
#if defined(_WIN32)
#include <windows.h>

namespace {

struct PinLibrary {
    PinLibrary(){
        HMODULE self;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                           reinterpret_cast<LPCWSTR>(this), &self);
    }
} pinLibrary;

}
#endif
