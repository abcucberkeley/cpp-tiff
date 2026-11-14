// Windows: loads fakemex.dll (and through it the library and the GCC runtime DLLs it loads) on a thread of its own,
// runs it, unloads it and lets the thread exit, as MATLAB does with a MEX file at "clear mex" and at exit.
//   unloadtest <file.tif> <write 0|1> <loads> <unload 0|1> [check]
// Exit code 0: the thread exited cleanly; 3: an access violation, reported with the module it fell in (one that was
// unloaded, if the address is in its old range); 5 (with check): after an unload, the library was still loaded
// (clear mex unloads it) or libgomp was not (the library keeps it loaded, src/winruntime.cpp); other: a failed step.
#include <windows.h>
#include <process.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct Module { const wchar_t* name; uintptr_t base, end; };
Module modules[] = {{L"fakemex.dll", 0, 0}, {L"libcppTiff.dll", 0, 0}, {L"libgomp-1.dll", 0, 0},
                    {L"libgcc_s_seh-1.dll", 0, 0}, {L"libwinpthread-1.dll", 0, 0}};

void say(const char* fmt, ...){
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    DWORD w;
    if(n > 0) WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buf, (DWORD)n, &w, NULL);
}

// The address ranges of the modules while they are loaded, to name one after it is unloaded
void record(){
    for(Module& m : modules){
        HMODULE h = GetModuleHandleW(m.name);
        if(!h) continue;
        const IMAGE_DOS_HEADER* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(h);
        const IMAGE_NT_HEADERS* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(reinterpret_cast<const char*>(h) + dos->e_lfanew);
        m.base = reinterpret_cast<uintptr_t>(h);
        m.end = m.base + nt->OptionalHeader.SizeOfImage;
    }
}

void states(const char* when){
    say("%s:", when);
    for(const Module& m : modules) say(" %ls %s;", m.name, GetModuleHandleW(m.name) ? "loaded" : "unloaded");
    say("\n");
}

LONG CALLBACK onException(EXCEPTION_POINTERS* e){
    if(e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    const uintptr_t at = reinterpret_cast<uintptr_t>(e->ExceptionRecord->ExceptionAddress);
    const Module* in = NULL;
    for(const Module& m : modules) if(at >= m.base && at < m.end) in = &m;
    if(in) say("ACCESS VIOLATION in thread %lu at %ls+0x%llx (%s)\n", GetCurrentThreadId(), in->name,
               (unsigned long long)(at - in->base), GetModuleHandleW(in->name) ? "loaded" : "unloaded");
    else say("ACCESS VIOLATION in thread %lu at %p, in none of the modules\n", GetCurrentThreadId(), (void*)at);
    TerminateProcess(GetCurrentProcess(), 3);
    return EXCEPTION_CONTINUE_SEARCH;
}

struct Args { const char* file; int write, loads, unload; bool check; };

unsigned __stdcall worker(void* p){
    const Args* a = static_cast<const Args*>(p);
    for(int i = 0; i < a->loads; i++){
        HMODULE h = LoadLibraryW(L"fakemex.dll");
        if(!h){ say("LoadLibrary(fakemex.dll) failed: %lu\n", GetLastError()); return 2; }
        record();
        typedef int (*Run)(const char*, int);
        Run run = reinterpret_cast<Run>(reinterpret_cast<void*>(GetProcAddress(h, "fakemexRun")));
        if(!run){ say("no fakemexRun\n"); return 2; }
        const int r = run(a->file, a->write);
        say("load %d: fakemexRun gave %d\n", i + 1, r);
        if(r) return 4;
        if(a->unload){
            FreeLibrary(h);
            states("after FreeLibrary");
            if(a->check && (GetModuleHandleW(L"libcppTiff.dll") || !GetModuleHandleW(L"libgomp-1.dll"))) return 5;
        }
    }
    say("thread %lu exits\n", GetCurrentThreadId());
    return 0;
}

}

int main(int argc, char** argv){
    if(argc != 5 && !(argc == 6 && !strcmp(argv[5], "check"))){
        say("usage: unloadtest <file.tif> <write 0|1> <loads> <unload 0|1> [check]\n");
        return 1;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    AddVectoredExceptionHandler(1, onException);
    Args a = {argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc == 6};
    HANDLE t = reinterpret_cast<HANDLE>(_beginthreadex(NULL, 0, worker, &a, 0, NULL));
    if(!t){ say("no thread\n"); return 1; }
    WaitForSingleObject(t, INFINITE);
    DWORD code = 99;
    GetExitCodeThread(t, &code);
    CloseHandle(t);
    states("after the thread exited");
    say("thread exit code %lu\n", code);
    return (int)code;
}
