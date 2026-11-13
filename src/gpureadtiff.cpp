#include "gpureadtiff.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <omp.h>
#include "tiffio.h"
#include "helperfunctions.h"
#include "parallelreadtiff.h"
#include "pathutil.h"
#include "tiffdirs.h"
#include "stripdecode.h"
#include "gpu/cudaapi.h"
#include "gpu/gpucommon.h"

using namespace ctcuda;

namespace {

// A strip or tile as the kernels take it (kernels.cu's Unit: 64 bytes)
struct Unit {
    uint64_t src, out;
    uint32_t slice, row0, col0, rows, cols, pitch, x, y;
    uint16_t samples, plane;
    uint8_t es, swap, predictor, pad[9];
};
static_assert(sizeof(Unit) == 64, "a unit is 64 bytes, as the kernels have it");
const uint64_t hostBit = 1ull << 63;

// How a unit's stored bytes become its decoded ones: as they are, by nvCOMP, by the GPU's LZW decoder (kernels.cu's
// ct_lzw), or on the host
enum How : uint8_t { Raw = 0, GpuZstd = 1, GpuDeflate = 2, Host = 3, GpuLzw = 4 };
// (kinds of GPU-decoded streams: nvCOMP's codecs, then the LZW kernel's)
const int LzwKind = NCodecs, NKinds = NCodecs + 1;

// Whether the GPU decodes LZW (kernels.cu's ct_lzw; CPPTIFF_GPU_LZW=0: the host does)
bool gpuLzw(){
    static const bool on = [](){
        const char* e = std::getenv("CPPTIFF_GPU_LZW");
        return !(e && *e && std::atoi(e) == 0);
    }();
    return on;
}

// A unit as planned: where its stored bytes are in the file, how they are decoded, and where its bytes go
// in its part's staging (stored ones, or for a unit decoded on the host its decoded ones) and nvCOMP's output
struct Planned {
    Unit u;
    uint64_t at = 0, stored = 0, decoded = 0, dir = 0;
    uint32_t read = 0, strile = 0;
    How how = Raw;
    bool tiled = false;
    uint64_t stage = 0, decOff = 0;
};

struct ReadState {
    const GpuTiffRead* r = nullptr;
    TiffInfo info;
    uint64_t start = 0, z = 0, es = 0, bytes = 0;
    bool cpu = false;          // read on the CPU and copied
};

// Each thread's handle of each file, opened when first needed
struct Handles {
    std::vector<std::vector<TIFF*>> h;
    std::vector<std::vector<uint8_t>> raw;
    const std::vector<ReadState>& reads;
    Handles(int threads, const std::vector<ReadState>& r) : h(threads, std::vector<TIFF*>(r.size(), nullptr)), raw(threads), reads(r){}
    ~Handles(){ for(auto &t : h) for(TIFF* x : t) if(x) TIFFClose(x); }
    TIFF* get(int t, uint32_t read){
        TIFF*& x = h[t][read];
        if(!x) x = tiffOpenPath(reads[read].r->fileName.c_str(), "rm");
        return x;
    }
};

// Directory dir current on tif, read straight from where the walk found it
bool toDirectory(TIFF* tif, uint64_t dir, const TiffInfo &info){
    if(TIFFCurrentDirectory(tif) == (tdir_t)dir) return true;
    if(dir < info.offsets.size()) _TIFFCheckDirNumberAndOffset(tif, (tdir_t)dir, info.offsets[dir]);
    return TIFFSetDirectory(tif, (tdir_t)dir) != 0;
}

Unit unitFor(void* out, uint64_t slice, uint64_t row0, uint64_t col0, uint64_t rows, uint64_t cols, uint64_t pitch,
             uint64_t x, uint64_t y, uint64_t samples, uint16_t plane, uint64_t es, bool swap, uint16_t predictor){
    Unit u;
    std::memset(&u, 0, sizeof(u));
    u.out = (uint64_t)(uintptr_t)out;
    u.slice = (uint32_t)slice; u.row0 = (uint32_t)row0; u.col0 = (uint32_t)col0; u.rows = (uint32_t)rows; u.cols = (uint32_t)cols;
    u.pitch = (uint32_t)pitch; u.x = (uint32_t)x; u.y = (uint32_t)y; u.samples = (uint16_t)samples; u.plane = plane;
    u.es = (uint8_t)es; u.swap = swap ? 1 : 0; u.predictor = (uint8_t)predictor;
    return u;
}

// The units of read ri: its slices' strips or tiles (or, for an ImageJ stack, blocks of its slices' rows) and how
// each is decoded; false for a file the GPU path does not take (it is then read on the CPU)
bool planRead(ReadState &rs, uint32_t ri, const Api* a, Handles &hs, int threads, std::vector<Planned> &units){
    const TiffInfo &info = rs.info;
    const uint64_t x = info.x, y = info.y, spp = info.samples, es = rs.es, bits = info.bits;
    const uint64_t rowElems = x*spp, sliceBytes = y*rowElems*es;
    void* const out = rs.r->out;
    if(info.imageJ){
        // One directory, and the slices back to back from its strip's offset (see readTiffParallelImageJ): in blocks
        // of whole rows of at most 64 MB, so that a big slice spreads over parts
        TIFF* tif = hs.get(0, ri);
        uint64_t* offsets = nullptr;
        if(!tif || !TIFFGetField(tif, TIFFTAG_STRIPOFFSETS, &offsets) || !offsets) return false;
        const bool swap = es > 1 && TIFFIsByteSwapped(tif);
        const uint64_t data = offsets[0];
        const uint64_t blockRows = std::max<uint64_t>(1, std::min<uint64_t>(y, ((uint64_t)64 << 20)/std::max<uint64_t>(1, rowElems*es)));
        for(uint64_t k = 0; k < rs.z; k++)
            for(uint64_t r0 = 0; r0 < y; r0 += blockRows){
                Planned p;
                const uint64_t rows = std::min(blockRows, y - r0);
                p.u = unitFor(out, k, r0, 0, rows, x, rowElems, x, y, spp, 0xFFFF, es, swap, 1);
                p.at = data + (rs.start + k)*sliceBytes + r0*rowElems*es;
                p.stored = p.decoded = rows*rowElems*es;
                p.read = ri;
                p.how = Raw;
                units.push_back(p);
            }
        return true;
    }
    // Each directory's strips or tiles, the directories shared out between the threads
    std::vector<std::vector<Planned>> per(rs.z);
    std::atomic<int> bad{0};
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for(int64_t k = 0; k < (int64_t)rs.z; k++){
        if(bad) continue;
        const int t = omp_get_thread_num();
        TIFF* tif = hs.get(t, ri);
        const uint64_t dir = rs.start + (uint64_t)k;
        if(!tif || !toDirectory(tif, dir, info)){ bad = 1; continue; }
        uint32_t w = 0, h = 0;
        uint16_t b = 1, s = 1, comp = COMPRESSION_NONE, pred = PREDICTOR_NONE, fill = FILLORDER_MSB2LSB, pc = PLANARCONFIG_CONTIG;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &b);
        TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &s);
        TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &comp);
        // (only codecs with a predictor have the tag: asked of others, libtiff reports an error)
        if(comp != COMPRESSION_NONE) TIFFGetFieldDefaulted(tif, TIFFTAG_PREDICTOR, &pred);
        TIFFGetFieldDefaulted(tif, TIFFTAG_FILLORDER, &fill);
        TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &pc);
        // (a directory of another size or type: the CPU reader's to report)
        if(w != x || h != y || b != bits || s != spp){ bad = 1; continue; }
        const bool planar = spp > 1 && pc == PLANARCONFIG_SEPARATE, tiled = TIFFIsTiled(tif);
        const bool byteSwapped = es > 1 && TIFFIsByteSwapped(tif);
        How gpuHow = Host;
        const bool kernelsTake = fill == FILLORDER_MSB2LSB && (pred == PREDICTOR_NONE || pred == PREDICTOR_HORIZONTAL);
        if(kernelsTake){
            if(comp == COMPRESSION_NONE) gpuHow = Raw;
            else if(comp == COMPRESSION_ZSTD && a->decompressAsync[Zstd]) gpuHow = GpuZstd;
            else if((comp == COMPRESSION_ADOBE_DEFLATE || comp == COMPRESSION_DEFLATE) && a->decompressAsync[Deflate]) gpuHow = GpuDeflate;
        }
        const uint64_t planes = planar ? spp : 1, unitSpp = planar ? 1 : spp;
        uint32_t tw = 0, th = 0, rps32 = 0;
        uint64_t across = 1, perPlane = 1, rps = 1;
        if(tiled){
            TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
            TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);
            if(!tw || !th){ bad = 1; continue; }
            across = (x + tw - 1)/tw;
            perPlane = across*((y + th - 1)/th);
        }
        else{
            TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps32);
            rps = std::min<uint64_t>(std::max<uint64_t>(rps32, 1), y);
            perPlane = (y + rps - 1)/rps;
        }
        const uint64_t n = perPlane*planes;
        if(n != (uint64_t)(tiled ? TIFFNumberOfTiles(tif) : TIFFNumberOfStrips(tif))){ bad = 1; continue; }
        if(kernelsTake && comp == COMPRESSION_LZW && gpuLzw()) gpuHow = GpuLzw;
        std::vector<Planned> &list = per[(size_t)k];
        list.reserve((size_t)n);
        for(uint64_t i = 0; i < n; i++){
            Planned p;
            const uint64_t plane = i/perPlane, j = i % perPlane;
            uint64_t row0, col0, rows, cols, pitch, decoded;
            if(tiled){
                row0 = (j/across)*th;
                col0 = (j % across)*tw;
                rows = std::min<uint64_t>(th, y - row0);
                cols = std::min<uint64_t>(tw, x - col0);
                pitch = (uint64_t)tw*unitSpp;
                decoded = (uint64_t)th*pitch*es;
            }
            else{
                row0 = j*rps;
                col0 = 0;
                rows = std::min<uint64_t>(rps, y - row0);
                cols = x;
                pitch = x*unitSpp;
                decoded = rows*pitch*es;
            }
            p.at = TIFFGetStrileOffset(tif, (uint32_t)i);
            p.stored = TIFFGetStrileByteCount(tif, (uint32_t)i);
            How how = gpuHow;
            // (a short or empty uncompressed strip, or an empty compressed one: libtiff's to decode)
            if(!p.stored || (how == Raw && p.stored < decoded) || (how == GpuDeflate && p.stored < 7) || (how == GpuLzw && p.stored < 2)) how = Host;
            if(how == Raw) p.stored = decoded;
            const bool gpu = how != Host;
            p.u = unitFor(out, (uint64_t)k, row0, col0, rows, cols, pitch, x, y, spp, planar ? (uint16_t)plane : (uint16_t)0xFFFF,
                          es, gpu && byteSwapped, gpu ? pred : PREDICTOR_NONE);
            p.how = how;
            p.decoded = decoded;
            p.read = ri;
            p.dir = dir;
            p.strile = (uint32_t)i;
            p.tiled = tiled;
            list.push_back(p);
        }
    }
    if(bad) return false;
    for(auto &l : per) units.insert(units.end(), l.begin(), l.end());
    return true;
}

// A part of the units: those that go to the GPU together, and where their tables are in its staging
struct Part {
    size_t begin = 0, end = 0;
    uint64_t staging = 0, decoded = 0;
    uint64_t unitsAt = 0, listAt = 0, arraysAt = 0;   // (offsets in the staging: the unit table, the predictor list, nvCOMP's arrays)
    uint32_t nGpu[NKinds] = {0, 0, 0}, nList = 0;
    uint64_t maxChunk[NKinds] = {0, 0, 0}, total[NKinds] = {0, 0, 0};
    uint32_t esCount[4] = {0, 0, 0, 0};               // (units of 1-, 2-, 4- and 8-byte elements, in that order in the table)
    uint32_t maxRows[4] = {0, 0, 0, 0}, maxTiles[4] = {0, 0, 0, 0};
};

int esIndex(uint64_t es){ return es == 1 ? 0 : es == 2 ? 1 : es == 4 ? 2 : 3; }

// Lays out part p's staging: the GPU codecs' stored bytes (zstd's, then deflate's, aligned as nvCOMP asks; a
// deflate stream's data starts two bytes in, after its zlib header), the other units' bytes, then the tables;
// and nvCOMP's outputs in the decode buffer
void layOut(std::vector<Planned> &units, Part &p, const Ctx &c){
    uint64_t off = 0, dec = 0;
    for(int pass = 0; pass < 4; pass++)
        for(size_t i = p.begin; i < p.end; i++){
            Planned &q = units[i];
            const int codec = q.how == GpuZstd ? Zstd : q.how == GpuDeflate ? Deflate : q.how == GpuLzw ? LzwKind : -1;
            if(pass == 0 && codec == Zstd){
                off = alignUp(off, std::max<uint64_t>(c.align[Zstd].input, 8));
                q.stage = off;
                off += q.stored;
            }
            else if(pass == 1 && codec == Deflate){
                off = alignUp(off + 2, std::max<uint64_t>(c.align[Deflate].input, 8)) - 2;
                q.stage = off;
                off += q.stored;
            }
            else if(pass == 2 && codec == LzwKind){
                off = alignUp(off, 8);
                q.stage = off;
                off += q.stored;
            }
            else if(pass == 3 && codec < 0){
                off = alignUp(off, 16);
                q.stage = off;
                off += q.decoded;
            }
            else continue;
            if(codec >= 0){
                dec = alignUp(dec, codec == LzwKind ? 16 : std::max<uint64_t>(c.align[codec].output, 16));
                q.decOff = dec;
                dec += q.decoded;
                p.nGpu[codec]++;
                p.maxChunk[codec] = std::max(p.maxChunk[codec], q.decoded);
                p.total[codec] += q.decoded;
            }
        }
    uint32_t nAll = 0;
    for(size_t i = p.begin; i < p.end; i++){
        const Planned &q = units[i];
        p.esCount[esIndex(q.u.es)]++;
        if(q.u.predictor == PREDICTOR_HORIZONTAL) p.nList++;
        const int e = esIndex(q.u.es);
        p.maxRows[e] = std::max(p.maxRows[e], q.u.rows);
        const uint32_t spp = q.u.plane == 0xFFFF ? q.u.samples : 1;
        const uint64_t tiles = (uint64_t)((q.u.cols + 31)/32)*((q.u.rows + 31)/32)*spp;
        p.maxTiles[e] = (uint32_t)std::min<uint64_t>(std::max<uint64_t>(p.maxTiles[e], tiles), 0xFFFFFFFFu);
        nAll++;
    }
    off = alignUp(off, 64);
    p.unitsAt = off;
    off += (uint64_t)nAll*sizeof(Unit);
    p.listAt = off;
    off += alignUp((uint64_t)p.nList*4, 64);
    p.arraysAt = off;
    // nvCOMP's arrays, zstd's streams then deflate's, then the LZW kernel's: input pointers, input sizes, output
    // sizes, output pointers
    off += 4*8*(uint64_t)(p.nGpu[Zstd] + p.nGpu[Deflate] + p.nGpu[LzwKind]);
    p.staging = off;
    p.decoded = dec;
}

double secondsSince(std::chrono::steady_clock::time_point t0){
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

bool gpuReadTiffAvailable(std::string &why){
    const Api* a = api(&why);
    if(!a) return false;
    CUdevice dev = 0;
    if(a->cuDeviceGet(&dev, 0)){
        why = "no NVIDIA GPU was found";
        return false;
    }
    return true;
}

void gpuReadTiffSetNvcompLibrary(const std::string &path){
    setNvcompLibrary(path);
}

void gpuReleaseBuffers(){
    // (no GPU read or write has loaded the driver: nothing was kept)
    if(const Api* a = apiIfLoaded()) releaseBuffers(a);
}

uint8_t gpuReadTiff(const std::vector<GpuTiffRead> &reads, const bool flipXY, void* stream, const int threadsGiven,
                    std::string &errString){
    TIFFSetWarningHandler(DummyHandler);
    errString.clear();
    if(reads.empty()) return 0;
    std::string why;
    const Api* a = api(&why);
    if(!a){
        errString = "GPU reads cannot run here: "+why;
        return 1;
    }
    const int threads = threadsGiven > 0 ? threadsGiven : std::max(1, omp_get_max_threads());
    CUstream s = (CUstream)stream;
    // CPPTIFF_GPU_PROFILE: where the read's time went, to stderr
    const bool profile = std::getenv("CPPTIFF_GPU_PROFILE") && *std::getenv("CPPTIFF_GPU_PROFILE");
    const auto tStart = std::chrono::steady_clock::now();
    double tInfo = 0, tPlan = 0, tFill = 0, tWait = 0, tCpu = 0;
    uint64_t bRaw = 0, bGpu = 0, bGpuOut = 0, bHost = 0, bLzw = 0, bLzwOut = 0, nCpu = 0;
    size_t nParts = 0;
    try{
        // Every read's output in one context, the one the GPU work runs in
        for(const auto &r : reads)
            if(!r.out || !isGpuMemory(a, r.out)) throw Fail("\""+r.fileName+"\": its output is not GPU memory");
        const CUcontext ctx = memoryContext(a, reads[0].out);
        for(const auto &r : reads)
            if(memoryContext(a, r.out) != ctx) throw Fail("the reads' outputs are on more than one GPU (CUDA context)");
        CtxScope scope(a, ctx);
        Ctx &c = context(a, ctx);

        // What each file is, and its slices
        auto t0 = std::chrono::steady_clock::now();
        std::vector<ReadState> rs(reads.size());
        for(size_t i = 0; i < reads.size(); i++){
            ReadState &q = rs[i];
            q.r = &reads[i];
            if(getTiffInfo(reads[i].fileName.c_str(), q.info)) throw Fail("File \""+reads[i].fileName+"\" cannot be opened");
            const TiffInfo &info = q.info;
            if(info.bits != 8 && info.bits != 16 && info.bits != 32 && info.bits != 64)
                throw Fail("\""+reads[i].fileName+"\": "+std::to_string(info.bits)+"-bit samples cannot be read");
            uint64_t start = 0, z = info.z;
            const std::vector<uint64_t> &zr = reads[i].zRange;
            if(zr.size()){
                const uint64_t end = zr.size() == 2 ? zr[1] : zr[0] + 1;
                if(zr.size() > 2 || zr[0] >= end || end > info.z)
                    throw Fail("\""+reads[i].fileName+"\": slices ["+std::to_string(zr[0])+", "+std::to_string(end)+
                               ") are not in the file, which has "+std::to_string(info.z));
                start = zr[0];
                z = end - start;
            }
            q.start = start;
            q.z = z;
            q.es = info.bits/8;
            q.bytes = info.x*info.y*info.samples*q.es*z;
        }

        tInfo = secondsSince(t0);
        // Every read's units, and the reads the GPU path does not take
        t0 = std::chrono::steady_clock::now();
        Handles hs(threads, rs);
        std::vector<Planned> units;
        for(size_t i = 0; i < rs.size(); i++){
            const size_t before = units.size();
            if(!planRead(rs[i], (uint32_t)i, a, hs, threads, units)){
                units.resize(before);
                rs[i].cpu = true;
            }
        }

        tPlan = secondsSince(t0);
        for(const Planned &q : units){
            if(q.how == Raw) bRaw += q.stored;
            else if(q.how == Host) bHost += q.decoded;
            else if(q.how == GpuLzw){ bLzw += q.stored; bLzwOut += q.decoded; }
            else{ bGpu += q.stored; bGpuOut += q.decoded; }
        }
        // Parts of at most partMax bytes of staging and of decoded bytes (a unit bigger than that is a part of its
        // own), two of them at a time: the host fills one while the GPU works on the other. nvCOMP decompresses a
        // batch's streams side by side, so a part holds about 4 streams for each of the GPU's multiprocessors: with
        // strips of 2 MB, parts of 256 MB gave a B200 127 at a time, and zstd read 2.3 times as fast in parts of
        // 1 GB; parts of 256 MB at least (smaller ones overlap the host's reading and the GPU's work more: deflate
        // strips of 256 KB read 1.2 times as fast in them as in parts of 1 GB), at most 1 GB, and no more than an
        // eighth of the GPU's free memory. CPPTIFF_GPU_PART_MAX_MB sets it instead.
        uint64_t codecBytes = 0, codecUnits = 0;
        for(const Planned &q : units)
            if(q.how == GpuZstd || q.how == GpuDeflate || q.how == GpuLzw){ codecBytes += q.decoded; codecUnits++; }
        uint64_t partMax = (uint64_t)256 << 20;
        if(codecUnits) partMax = std::max<uint64_t>(partMax, 4*(uint64_t)std::max(1, c.sms)*(codecBytes/codecUnits));
        partMax = std::min<uint64_t>(partMax, (uint64_t)1 << 30);
        size_t freeMem = 0, totalMem = 0;
        if(!a->cuMemGetInfo(&freeMem, &totalMem) && freeMem) partMax = std::min<uint64_t>(partMax, std::max<uint64_t>(freeMem/8, (uint64_t)64 << 20));
        if(std::getenv("CPPTIFF_GPU_PART_MAX_MB") && *std::getenv("CPPTIFF_GPU_PART_MAX_MB"))
            partMax = std::max<uint64_t>(1, envNumber("CPPTIFF_GPU_PART_MAX_MB", 256)) << 20;
        std::vector<Part> parts;
        {
            Part p;
            uint64_t st = 0, de = 0;
            for(size_t i = 0; i < units.size(); i++){
                const Planned &q = units[i];
                const bool onGpu = q.how == GpuZstd || q.how == GpuDeflate || q.how == GpuLzw;
                const uint64_t sb = (onGpu ? q.stored : q.decoded) + 64;
                const uint64_t db = onGpu ? q.decoded + 64 : 0;
                if(i > p.begin && (st + sb > partMax || de + db > partMax)){
                    p.end = i;
                    parts.push_back(p);
                    p = Part();
                    p.begin = i;
                    st = de = 0;
                }
                st += sb;
                de += db;
            }
            p.end = units.size();
            if(p.end > p.begin) parts.push_back(p);
        }
        nParts = parts.size();
        uint64_t totalBytes = 0;
        for(const auto &q : rs) totalBytes += q.bytes;
        const double timeoutDefault = 30.0 + (double)totalBytes/50e6;
        const double timeout = std::getenv("CPPTIFF_GPU_TIMEOUT") ? (double)envNumber("CPPTIFF_GPU_TIMEOUT", 0) : timeoutDefault;

        Slot* slots[2] = {&takeSlot(c), &takeSlot(c)};
        // (given back as the call ends; one whose GPU work has not finished, as when the call fails, is never reused)
        struct Give {
            const Api* a; Ctx &c; Slot** s;
            ~Give(){ for(int k = 0; k < 2; k++){ if(s[k]->done && a->cuEventQuery(s[k]->done) != 0) s[k]->lost = true; giveSlot(c, *s[k]); } }
        } give{a, c, slots};
        // Each slot's parts on a stream of its own, so that one part's copy to the GPU overlaps the other's
        // decompression (on the caller's stream alone, each copy waited for the part before), both after the
        // caller's work on its stream
        CUevent callerDone = nullptr;
        ck(a, a->cuEventCreate(&callerDone, 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
        struct EventGone { const Api* a; CUevent e; ~EventGone(){ a->cuEventDestroy(e); } } callerGone{a, callerDone};
        ck(a, a->cuEventRecord(callerDone, s), "cuEventRecord");
        for(Slot* sl : slots){
            if(!sl->stream) ck(a, a->cuStreamCreate(&sl->stream, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate");
            ck(a, a->cuStreamWaitEvent(sl->stream, callerDone, 0), "cuStreamWaitEvent");
            if(!sl->done) ck(a, a->cuEventCreate(&sl->done, 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
            if(!sl->small){
                void* q = nullptr;
                ck(a, a->cuMemHostAlloc(&q, 64, 0), "cuMemHostAlloc");
                sl->small = static_cast<uint8_t*>(q);
            }
        }
        // (the part each slot last took, and the reads in it, for its check)
        long long inSlot[2] = {-1, -1};
        auto checkSlot = [&](int k){
            if(inSlot[k] < 0) return;
            Slot &sl = *slots[k];
            const auto tw = std::chrono::steady_clock::now();
            try{ waitFor(a, sl.done, timeout, "GPU read"); }
            catch(...){ sl.lost = true; throw; }
            tWait += secondsSince(tw);
            const uint32_t bad = reinterpret_cast<const uint32_t*>(sl.small)[0];
            if(bad){
                // (streams nvCOMP could not decompress whole: each read with a unit in that part is read again on the CPU)
                const Part &p = parts[(size_t)inSlot[k]];
                for(size_t i = p.begin; i < p.end; i++) rs[units[i].read].cpu = true;
            }
            inSlot[k] = -1;
        };

        for(size_t pi = 0; pi < parts.size(); pi++){
            const int k = (int)(pi % 2);
            checkSlot(k);
            Slot &sl = *slots[k];
            Part &p = parts[pi];
            layOut(units, p, c);
            growPinned(a, sl.host, sl.hostBytes, p.staging);
            uint8_t* host = sl.host;

            // The stored bytes read, and the units the GPU does not decode decoded, by the host's threads, a thread a
            // run of units (so that its handles move from directory to directory in order)
            const auto tf = std::chrono::steady_clock::now();
            std::unique_ptr<std::atomic<uint8_t>[]> readBad(new std::atomic<uint8_t>[rs.size()]);
            for(size_t i = 0; i < rs.size(); i++) readBad[i] = 0;
            // The part's reads: a unit each, but runs of uncompressed units that follow each other in the file and in
            // the staging read with one call each, of at most 4 MB (as the CPU reader reads a slice's strips: on
            // network storage every read call costs). A part that only reads hands each thread runs of about a quarter
            // of its share at a time, in the file's order: one at a time each thread's reads jumped around the file,
            // which made cold reads of small deflate strips from VAST 1.7-2.2 times as slow (the NFS client's
            // read-ahead follows each open file's reads); a part that decodes on the host (LZW with CPPTIFF_GPU_LZW=0,
            // and the codecs the GPU does not decode) hands them out one at a time, which balances the decoding (warm
            // LZW 1.13 times as fast as in runs; cold as fast), as runs of 16 did not (a part of a few dozen big strips
            // left most threads idle)
            std::vector<std::pair<size_t, size_t>> tasks;
            for(size_t i = p.begin; i < p.end; i++){
                const Planned &q = units[i];
                if(!tasks.empty() && q.how == Raw){
                    const size_t f = tasks.back().first, l = tasks.back().second - 1;
                    const Planned &prev = units[l];
                    if(prev.how == Raw && prev.read == q.read && prev.at + prev.stored == q.at && prev.stage + prev.stored == q.stage &&
                       q.stage + q.stored - units[f].stage <= ((uint64_t)4 << 20)){
                        tasks.back().second = i + 1;
                        continue;
                    }
                }
                tasks.push_back({i, i + 1});
            }
            bool decodes = false;
            for(size_t i = p.begin; i < p.end && !decodes; i++) decodes = units[i].how == Host;
            const int64_t run = decodes ? 1 : std::max<int64_t>(1, (int64_t)tasks.size()/(4*(int64_t)threads));
            #pragma omp parallel for num_threads(threads) schedule(dynamic, run)
            for(int64_t ti = 0; ti < (int64_t)tasks.size(); ti++){
                Planned &q = units[tasks[(size_t)ti].first];
                const Planned &last = units[tasks[(size_t)ti].second - 1];
                if(rs[q.read].cpu || readBad[q.read]) continue;
                const int t = omp_get_thread_num();
                TIFF* tif = hs.get(t, q.read);
                bool ok = tif != nullptr;
                uint8_t* dst = host + q.stage;
                if(ok && q.how == Host){
                    ok = toDirectory(tif, q.dir, rs[q.read].info);
                    const tmsize_t n = !ok ? -1 : q.tiled ? decodeTile(tif, q.strile, dst, (tmsize_t)q.decoded, hs.raw[(size_t)t])
                                                        : decodeStrip(tif, q.strile, dst, (tmsize_t)q.decoded, hs.raw[(size_t)t]);
                    ok = ok && n == (tmsize_t)q.decoded;
                }
                else if(ok){
                    const uint64_t n = last.stage + last.stored - q.stage;
                    thandle_t h = TIFFClientdata(tif);
                    ok = TIFFGetSeekProc(tif)(h, (toff_t)q.at, SEEK_SET) == (toff_t)q.at &&
                         (uint64_t)TIFFGetReadProc(tif)(h, dst, (tmsize_t)n) == n;
                    // (a zlib stream nvCOMP takes: deflate, no preset dictionary, a valid header check)
                    if(ok && q.how == GpuDeflate)
                        ok = (dst[0] & 0x0F) == 8 && (dst[0] >> 4) <= 7 && !(dst[1] & 0x20) && ((dst[0] << 8) | dst[1]) % 31 == 0;
                }
                if(!ok) readBad[q.read] = 1;
            }
            for(size_t i = 0; i < rs.size(); i++) if(readBad[i]) rs[i].cpu = true;
            tFill += secondsSince(tf);

            // The tables: the units (by element size), the predictor's list, nvCOMP's arrays
            growDevice(a, sl.comp, sl.compBytes, p.staging);
            growDevice(a, sl.dec, sl.decBytes, std::max<uint64_t>(p.decoded, 16));
            // (streams of reads read on the CPU instead left out of nvCOMP's arrays)
            uint32_t nActive[NKinds] = {0, 0, 0};
            for(size_t i = p.begin; i < p.end; i++){
                const Planned &q = units[i];
                if(rs[q.read].cpu) continue;
                if(q.how == GpuZstd) nActive[Zstd]++;
                else if(q.how == GpuDeflate) nActive[Deflate]++;
                else if(q.how == GpuLzw) nActive[LzwKind]++;
            }
            const uint32_t nCodecStreams = nActive[Zstd] + nActive[Deflate] + nActive[LzwKind];
            const uint64_t workNeed = 64 + alignUp((uint64_t)nCodecStreams*8, 64) + alignUp((uint64_t)nCodecStreams*4, 64);
            growDevice(a, sl.work, sl.workBytes, workNeed);
            Unit* table = reinterpret_cast<Unit*>(host + p.unitsAt);
            uint32_t* list = reinterpret_cast<uint32_t*>(host + p.listAt);
            uint64_t* arr = reinterpret_cast<uint64_t*>(host + p.arraysAt);
            uint64_t* inPtr = arr;
            uint64_t* inBytes = arr + nCodecStreams;
            uint64_t* outBytes = arr + 2*nCodecStreams;
            uint64_t* outPtr = arr + 3*nCodecStreams;
            uint32_t esAt[4] = {0, p.esCount[0], p.esCount[0] + p.esCount[1], p.esCount[0] + p.esCount[1] + p.esCount[2]};
            uint32_t esFill[4] = {esAt[0], esAt[1], esAt[2], esAt[3]};
            uint32_t nList = 0, zs = 0, ds = nActive[Zstd], ls = nActive[Zstd] + nActive[Deflate];
            for(size_t i = p.begin; i < p.end; i++){
                Planned &q = units[i];
                Unit u = q.u;
                const bool skip = rs[q.read].cpu;
                if(q.how == GpuZstd || q.how == GpuDeflate || q.how == GpuLzw){
                    u.src = q.decOff;
                    if(!skip){
                        const uint32_t j = q.how == GpuZstd ? zs++ : q.how == GpuDeflate ? ds++ : ls++;
                        const uint64_t skipHead = q.how == GpuDeflate ? 2 : 0, skipTail = q.how == GpuDeflate ? 4 : 0;
                        inPtr[j] = sl.comp + q.stage + skipHead;
                        inBytes[j] = q.stored - skipHead - skipTail;
                        outBytes[j] = q.decoded;
                        outPtr[j] = sl.dec + q.decOff;
                    }
                }
                else u.src = hostBit | q.stage;
                // (a read the CPU reads instead: its units are placed nowhere)
                if(skip) u.rows = 0;
                const uint32_t at = esFill[esIndex(u.es)]++;
                table[at] = u;
                if(u.predictor == PREDICTOR_HORIZONTAL && !skip) list[nList++] = at;
            }

            // The GPU's work: the copy, the decompression and its check, the predictor, the placing (on the slot's stream)
            const CUstream ss = sl.stream;
            ck(a, a->cuMemcpyHtoDAsync(sl.comp, host, p.staging, ss), "cuMemcpyHtoDAsync");
            const CUdeviceptr checkAt = sl.work, gotAt = sl.work + 64, statusAt = gotAt + alignUp((uint64_t)nCodecStreams*8, 64);
            ck(a, a->cuMemsetD8Async(checkAt, 0, 4, ss), "cuMemsetD8Async");
            ck(a, a->cuMemsetD8Async(checkAt + 4, 0xFF, 4, ss), "cuMemsetD8Async");
            uint32_t base = 0;
            for(int codec = 0; codec < NCodecs; codec++){
                const uint32_t n = nActive[codec];
                if(!n) continue;
                NvcompOpts o{};
                size_t temp = 0;
                const int tr = a->tempSize6[codec] ? a->tempSize6[codec](n, p.maxChunk[codec], o, &temp, p.total[codec], ss)
                                                   : a->tempSize5[codec](n, p.maxChunk[codec], o, &temp, p.total[codec]);
                if(tr) throw Fail("nvCOMP could not size its temporary memory (status "+std::to_string(tr)+")");
                growDevice(a, sl.temp, sl.tempBytes, std::max<size_t>(temp, 16));
                const CUdeviceptr arrays = sl.comp + p.arraysAt;
                const int dr = a->decompressAsync[codec](
                    reinterpret_cast<const void* const*>(arrays + 8*(uint64_t)base),
                    reinterpret_cast<const size_t*>(arrays + 8*((uint64_t)nCodecStreams + base)),
                    reinterpret_cast<const size_t*>(arrays + 8*(2*(uint64_t)nCodecStreams + base)),
                    reinterpret_cast<size_t*>(gotAt + 8*(uint64_t)base), n,
                    reinterpret_cast<void*>(sl.temp), sl.tempBytes,
                    reinterpret_cast<void* const*>(arrays + 8*(3*(uint64_t)nCodecStreams + base)), o,
                    reinterpret_cast<int*>(statusAt + 4*(uint64_t)base), ss);
                if(dr) throw Fail("nvCOMP could not start decompressing (status "+std::to_string(dr)+")");
                base += n;
            }
            if(nActive[LzwKind]){
                const CUdeviceptr arrays = sl.comp + p.arraysAt;
                CUdeviceptr ip = arrays + 8*(uint64_t)base, ib = arrays + 8*((uint64_t)nCodecStreams + base);
                CUdeviceptr ob = arrays + 8*(2*(uint64_t)nCodecStreams + base), op = arrays + 8*(3*(uint64_t)nCodecStreams + base);
                CUdeviceptr gp = gotAt + 8*(uint64_t)base, sp = statusAt + 4*(uint64_t)base;
                uint32_t n = nActive[LzwKind];
                void* args[] = {&ip, &ib, &ob, &op, &gp, &sp, &n};
                launch(a, c.lzw, std::min<uint32_t>(n, 65535), 1, 32, 1, ss, args);
                base += n;
            }
            if(nCodecStreams){
                CUdeviceptr st = statusAt, got = gotAt, want = sl.comp + p.arraysAt + 8*2*(uint64_t)nCodecStreams;
                CUdeviceptr badp = checkAt, firstp = checkAt + 4;
                uint32_t n = nCodecStreams;
                void* args[] = {&st, &got, &want, &n, &badp, &firstp};
                launch(a, c.check, std::min<uint32_t>(1024, (n + 255)/256), 1, 256, 1, ss, args);
            }
            ck(a, a->cuMemcpyDtoHAsync(sl.small, checkAt, 8, ss), "cuMemcpyDtoHAsync");
            CUdeviceptr unitsDev = sl.comp + p.unitsAt, dec = sl.dec, hostDev = sl.comp;
            if(nList){
                CUdeviceptr listDev = sl.comp + p.listAt;
                void* args[] = {&unitsDev, &listDev, &nList, &dec, &hostDev};
                launch(a, c.unpredict, nList, 1, 128, 1, ss, args);
            }
            for(int e = 0; e < 4; e++){
                uint32_t n = p.esCount[e];
                if(!n) continue;
                CUdeviceptr tab = unitsDev + (uint64_t)esAt[e]*sizeof(Unit);
                void* args[] = {&tab, &n, &dec, &hostDev};
                if(flipXY) launch(a, c.placeF[e], n, std::max<uint32_t>(1, std::min<uint32_t>(p.maxTiles[e], 65535)), 32, 8, ss, args);
                else launch(a, c.placeC[e], n, std::max<uint32_t>(1, std::min<uint32_t>((p.maxRows[e] + 3)/4, 65535)), 256, 1, ss, args);
            }
            ck(a, a->cuEventRecord(sl.done, ss), "cuEventRecord");
            inSlot[k] = (long long)pi;
        }
        checkSlot(0);
        checkSlot(1);

        // The reads the GPU path did not take, read on the CPU and copied (after the GPU's work, which may have
        // written parts of them)
        t0 = std::chrono::steady_clock::now();
        // (CPPTIFF_GPU_STRICT=1, for tests: such a read fails instead, so that a GPU path that does not work cannot
        // pass for one that does)
        const bool strict = std::getenv("CPPTIFF_GPU_STRICT") && std::atoi(std::getenv("CPPTIFF_GPU_STRICT")) == 1;
        for(auto &q : rs){
            if(!q.cpu) continue;
            if(strict) throw Fail("\""+q.r->fileName+"\" would have been read on the CPU (CPPTIFF_GPU_STRICT)");
            nCpu++;
            std::vector<uint8_t> buf(q.bytes);
            std::vector<uint64_t> zr;
            if(q.r->zRange.size()) zr = {q.start, q.start + q.z};
            if(!readTiffParallelInfo(q.r->fileName.c_str(), q.info, buf.data(), flipXY ? 1 : 0, zr))
                throw Fail("File \""+q.r->fileName+"\" could not be read");
            ck(a, a->cuMemcpyHtoDAsync((CUdeviceptr)(uintptr_t)q.r->out, buf.data(), q.bytes, s), "cuMemcpyHtoDAsync");
            ck(a, a->cuStreamSynchronize(s), "cuStreamSynchronize");
        }
        tCpu = secondsSince(t0);
        if(profile)
            std::fprintf(stderr, "cpp-tiff GPU read: %zu reads in %zu parts; %.1f MB stored as is, %.1f MB for nvCOMP (%.1f MB decoded), "
                         "%.1f MB for the GPU's LZW decoder (%.1f MB decoded), %.1f MB decoded on the host; %zu read on the CPU instead. "
                         "Info %.3f s, plan %.3f s, host reads and decoding %.3f s, waiting for the GPU %.3f s, CPU reads %.3f s; %.3f s in "
                         "all\n", reads.size(), nParts, bRaw/1e6, bGpu/1e6, bGpuOut/1e6, bLzw/1e6, bLzwOut/1e6, bHost/1e6, (size_t)nCpu,
                         tInfo, tPlan, tFill, tWait, tCpu, secondsSince(tStart));
    }
    catch(const Fail &e){
        errString = e.what();
        return 1;
    }
    catch(const std::bad_alloc&){
        errString = "Not enough host memory for the GPU read";
        return 1;
    }
    return 0;
}
