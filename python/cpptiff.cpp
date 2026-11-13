#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"
#include "gpureadtiff.h"

// The NumPy data type of a file's samples: bits and sample format (2 signed
// integers, 3 floats, otherwise unsigned integers). 16-bit floats are float16:
// they were read back as uint16.
static pybind11::dtype numpyDtype(const uint64_t bits, const uint64_t fmt){
    const std::string kind = fmt == 3 && bits >= 16 ? "float" : fmt == 2 ? "int" : "uint";
    return pybind11::dtype(kind + std::to_string(bits));
}

// A NumPy array over the reader's buffer, which it frees. Row-major data, (z, y,
// x[, c]), is shaped (x, y, z[, c]) for the wrapper to transpose; MATLAB's
// column-major layout (fortran) is returned as it is, (y, x[, c], z).
static pybind11::array wrapBuffer(void* data, const pybind11::dtype& dt, const uint64_t y, const uint64_t x,
                                  const uint64_t z, const uint64_t samples, const bool fortran){
    const pybind11::capsule owner(data, [](void* p){ free(p); });
    const ssize_t e = dt.itemsize(), Y = (ssize_t)y, X = (ssize_t)x, Z = (ssize_t)z, C = (ssize_t)samples;
    std::vector<ssize_t> shape, strides;
    if(fortran){
        shape = {Y, X};
        strides = {e, Y*e};
        if(samples > 1){ shape.push_back(C); strides.push_back(Y*X*e); }
        shape.push_back(Z);
        strides.push_back(Y*X*C*e);
    }
    else if(samples > 1){
        shape = {X, Y, Z, C};
        strides = {C*e, X*C*e, X*Y*C*e, e};
    }
    else{
        shape = {X, Y, Z};
        strides = {e, X*e, X*Y*e};
    }
    return pybind11::array(dt, shape, strides, data, owner);
}

static std::string shapeText(const std::vector<uint64_t>& shape){
    std::string text;
    for(size_t d = 0; d < shape.size(); d++) text += (d ? ", " : "") + std::to_string(shape[d]);
    return "(" + text + (shape.size() == 1 ? ",)" : ")");
}

// A call's threads (numThreads; 0 for the calling thread's default:
// OMP_NUM_THREADS, or every CPU), set for the calling thread for the whole
// call and then restored.
struct CallThreads {
    int prev = 0;
    explicit CallThreads(const int64_t numThreads){
        if(numThreads > 0){
            prev = callThreads();
            setCallThreads((int)numThreads);
        }
    }
    ~CallThreads(){
        if(prev > 0) setCallThreads(prev);
    }
};

// The GIL is released while the file is opened and read, so other Python
// threads run, and read other files, meanwhile. The file's metadata comes from
// one open and one walk of its directories: the separate getImageSize,
// isImageJIm, getDataType, getSamplesPerPixel and getSampleFormat calls opened
// it six times and walked its directories once more than the read itself, so
// small images now read 1.3-1.5x faster and 2000 slices of 64 x 64 1.6x.
// order "F" reads into MATLAB's column-major layout, (y, x[, c], z), as
// parallelReadTiff does: the reader fills it as it decodes, while a copy from
// (z, y, x) took longer than the whole read.
pybind11::array pybind11_read_tiff(const std::string& fileName, const std::vector<uint64_t> &zRange = {},
                                   const int64_t numThreads = 0, const pybind11::object &out = pybind11::none(),
                                   const std::string &order = "C"){
    if(numThreads < 0) throw pybind11::value_error("num_threads must be at least 1");
    if(order != "C" && order != "F") throw pybind11::value_error("order must be 'C' or 'F', not '" + order + "'");
    const bool fortran = order == "F";
    const CallThreads threads(numThreads);
    TiffInfo tinfo;
    uint8_t bad = 0;
    {
        pybind11::gil_scoped_release release;
        bad = getTiffInfo(fileName.c_str(), tinfo);
    }
    if(bad) throw std::runtime_error("Could not open \"" + fileName + "\" as a TIFF");

    // The slices: all of them, one (zRange {s}), or [start, end). Slices past
    // the end failed with "the file is damaged" and printed the readers' errors.
    uint64_t z = tinfo.z;
    if(zRange.size()){
        if(zRange.size() > 2)
            throw pybind11::value_error("z_range is one slice or [start, end), not " + std::to_string(zRange.size()) + " numbers");
        const uint64_t end = zRange.size() == 2 ? zRange[1] : zRange[0] + 1;
        const std::string given = zRange.size() == 2 ? "[" + std::to_string(zRange[0]) + ", " + std::to_string(zRange[1]) + "]"
                                                     : std::to_string(zRange[0]);
        if(zRange[0] >= end) throw pybind11::value_error("z_range " + given + " is empty: its start must be less than its end");
        if(end > tinfo.z)
            throw pybind11::index_error("z_range " + given + " is out of bounds: \"" + fileName + "\" has " + std::to_string(tinfo.z) +
                                        (tinfo.z == 1 ? " slice (0)" : " slices (0 to " + std::to_string(tinfo.z - 1) + ")"));
        z = end - zRange[0];
    }
    const uint64_t y = tinfo.y, x = tinfo.x, bits = tinfo.bits, samples = tinfo.samples, fmt = tinfo.sampleFormat;
    const std::string readError = "Failed to read \"" + fileName + "\": the file is damaged or uses an unsupported layout, such as samples that are not 8, 16, 32 or 64 bits";
    if(bits != 8 && bits != 16 && bits != 32 && bits != 64) throw std::runtime_error(readError);
    const pybind11::dtype dt = numpyDtype(bits, fmt);

    // Into the caller's array (out): the shape read_tiff returns, with or
    // without a z axis of 1 for one slice -- (z, y, x[, c]) C-contiguous, or for
    // order F (y, x[, c], z) F-contiguous -- the file's data type, and
    // writeable. It is returned as it is.
    if(!out.is_none()){
        if(!pybind11::isinstance<pybind11::array>(out)) throw pybind11::type_error("out must be a NumPy array");
        pybind11::array a = out.cast<pybind11::array>();
        if(!a.dtype().equal(dt)){
            throw pybind11::value_error("out has data type " + pybind11::str(a.dtype()).cast<std::string>() +
                                        ", not the file's " + pybind11::str(dt).cast<std::string>());
        }
        // Both layouts' shapes, without and with the z axis of a single slice
        std::vector<uint64_t> cShape = {z, y, x}, fShape = {y, x};
        if(samples > 1){ cShape.push_back(samples); fShape.push_back(samples); }
        fShape.push_back(z);
        const std::vector<uint64_t> cOne(cShape.begin() + 1, cShape.end()), fOne(fShape.begin(), fShape.end() - 1);
        const std::vector<uint64_t>& want = fortran ? fShape : cShape;
        const std::vector<uint64_t>& wantOne = fortran ? fOne : cOne;
        std::vector<uint64_t> have(a.ndim());
        for(ssize_t d = 0; d < a.ndim(); d++) have[d] = (uint64_t)a.shape(d);
        const pybind11::object flags = a.attr("flags");
        const bool cContiguous = flags.attr("c_contiguous").cast<bool>(), fContiguous = flags.attr("f_contiguous").cast<bool>();
        if(have != want && !(z == 1 && have == wantOne)){
            std::string text = "out must have the shape read_tiff returns, " + shapeText(want);
            // (an array in the other layout most likely meant the other order)
            const std::vector<uint64_t>& other = fortran ? cShape : fShape;
            const std::vector<uint64_t>& otherOne = fortran ? cOne : fOne;
            if((have == other || (z == 1 && have == otherOne)) && (fortran ? cContiguous : fContiguous))
                text += fortran ? ": a (z, y, x) C-contiguous out reads with order='C'"
                                : ": a (y, x, z) F-contiguous out, MATLAB's layout, reads with order='F'";
            throw pybind11::value_error(text);
        }
        if(!flags.attr("writeable").cast<bool>()) throw pybind11::value_error("out must be writeable");
        if(fortran && !fContiguous) throw pybind11::value_error("out must be F-contiguous for order='F'");
        if(!fortran && !cContiguous) throw pybind11::value_error("out must be C-contiguous (or pass order='F' for a (y, x, z) F-contiguous out)");
        void* ptr = a.mutable_data();
        void* data = NULL;
        {
            pybind11::gil_scoped_release release;
            data = readTiffParallelInfo(fileName.c_str(), tinfo, ptr, fortran ? 1 : 0, zRange);
        }
        if(!data) throw std::runtime_error(readError);
        return a;
    }

    void* data = NULL;
    {
        pybind11::gil_scoped_release release;
        data = readTiffParallelInfo(fileName.c_str(), tinfo, NULL, fortran ? 1 : 0, zRange);
    }
    if(!data) throw std::runtime_error(readError);
    return wrapBuffer(data, dt, y, x, z, samples, fortran);
}

// The GIL is released while the file is written, so other Python threads run
// meanwhile.
void pybind11_write_tiff(const std::string &fileName, const pybind11::array &data, const bool transpose = false, const std::string &compression = "lzw", const uint64_t tile = 0,
                         const int64_t numThreads = 0) {
    if(numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const CallThreads threads(numThreads);
    // Determine the dtype based on the NumPy array type
    pybind11::buffer_info info = data.request();

    // Map the NumPy dtype to (bit depth, TIFF sample format) by kind + width. Using the
    // dtype kind avoids the long/long-long format-string ambiguity that breaks 64-bit matching.
    char kind = data.dtype().kind();   // 'u' = unsigned int, 'i' = signed int, 'f' = float
    uint64_t dtype = (uint64_t)(info.itemsize * 8);
    uint16_t sampleFormat = 1;
    if (kind == 'i') sampleFormat = 2;
    else if (kind == 'f') sampleFormat = 3;
    if ((kind != 'u' && kind != 'i' && kind != 'f') || (dtype != 8 && dtype != 16 && dtype != 32 && dtype != 64) ||
        (kind == 'f' && dtype == 8))
        throw pybind11::type_error("data type " + pybind11::str(data.dtype()).cast<std::string>() + " cannot be written: "
                                   "use integers of 8, 16, 32 or 64 bits, or floats of 16, 32 or 64 bits");
    // The bytes are written as they are, so data in the other byte order was
    // written with every value wrong (write_tiff converts it).
    if (!data.dtype().attr("isnative").cast<bool>())
        throw pybind11::value_error("data must be in this machine's byte order");

    // Get the dimensions of the array: (x, y[, z[, c]]), the wrapper's view of
    // row-major (z, y, x[, c]) data. A 2D array only has two entries in shape,
    // so reading shape[2] unconditionally is out of bounds -- it fed a garbage z
    // to the writer (seen producing a 247 MB file for a 24x40 image).
    if (info.ndim < 2 || info.ndim > 4)
        throw std::runtime_error("data must be 2D, 3D, or 4D color data");
    uint64_t dims[4] = {static_cast<uint64_t>(info.shape[0]),
                        static_cast<uint64_t>(info.shape[1]),
                        info.ndim > 2 ? static_cast<uint64_t>(info.shape[2]) : 1,
                        info.ndim > 3 ? static_cast<uint64_t>(info.shape[3]) : 1};
    if (dims[3] != 1 && dims[3] != 3 && dims[3] != 4)
        throw std::runtime_error("color data must have 3 (RGB) or 4 (RGBA) samples");
    // An empty array was written as one slice read from past its end (the
    // library turns z = 0 into one slice).
    for (ssize_t d = 0; d < info.ndim; d++)
        if (info.shape[d] == 0) throw pybind11::value_error("cannot write an empty array: a TIFF image has at least one pixel");
    // The writer reads the memory directly, ignoring strides, so the array must
    // be laid out as the wrapper passes it: (x, y[, z[, c]]) over row-major (z,
    // y, x[, c]) data, or with transpose over MATLAB's column-major (y, x[, c],
    // z) data. (Axes of length 1 can have any stride.)
    {
        const ssize_t e = info.itemsize, X = (ssize_t)dims[0], Y = (ssize_t)dims[1], C = (ssize_t)dims[3];
        const ssize_t want[4] = {transpose ? Y*e : C*e, transpose ? e : X*C*e, X*Y*C*e, transpose ? X*Y*e : e};
        for (ssize_t d = 0; d < info.ndim; d++)
            if (info.shape[d] > 1 && info.strides[d] != want[d])
                throw pybind11::value_error("data is not laid out as the writer reads it: write it with write_tiff");
    }

    // Call the function to write the data to a TIFF file. The library prints
    // why a write failed; without this check Python never heard about it.
    uint8_t err = 0;
    {
        pybind11::gil_scoped_release release;
        err = writeTiffParallelHelper(fileName.c_str(), info.ptr, dtype, "w", dims[0], dims[1], dims[2], 0, transpose, compression, sampleFormat, tile, dims[3]);
    }
    if(err) throw std::runtime_error("Failed to write \"" + fileName + "\"");
}

// The shape read_tiff returns for the whole file, (z, y, x[, c]). z is an ImageJ
// stack's slices, which read_tiff reads: it was the stack's one directory.
pybind11::tuple pybind11_get_image_shape(const std::string& fileName){
    TiffInfo tinfo;
    uint8_t bad = 0;
    {
        pybind11::gil_scoped_release release;
        bad = getTiffInfo(fileName.c_str(), tinfo);
    }
    if(bad) throw std::runtime_error("Could not open \"" + fileName + "\" as a TIFF");
    const uint64_t samples = tinfo.samples, z = tinfo.z, y = tinfo.y, x = tinfo.x;
	if(samples > 1) return pybind11::make_tuple(z, y, x, samples);
	return pybind11::make_tuple(z, y, x);
}


// The slices z_range picks of a file: (start, count); throws as pybind11_read_tiff does for a bad range
static std::pair<uint64_t, uint64_t> sliceRange(const std::string &fileName, const TiffInfo &tinfo, const std::vector<uint64_t> &zRange){
    if(!zRange.size()) return {0, tinfo.z};
    if(zRange.size() > 2)
        throw pybind11::value_error("z_range is one slice or [start, end), not " + std::to_string(zRange.size()) + " numbers");
    const uint64_t end = zRange.size() == 2 ? zRange[1] : zRange[0] + 1;
    const std::string given = zRange.size() == 2 ? "[" + std::to_string(zRange[0]) + ", " + std::to_string(zRange[1]) + "]"
                                                 : std::to_string(zRange[0]);
    if(zRange[0] >= end) throw pybind11::value_error("z_range " + given + " is empty: its start must be less than its end");
    if(end > tinfo.z)
        throw pybind11::index_error("z_range " + given + " is out of bounds: \"" + fileName + "\" has " + std::to_string(tinfo.z) +
                                    (tinfo.z == 1 ? " slice (0)" : " slices (0 to " + std::to_string(tinfo.z - 1) + ")"));
    return {zRange[0], end - zRange[0]};
}

// What read_tiff returns for a file's slices, for new GPU arrays: (shape, NumPy data type name), the shape in
// order's layout without the z axis of a single slice, as read_tiff gives NumPy arrays
pybind11::tuple pybind11_read_shape(const std::string &fileName, const std::vector<uint64_t> &zRange, const std::string &order){
    TiffInfo tinfo;
    uint8_t bad = 0;
    {
        pybind11::gil_scoped_release release;
        bad = getTiffInfo(fileName.c_str(), tinfo);
    }
    if(bad) throw std::runtime_error("Could not open \"" + fileName + "\" as a TIFF");
    const uint64_t z = sliceRange(fileName, tinfo, zRange).second;
    if(tinfo.bits != 8 && tinfo.bits != 16 && tinfo.bits != 32 && tinfo.bits != 64)
        throw std::runtime_error("\"" + fileName + "\" has " + std::to_string(tinfo.bits) + "-bit samples, which cannot be read");
    std::vector<uint64_t> shape;
    if(order == "F"){
        shape = {tinfo.y, tinfo.x};
        if(tinfo.samples > 1) shape.push_back(tinfo.samples);
        if(z > 1) shape.push_back(z);
    }
    else{
        if(z > 1) shape.push_back(z);
        shape.insert(shape.end(), {tinfo.y, tinfo.x});
        if(tinfo.samples > 1) shape.push_back(tinfo.samples);
    }
    return pybind11::make_tuple(shape, pybind11::str(numpyDtype(tinfo.bits, tinfo.sampleFormat)));
}

// Reads into GPU memory: requests of (file name, z range, device pointer, the memory's NumPy type string,
// its shape), all in order's layout (C: (z, y, x[, c]); F: MATLAB's (y, x[, c], z)), with the GPU work on
// stream (a CUDA stream handle, 0 for the default stream); each request's memory checked against its
// file as out= is for read_tiff. The GIL is released while the files are read.
void pybind11_read_tiff_gpu(const std::vector<std::tuple<std::string, std::vector<uint64_t>, uint64_t, std::string, std::vector<uint64_t>>> &requests,
                            const std::string &order, const int64_t numThreads, const uint64_t stream){
    if(numThreads < 0) throw pybind11::value_error("num_threads must be at least 1");
    if(order != "C" && order != "F") throw pybind11::value_error("order must be 'C' or 'F', not '" + order + "'");
    const bool fortran = order == "F";
    std::vector<GpuTiffRead> reads;
    for(const auto &r : requests){
        const std::string &fileName = std::get<0>(r);
        const std::vector<uint64_t> &zRange = std::get<1>(r);
        TiffInfo tinfo;
        uint8_t bad = 0;
        {
            pybind11::gil_scoped_release release;
            bad = getTiffInfo(fileName.c_str(), tinfo);
        }
        if(bad) throw std::runtime_error("Could not open \"" + fileName + "\" as a TIFF");
        const uint64_t z = sliceRange(fileName, tinfo, zRange).second;
        if(tinfo.bits != 8 && tinfo.bits != 16 && tinfo.bits != 32 && tinfo.bits != 64)
            throw std::runtime_error("\"" + fileName + "\" has " + std::to_string(tinfo.bits) + "-bit samples, which cannot be read");
        const std::string want = pybind11::str(numpyDtype(tinfo.bits, tinfo.sampleFormat).attr("str")).cast<std::string>();
        const std::string &have = std::get<3>(r);
        // (one-byte types: NumPy's "|u1" and "<u1" are the same)
        if(have != want && !(have.size() == 3 && want.size() == 3 && have.substr(1) == want.substr(1) && want[2] == '1'))
            throw pybind11::value_error("out has data type " + have + ", not the file's " + want + " (\"" + fileName + "\")");
        std::vector<uint64_t> cShape = {z, tinfo.y, tinfo.x}, fShape = {tinfo.y, tinfo.x};
        if(tinfo.samples > 1){ cShape.push_back(tinfo.samples); fShape.push_back(tinfo.samples); }
        fShape.push_back(z);
        const std::vector<uint64_t> cOne(cShape.begin() + 1, cShape.end()), fOne(fShape.begin(), fShape.end() - 1);
        const std::vector<uint64_t> &shape = std::get<4>(r), &wantShape = fortran ? fShape : cShape, &wantOne = fortran ? fOne : cOne;
        if(shape != wantShape && !(z == 1 && shape == wantOne))
            throw pybind11::value_error("out must have the shape read_tiff returns, " + shapeText(wantShape) + " (\"" + fileName + "\")");
        reads.push_back(GpuTiffRead{fileName, zRange, (void*)(uintptr_t)std::get<2>(r)});
    }
    const CallThreads threads(numThreads);
    std::string err;
    uint8_t bad = 0;
    {
        pybind11::gil_scoped_release release;
        bad = gpuReadTiff(reads, fortran, (void*)(uintptr_t)stream, (int)numThreads, err);
    }
    if(bad) throw std::runtime_error(err);
}

pybind11::tuple pybind11_gpu_available(){
    std::string why;
    const bool ok = gpuReadTiffAvailable(why);
    return pybind11::make_tuple(ok, why);
}

PYBIND11_MODULE(cpptiff, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-tiff python bindings";

	m.def("pybind11_read_tiff", &pybind11_read_tiff, pybind11::arg("fileName"), pybind11::arg("zRange") = std::vector<uint64_t>{},
	      pybind11::arg("numThreads") = 0, pybind11::arg("out") = pybind11::none(), pybind11::arg("order") = "C", "Read a tiff file");

	m.def("pybind11_write_tiff", &pybind11_write_tiff, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("transpose"), pybind11::arg("compression"), pybind11::arg("tile") = 0,
	      pybind11::arg("numThreads") = 0, "Write a tiff file");

	m.def("pybind11_get_image_shape", &pybind11_get_image_shape, "Get the image shape without reading the entire image");

	m.def("pybind11_read_shape", &pybind11_read_shape, "The shape and data type read_tiff returns for a file's slices");

	m.def("pybind11_read_tiff_gpu", &pybind11_read_tiff_gpu, pybind11::arg("requests"), pybind11::arg("order"),
	      pybind11::arg("numThreads") = 0, pybind11::arg("stream") = 0, "Read tiff files into GPU memory");

	m.def("pybind11_gpu_available", &pybind11_gpu_available, "Whether GPU reads can run here, and if not, why");

	m.def("pybind11_gpu_set_nvcomp", &gpuReadTiffSetNvcompLibrary, "Where to load nvCOMP from, before the first GPU read");

	m.def("pybind11_default_threads", &callThreads,
	      "The threads a call uses by default from the calling thread (OMP_NUM_THREADS, or every CPU)");
}
