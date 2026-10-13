#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"

template <typename T>
pybind11::array_t<T> create_pybind11_array(void* data, const uint64_t* dims, uint64_t samples) {
    auto deleter = [](void* ptr) { free(ptr); };

    if(samples > 1){
        // Channel-last interleaved buffer (z, y, x, c). Shape (x, y, z, c); the
        // wrapper transposes (2,1,0,3) -> (z, y, x, c). Strides are in bytes.
        std::vector<ssize_t> strides = {
            static_cast<ssize_t>(samples * sizeof(T)),
            static_cast<ssize_t>(dims[1] * samples * sizeof(T)),
            static_cast<ssize_t>(dims[1] * dims[0] * samples * sizeof(T)),
            static_cast<ssize_t>(sizeof(T))
        };
        return pybind11::array_t<T>(
            {dims[1], dims[0], dims[2], samples},
            strides,
            static_cast<T*>(data),
            pybind11::capsule(data, deleter)
        );
    }

    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(T)),
        static_cast<ssize_t>(dims[1] * sizeof(T)),
        static_cast<ssize_t>(dims[1] * dims[0] * sizeof(T))
    };

    return pybind11::array_t<T>(
        {dims[1], dims[0], dims[2]},  // shape (y, x, z)
        strides,
        static_cast<T*>(data),
        pybind11::capsule(data, deleter)
    );
}

// The NumPy data type of a file's samples: bits and sample format (2 signed
// integers, 3 floats, otherwise unsigned integers)
static pybind11::dtype numpyDtype(const uint64_t bits, const uint64_t fmt){
    const std::string kind = fmt == 3 && bits >= 32 ? "float" : fmt == 2 ? "int" : "uint";
    return pybind11::dtype(kind + std::to_string(bits));
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
pybind11::array pybind11_read_tiff(const std::string& fileName, const std::vector<uint64_t> &zRange = {},
                                   const int64_t numThreads = 0, const pybind11::object &out = pybind11::none()){
    if(numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const CallThreads threads(numThreads);
    TiffInfo tinfo;
    uint8_t bad = 0;
    {
        pybind11::gil_scoped_release release;
        bad = getTiffInfo(fileName.c_str(), tinfo);
    }
    if(bad) throw std::runtime_error("Could not open \"" + fileName + "\" as a TIFF");
    uint64_t dims[3] = {tinfo.y, tinfo.x, tinfo.z};
    if(zRange.size()) dims[2] = zRange.size() == 2 ? zRange[1]-zRange[0] : 1;
    const uint64_t dtype = tinfo.bits, samples = tinfo.samples, fmt = tinfo.sampleFormat;
    const std::string readError = "Failed to read \"" + fileName + "\": the file is damaged or uses an unsupported layout, such as samples that are not 8, 16, 32 or 64 bits";

    // Into the caller's array (out): the shape read_tiff returns, (z, y, x[, c])
    // or (y, x[, c]) for one slice, the file's data type, writeable and
    // C-contiguous. It is returned as it is.
    if(!out.is_none()){
        if(!pybind11::isinstance<pybind11::array>(out)) throw std::runtime_error("out must be a NumPy array");
        pybind11::array a = out.cast<pybind11::array>();
        if(dtype != 8 && dtype != 16 && dtype != 32 && dtype != 64) throw std::runtime_error(readError);
        const pybind11::dtype dt = numpyDtype(dtype, fmt);
        if(!a.dtype().equal(dt)){
            throw std::runtime_error("out has data type " + pybind11::str(a.dtype()).cast<std::string>() +
                                     ", not the file's " + pybind11::str(dt).cast<std::string>());
        }
        std::vector<uint64_t> want = {dims[2], dims[0], dims[1]};
        if(samples > 1) want.push_back(samples);
        std::vector<uint64_t> squeezed(want.begin() + (dims[2] == 1 ? 1 : 0), want.end());
        std::vector<uint64_t> have(a.ndim());
        for(ssize_t d = 0; d < a.ndim(); d++) have[d] = (uint64_t)a.shape(d);
        if(have != want && have != squeezed){
            std::string text;
            for(size_t d = 0; d < want.size(); d++) text += (d ? ", " : "") + std::to_string(want[d]);
            throw std::runtime_error("out must have the shape read_tiff returns, (" + text + ")");
        }
        const pybind11::object flags = a.attr("flags");
        if(!flags.attr("writeable").cast<bool>()) throw std::runtime_error("out must be writeable");
        if(!flags.attr("c_contiguous").cast<bool>()) throw std::runtime_error("out must be C-contiguous");
        void* ptr = a.mutable_data();
        void* data = NULL;
        {
            pybind11::gil_scoped_release release;
            data = readTiffParallelInfo(fileName.c_str(), tinfo, ptr, 0, zRange);
        }
        if(!data) throw std::runtime_error(readError);
        return a;
    }

    void* data = NULL;
    {
        pybind11::gil_scoped_release release;
        data = readTiffParallelInfo(fileName.c_str(), tinfo, NULL, 0, zRange);
    }
	if(!data) throw std::runtime_error(readError);

	switch (dtype) {
        case 8:
            if(fmt == 2) return create_pybind11_array<int8_t>(data, dims, samples);
            return create_pybind11_array<uint8_t>(data, dims, samples);
        case 16:
            if(fmt == 2) return create_pybind11_array<int16_t>(data, dims, samples);
            return create_pybind11_array<uint16_t>(data, dims, samples);
        case 32:
            if(fmt == 3) return create_pybind11_array<float>(data, dims, samples);
            if(fmt == 2) return create_pybind11_array<int32_t>(data, dims, samples);
            return create_pybind11_array<uint32_t>(data, dims, samples);
        case 64:
            if(fmt == 3) return create_pybind11_array<double>(data, dims, samples);
            if(fmt == 2) return create_pybind11_array<int64_t>(data, dims, samples);
            return create_pybind11_array<uint64_t>(data, dims, samples);
        default:
            free(data);
            throw std::runtime_error("Unsupported data type");
    }
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
    uint16_t sampleFormat;
    if (kind == 'u')      sampleFormat = 1;
    else if (kind == 'i') sampleFormat = 2;
    else if (kind == 'f') sampleFormat = 3;
    else throw std::runtime_error("Unsupported data type");
    if (dtype != 8 && dtype != 16 && dtype != 32 && dtype != 64)
        throw std::runtime_error("Unsupported data type");

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

    // Call the function to write the data to a TIFF file. The library prints
    // why a write failed; without this check Python never heard about it.
    uint8_t err = 0;
    {
        pybind11::gil_scoped_release release;
        err = writeTiffParallelHelper(fileName.c_str(), info.ptr, dtype, "w", dims[0], dims[1], dims[2], 0, transpose, compression, sampleFormat, tile, dims[3]);
    }
    if(err) throw std::runtime_error("Failed to write \"" + fileName + "\"");
}

// z is the number of directories (an ImageJ stack's is 1), as before.
pybind11::tuple pybind11_get_image_shape(const std::string& fileName){
    TiffInfo tinfo;
    uint8_t bad = 0;
    {
        pybind11::gil_scoped_release release;
        bad = getTiffInfo(fileName.c_str(), tinfo);
    }
    if(bad) throw std::runtime_error("Could not open \"" + fileName + "\" as a TIFF");
    const uint64_t samples = tinfo.samples, z = tinfo.directories, y = tinfo.y, x = tinfo.x;
	if(samples > 1) return pybind11::make_tuple(z, y, x, samples);
	return pybind11::make_tuple(z, y, x);
}


PYBIND11_MODULE(cpptiff, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-tiff python bindings";

	m.def("pybind11_read_tiff", &pybind11_read_tiff, pybind11::arg("fileName"), pybind11::arg("zRange") = std::vector<uint64_t>{},
	      pybind11::arg("numThreads") = 0, pybind11::arg("out") = pybind11::none(), "Read a tiff file");

	m.def("pybind11_write_tiff", &pybind11_write_tiff, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("transpose"), pybind11::arg("compression"), pybind11::arg("tile") = 0,
	      pybind11::arg("numThreads") = 0, "Write a tiff file");

	m.def("pybind11_get_image_shape", &pybind11_get_image_shape, "Get the image shape without reading the entire image");

	m.def("pybind11_default_threads", &callThreads,
	      "The threads a call uses by default from the calling thread (OMP_NUM_THREADS, or every CPU)");
}
