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

pybind11::array pybind11_read_tiff(const std::string& fileName, const std::vector<uint64_t> &zRange = {}){
	uint64_t* dimsPtr = getImageSize(fileName.c_str());
	uint64_t dims[3] = {dimsPtr[0], dimsPtr[1], dimsPtr[2]};
	free(dimsPtr);
    if(isImageJIm(fileName.c_str())){
        uint64_t tempZ = imageJImGetZ(fileName.c_str());
        if(tempZ) dims[2] = tempZ;
    }
	uint64_t dtype = getDataType(fileName.c_str());
	uint64_t samples = getSamplesPerPixel(fileName.c_str());
	uint64_t fmt = getSampleFormat(fileName.c_str());  // 1=uint, 2=int, 3=float

	void* data = readTiffParallelWrapperNoXYFlip(fileName.c_str(), zRange);
	if(!data) throw std::runtime_error("Failed to read TIFF (unsupported format, e.g. planar RGB)");

    if(zRange.size()){
        if(zRange.size() == 2){
            dims[2] = zRange[1]-zRange[0];
        }
        else{
            dims[2] = 1;
        }
    }

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
            throw std::runtime_error("Unsupported data type");
    }
}

void pybind11_write_tiff(const std::string &fileName, const pybind11::array &data, const bool transpose = false, const std::string &compression = "lzw") {
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

    // Get the dimensions of the array
    uint64_t dims[3] = {static_cast<uint64_t>(info.shape[0]), static_cast<uint64_t>(info.shape[1]), static_cast<uint64_t>(info.shape[2])};

    // Call the function to write the data to a TIFF file
    writeTiffParallelHelper(fileName.c_str(), info.ptr, dtype, "w", dims[0], dims[1], dims[2], 0, transpose, compression, sampleFormat);
}

pybind11::tuple pybind11_get_image_shape(const std::string& fileName){
    uint64_t* dims = getImageSize(fileName.c_str());
    uint64_t samples = getSamplesPerPixel(fileName.c_str());
    uint64_t z = dims[2], y = dims[0], x = dims[1];
    free(dims);
	if(samples > 1) return pybind11::make_tuple(z, y, x, samples);
	return pybind11::make_tuple(z, y, x);
}


PYBIND11_MODULE(cpptiff, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-tiff python bindings";

	m.def("pybind11_read_tiff", &pybind11_read_tiff, "Read a tiff file");

	m.def("pybind11_write_tiff", &pybind11_write_tiff, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("transpose"), pybind11::arg("compression"), "Write a tiff file");

	m.def("pybind11_get_image_shape", &pybind11_get_image_shape, "Get the image shape without reading the entire image");
}
