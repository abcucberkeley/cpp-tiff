#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"

pybind11::array pybind11_read_tiff(const std::string& fileName){
	uint64_t* dims = getImageSize(fileName.c_str());
	uint64_t dtype = getDataType(fileName.c_str());

	void* data = readTiffParallelWrapperNoXYFlip(fileName.c_str());

	// Custom deleter to release memory when Python array is destroyed
    auto deleter = [](void* ptr) {
        free(ptr);
    };

	switch (dtype) {
        case 8:  // 8-bit unsigned int
            return pybind11::array_t<uint8_t>(
                {dims[1], dims[0], dims[2]},  // shape (x, y, z)
                {sizeof(uint8_t),            // strides
                    dims[1] * sizeof(uint8_t),  
                    dims[1] * dims[0] * sizeof(uint8_t)},
                static_cast<uint8_t*>(data),
                pybind11::capsule(data, deleter)
            ); 
		case 16: // 16-bit unsigned int
            return pybind11::array_t<uint16_t>(
                {dims[1], dims[0], dims[2]},  // shape (x, y, z)
                {sizeof(uint16_t),            // strides
                    dims[1] * sizeof(uint16_t),
                    dims[1] * dims[0] * sizeof(uint16_t)},
                static_cast<uint16_t*>(data),
                pybind11::capsule(data, deleter)
            );
        case 32: // 32-bit float
            return pybind11::array_t<float>(
                {dims[1], dims[0], dims[2]},  // shape (x, y, z)
                {sizeof(float),            // strides
                    dims[1] * sizeof(float),  
                    dims[1] * dims[0] * sizeof(float)},
                static_cast<float*>(data),
                pybind11::capsule(data, deleter)
            );
        case 64: // 64-bit double
            return pybind11::array_t<double>(
                {dims[1], dims[0], dims[2]},  // shape (x, y, z)
                {sizeof(double),            // strides
                    dims[1] * sizeof(double),  
                    dims[1] * dims[0] * sizeof(double)},
                static_cast<double*>(data),
                pybind11::capsule(data, deleter)
            );
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

void pybind11_write_tiff(const std::string &fileName, const pybind11::array &data, const std::string &compression = "lzw") {
    // Determine the dtype based on the NumPy array type
    pybind11::buffer_info info = data.request();

    uint64_t dtype;
    if (info.format == pybind11::format_descriptor<uint8_t>::format()) {
        dtype = 8;
    }
	else if (info.format == pybind11::format_descriptor<uint16_t>::format()) {
        dtype = 16;
    }
	else if (info.format == pybind11::format_descriptor<float>::format()) {
        dtype = 32;
    }
	else if (info.format == pybind11::format_descriptor<double>::format()) {
        dtype = 64;
    }
	else {
        throw std::runtime_error("Unsupported data type");
    }

    // Get the dimensions of the array
    uint64_t dims[3] = {static_cast<uint64_t>(info.shape[0]), static_cast<uint64_t>(info.shape[1]), static_cast<uint64_t>(info.shape[2])};

    // Call the function to write the data to a TIFF file
    writeTiffParallelHelper(fileName.c_str(), info.ptr, dtype, "w", dims[0], dims[1], dims[2], 0, 0, compression);
}

PYBIND11_MODULE(cpptiff, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-tiff python bindings";

	m.def("pybind11_read_tiff", &pybind11_read_tiff, "Read a tiff file");

	m.def("pybind11_write_tiff", &pybind11_write_tiff, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("compression"), "Write a tiff file");
}
