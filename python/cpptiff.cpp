#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "parallelreadtiff.h"
#include "parallelwritetiff.h"
#include "helperfunctions.h"

template <typename T>
pybind11::array_t<T> create_pybind11_array(void* data, const uint64_t* dims) {
    auto deleter = [](void* ptr) { free(ptr); };

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

pybind11::array pybind11_read_tiff(const std::string& fileName){
	uint64_t* dims = getImageSize(fileName.c_str());
	uint64_t dtype = getDataType(fileName.c_str());

	void* data = readTiffParallelWrapperNoXYFlip(fileName.c_str());

	switch (dtype) {
        case 8:  // 8-bit unsigned int
            return create_pybind11_array<uint8_t>(data, dims);
		case 16: // 16-bit unsigned int
            return create_pybind11_array<uint16_t>(data, dims);
        case 32: // 32-bit float
            return create_pybind11_array<float>(data, dims);
        case 64: // 64-bit double
            return create_pybind11_array<double>(data, dims);
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
