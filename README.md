# cpp-tiff
An efficient parallel Tiff reader/writer that utilizes LibTIFF and OpenMP.

## Quick Start Guide (MATLAB)

### Prerequisites
1. None! The parallel reader and writer mex files will work with the most recent version of Matlab.

### Download and Install
1. Download the latest release for your OS from here: https://github.com/abcucberkeley/cpp-tiff/releases
2. Unzip the folder
3. You can now put the mex files wherever you'd like and add them to your path if needed
4. Note for Mac Users: You may need to restart Matlab before using the Mex files if you have an open session

### Usage

#### getImageSizeMex - Get the dimensions of a Tiff image
````
size = getImageSizeMex('path/to/file.tif');
````

#### parallelReadTiff - Read a Tiff image into an array
````
im = parallelReadTiff('path/to/file.tif');
````

#### parallelWriteTiff - Write an array out as a Tiff image
````
im = rand(100,100,100);
parallelWriteTiff('path/to/file.tif',im);
````

## Compiling with CMake

The C++ library can be compiled using the CMakeLists.txt file

### Prerequisites
1. Dependencies are included in the dependencies folder
2. Currently the only officially supported compiler is gcc on Linux and Mac and MinGW on Windows but others may work

### Download and Install
````
git clone https://github.com/abcucberkeley/cpp-zarr
cd cpp-zarr
mkdir build
cd build
cmake ..
make -j
make install
````
