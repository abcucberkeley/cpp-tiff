# cpp-tiff
An efficient parallel Tiff reader/writer that utilizes LibTIFF and OpenMP.

## Limitations
1. Currently RGB tiffs are not supported but support may be added in the future

## Python

A Python version of cpp-tiff is available through pip

### Prerequisites

#### Python
Python version >=3.8

#### OS
Linux: All Linux distros made within the past 10 years should work

Mac Apple Silicon (M1, M2, etc.): macOS 13 or newer is required

Mac Intel: macOS 12 or newer is required

Windows: Windows 10 or newer is required

### Installation
````
pip install cpp-tiff
````

### Usage

The reader returns a zyx numpy array for the given tiff file

The writer takes an output filename and a zyx numpy array

By default, the writer uses lzw compression but you can also pass 'none' as the third argument

#### Read and Write a tiff file
````
import cpptiff
im = cpptiff.read_tiff('filename.tif')
# Do some processing here
cpptiff.write_tiff('outputFilename.tif', im)
````

#### Read a single slice of a tiff file
````
import cpptiff
# Read the first slice
im = cpptiff.read_tiff('filename.tif', 0)
````

#### Read a specified slice range of a tiff file
````
import cpptiff
# Read the first two slices
im = cpptiff.read_tiff('filename.tif', [0, 1])
````

#### Get the shape of a tiff file without reading the image
````
import cpptiff
im_shape = cpptiff.get_image_shape('filename.tif')
````

## CMake

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

## MATLAB

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

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02475-4
