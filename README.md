# cpp-tiff
An efficient parallel Tiff reader/writer that utilizes LibTIFF and OpenMP.

## Python

A Python version of cpp-tiff is available through pip

### Prerequisites

#### Python
1. Python version 3.9-3.14

#### OS
Linux: All Linux distros made within the past 10 years should work

Mac Apple Silicon (M1, M2, etc.): macOS 13 or newer is required

Mac Intel: Support has been discontinued as of v1.5.0 (macOS 12 or newer is required)

Windows: Windows 10 or newer is required

### Installation
````
pip install cpp-tiff
````

### Usage

The reader returns a zyx numpy array for the given tiff file (zyxc for RGB/RGBA tiffs, with the color channel last, whether the file stores the colors interleaved or as separate planes)

The writer takes an output filename and a zyx numpy array, or a zyxc array for RGB/RGBA (3 or 4 colors last). A single yxc color image needs rgb=True, since a 3D array is otherwise written as a stack of grayscale images

By default, the writer uses lzw compression but you can also pass 'zstd' or 'none' as the third argument. Passing a tile size (a multiple of 16) writes square tiles instead of strips

#### Read and Write a tiff file
````
import cpptiff
im = cpptiff.read_tiff('filename.tif')
# Do some processing here
cpptiff.write_tiff('outputFilename.tif', im)
````

#### Write with zstd compression, or in tiles
````
import cpptiff
cpptiff.write_tiff('outputFilename.tif', im, 'zstd')
cpptiff.write_tiff('tiledFilename.tif', im, tile=256)
````

#### Write an RGB or RGBA tiff file
````
import numpy as np
import cpptiff
# A stack of RGB images is zyxc with 3 colors (4 for RGBA)
im = np.zeros((10, 512, 512, 3), dtype=np.uint8)
cpptiff.write_tiff('rgbStack.tif', im)
# A single RGB image is yxc
cpptiff.write_tiff('rgbImage.tif', im[0], rgb=True)
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
im = cpptiff.read_tiff('filename.tif', [0, 2])
````

#### Get the shape of a tiff file without reading the image
````
import cpptiff
im_shape = cpptiff.get_image_shape('filename.tif')
````

#### Read many files at once
read_tiff_batch reads a list of files (or slice ranges of them) with the reads overlapping, which is much faster than reading them one after another when each file is small. It returns the arrays in the order of the requests
````
import cpptiff
ims = cpptiff.read_tiff_batch(['a.tif', ('b.tif', [0, 2])])
````

#### Threads and reading into an existing array
By default every call uses all CPUs (or OMP_NUM_THREADS). num_threads sets the threads of one call, and since reads and writes release the GIL, calls from several Python threads run at once. out reads into an existing array (of the shape read_tiff returns and the file's data type) instead of a new one
````
import numpy as np
import cpptiff
im = cpptiff.read_tiff('filename.tif', num_threads=4)
cpptiff.read_tiff('filename.tif', out=im)
cpptiff.write_tiff('outputFilename.tif', im, num_threads=4)
````

## CMake

The C++ library can be compiled using the CMakeLists.txt file

### Prerequisites
1. CMake downloads the dependencies (LibTIFF, zlib, libdeflate and zstd) when it configures the build, so the first configure needs internet access
2. Currently the only officially supported compiler is gcc on Linux and Mac and MinGW on Windows but others may work

### Download and Install
````
git clone https://github.com/abcucberkeley/cpp-tiff
cd cpp-tiff
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
% [y x z], or [y x z c] for RGB/RGBA
size = getImageSizeMex('path/to/file.tif');
````

#### parallelReadTiff - Read a Tiff image into an array
````
% (y, x, z), or (y, x, c, z) for RGB/RGBA
im = parallelReadTiff('path/to/file.tif');
% Read slices 1 through 10
im = parallelReadTiff('path/to/file.tif',[1 10]);
````

#### parallelWriteTiff - Write an array out as a Tiff image
````
im = rand(100,100,100);
parallelWriteTiff('path/to/file.tif',im);
````

The optional third to fifth arguments are the mode ('w' writes a new file, 'a' appends slices to an existing one), the compression ('lzw' by default, 'zstd' or 'none') and the tile size (0 by default for strips, or a multiple of 16 for square tiles)
````
parallelWriteTiff('path/to/file.tif',im,'w','zstd');
parallelWriteTiff('path/to/tiled.tif',im,'w','lzw',256);
````

RGB/RGBA data is (y, x, c, z) with 3 or 4 colors. A single (y, x, c) color image needs the sixth argument set to true, since a 3D array is otherwise written as a stack of grayscale images
````
rgbStack = zeros(512,512,3,10,'uint8');
parallelWriteTiff('path/to/rgbStack.tif',rgbStack);
rgbImage = zeros(512,512,3,'uint8');
parallelWriteTiff('path/to/rgbImage.tif',rgbImage,'w','lzw',0,true);
````

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02475-4
