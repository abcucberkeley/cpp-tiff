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

#### With GPU reads
Reading straight into GPU memory needs Linux, an NVIDIA GPU (Volta or newer) and its driver. The gpu extra also installs nvCOMP, which decompresses on the GPU and which cpptiff finds by itself (see "Read straight into GPU memory" below)
````
pip install "cpp-tiff[gpu]"
# gpu=True returns CuPy arrays, so it also needs CuPy (reading into PyTorch tensors does not)
pip install cupy-cuda12x
````

### Usage

The reader returns a zyx numpy array for the given tiff file (zyxc for RGB/RGBA tiffs, with the color channel last, whether the file stores the colors interleaved or as separate planes)

The writer takes an output filename and a zyx numpy array, or a zyxc array for RGB/RGBA (3 or 4 colors last). A single yxc color image needs rgb=True, since a 3D array is otherwise written as a stack of grayscale images

By default, the writer uses lzw compression but you can also pass 'zstd' or 'none' (in any case) as the third argument. Passing a tile size (a multiple of 16) writes square tiles instead of strips

File names can be strings or path objects such as pathlib.Path. On Linux and macOS a name that starts with ~ or ~name is in that home folder, as a shell expands it; nothing else in a name is expanded

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
The range is [start, end): 0-based, with the end excluded, as a list, tuple, range or NumPy array
````
import cpptiff
# Read the first two slices
im = cpptiff.read_tiff('filename.tif', [0, 2])
````

#### Read and write MATLAB's (y, x, z) layout
order='F' reads into the column-major (y, x, z) array MATLAB's parallelReadTiff returns ((y, x, c, z) for RGB/RGBA), filled in that layout as the file is read, with no copy. write_tiff with order='F' takes that layout, and writes the same file as for the (z, y, x) array, without a copy when the array is Fortran-ordered
````
import cpptiff
im = cpptiff.read_tiff('filename.tif', order='F')   # (y, x, z), Fortran-ordered
cpptiff.write_tiff('outputFilename.tif', im, order='F')
````

#### Get the shape of a tiff file without reading the image
The shape is (z, y, x) or (z, y, x, c), or with order='F' (y, x, z) or (y, x, c, z)
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
By default every call uses all CPUs (or OMP_NUM_THREADS). num_threads sets the threads of one call, and since reads and writes release the GIL, calls from several Python threads run at once. out reads into an existing array (of the shape read_tiff returns and the file's data type, C-ordered, or Fortran-ordered with order='F') instead of a new one
````
import numpy as np
import cpptiff
im = cpptiff.read_tiff('filename.tif', num_threads=4)
cpptiff.read_tiff('filename.tif', out=im)
cpptiff.write_tiff('outputFilename.tif', im, num_threads=4)
````

#### Read straight into GPU memory
For data needed on the GPU anyway, the strips' stored bytes are read, copied to the GPU and decompressed there: zstd and deflate with nvCOMP, LZW and the other codecs on the CPU (nvCOMP has no LZW), into the same copy. The GPU then undoes the predictor, swaps the bytes of big-endian files and puts the slices in place, in either order. The values are the same as read_tiff's. Linux only for now
````
import cpptiff
# A CuPy array instead of a NumPy array (needs CuPy: pip install cupy-cuda12x)
im = cpptiff.read_tiff('filename.tif', gpu=True)
im = cpptiff.read_tiff('filename.tif', [0, 2], order='F', gpu=True)

# Or read into GPU memory you have: a CuPy array, a PyTorch CUDA tensor, or anything with __cuda_array_interface__
import torch
out = torch.empty(cpptiff.get_image_shape('filename.tif'), dtype=torch.uint16, device='cuda')
cpptiff.read_tiff('filename.tif', out=out, stream=torch.cuda.current_stream())

# A batch goes to the GPU together (one decompression for all of its strips, which is what GPUs do best)
ims = cpptiff.read_tiff_batch(['a.tif', ('b.tif', [0, 10])], gpu=True)

# Whether GPU reads can run here, and if not, why
ok, why = cpptiff.gpu_available()
````
CPPTIFF_GPU_PROFILE=1 prints where each GPU read's time went; CPPTIFF_GPU_PART_MAX_MB (256) sets the most a read sends to the GPU at once (a bigger read goes in parts, the next one read while the GPU works on the one before), and CPPTIFF_GPU_TIMEOUT the seconds a read waits for the GPU (nvCOMP can hang on damaged data; by default 30 plus 1 per 50 MB)

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

### Reading into GPU memory
gpureadtiff.h reads files straight into GPU memory, decompressing on the GPU (see the Python section; Linux only for now). Building needs nothing more: the library loads the NVIDIA driver and nvCOMP (5 or later: pip install nvidia-libnvcomp-cu12, or set CPPTIFF_NVCOMP_LIBRARY to its libnvcomp file) when a GPU read first needs them, and builds and runs without either

## MATLAB

### Prerequisites
1. None! The parallel reader and writer mex files will work with the most recent version of Matlab.

### Download and Install
1. Download the latest release for your OS from here: https://github.com/abcucberkeley/cpp-tiff/releases
2. Unzip the folder
3. You can now put the mex files wherever you'd like and add them to your path if needed
4. Note for Mac Users: You may need to restart Matlab before using the Mex files if you have an open session

### Usage

On Linux and macOS a path that starts with ~ or ~name is in that home folder, as a shell expands it; nothing else in a path is expanded: spaces, parentheses, $ and a ~ further on are kept as they are

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

#### Threads
On Linux the mex files run on MATLAB's own threads, so maxNumCompThreads sets how many they use and OMP_NUM_THREADS has no effect. On Mac and Windows it is the other way around: they use every CPU unless OMP_NUM_THREADS is set before MATLAB starts, and maxNumCompThreads has no effect
````
% Linux: read with 8 threads
maxNumCompThreads(8);
im = parallelReadTiff('path/to/file.tif');
````

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02475-4
