# Third-party notices

cpp-tiff is BSD 3-Clause (`LICENSE`). Its library, Python module, wheels and MATLAB folders contain, or are shipped
with, the works below. Each one's license is kept, as its authors distribute it, in the file named under `licenses/`.

Where things end up:

- **the library**: `libcppTiff` (`.so`, `.dylib`, `.dll`), in every wheel (`cpp_tiff.libs/` in the Linux wheels,
  `cpptiff/` in the macOS and Windows wheels) and in every MATLAB folder (`linux/`, `macArm/`, `windows/`, and the
  same folders in `allMatlab.tar.gz`);
- **the module**: the Python extension module `cpptiff/cpptiff.*`, in every wheel;
- **the mex files**: `getImageSizeMex`, `parallelReadTiff` and `parallelWriteTiff`, in every MATLAB folder.

| Component | Version | License (SPDX) | License file(s) |
|---|---|---|---|
| libtiff | 4.7.1 | libtiff (its `tif_lzw.c` also carries the University of California's notice, in the same file) | `licenses/libtiff-4.7.1/LICENSE.md` |
| zlib | 1.3.1 | Zlib | `licenses/zlib-1.3.1/LICENSE` |
| libdeflate | 1.25 | MIT | `licenses/libdeflate-1.25/COPYING` |
| Zstandard (zstd) | 1.5.7 | BSD-3-Clause (zstd is dual-licensed BSD-3-Clause OR GPL-2.0-only; cpp-tiff uses it under BSD-3-Clause) | `licenses/zstd-1.5.7/LICENSE` |
| pybind11 | the latest at build time (`pybind11>=2.10.0`) | BSD-3-Clause | `licenses/pybind11/LICENSE` |
| GCC runtime libraries (libstdc++, libgcc, libgomp) | see below | GPL-3.0-or-later WITH GCC-exception-3.1 | `licenses/gcc-runtime/COPYING3`, `licenses/gcc-runtime/COPYING.RUNTIME` |
| mingw-w64 winpthreads | mingw-w64 v10 | MIT AND BSD-3-Clause | `licenses/mingw-w64-winpthreads/COPYING` |
| mingw-w64 runtime | mingw-w64 v10 | ZPL-2.1, with parts under other permissive licenses (see the file) | `licenses/mingw-w64-runtime/COPYING.MinGW-w64-runtime.txt` |

## libtiff 4.7.1

Source: https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz

Compiled statically into the library (all platforms), so it is in every wheel and every MATLAB folder.

## zlib 1.3.1

Source: https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz

Compiled statically into the library (all platforms), so it is in every wheel and every MATLAB folder.

## libdeflate 1.25

Source: https://github.com/ebiggers/libdeflate/releases/download/v1.25/libdeflate-1.25.tar.gz

Compiled statically into the library (all platforms), so it is in every wheel and every MATLAB folder.

## Zstandard (zstd) 1.5.7

Source: https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz

Compiled statically into the library (all platforms), so it is in every wheel and every MATLAB folder. zstd is
offered under either BSD-3-Clause or GPL-2.0-only; cpp-tiff uses it under BSD-3-Clause, whose text is its `LICENSE`.

## pybind11

Source: https://pypi.org/project/pybind11/ (https://github.com/pybind/pybind11)

The build takes `pybind11>=2.10.0`, so the latest release at build time. Its support code is compiled into the
module, in every wheel; it is not in the MATLAB folders.

## GCC runtime libraries

libstdc++, libgcc and libgomp, under the GNU General Public License, version 3, with the GCC Runtime Library
Exception, version 3.1, which allows programs compiled with GCC and linked with these libraries to be distributed
under licenses of their own. GCC's sources are at https://gcc.gnu.org and https://ftp.gnu.org/gnu/gcc/.

- **Linux wheels**: the library and the module link libstdc++ and libgcc statically, from the GCC of the
  manylinux_2_28 build image (gcc-toolset-14). The wheels carry libgomp as `cpp_tiff.libs/libgomp-e985bcbb.so.1.0.0`,
  the image's system libgomp (AlmaLinux 8's GCC 8.5.0 package; its sources are AlmaLinux's `gcc` source package).
- **Linux MATLAB folder** (`linux/`): the library links libstdc++ and libgcc statically, from GCC 13.2.0 (the mex
  files load MATLAB's own libstdc++); the folder carries the same libgomp file as the Linux wheels, as
  `libgomp.so.1`.
- **macOS wheels and MATLAB folder**: Homebrew's gcc@13, GCC 13.3.0 (https://formulae.brew.sh/formula/gcc@13, built
  from https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/). The library, the module and the mex files link libstdc++ statically.
  The wheels carry `.dylibs/libgomp.1.dylib`; `macArm/` carries `libgomp.1.dylib`, `libstdc++.6.0.32.dylib` and
  `libgcc_s.1.1.0.dylib`.
- **Windows wheels and MATLAB folder**: MinGW-builds GCC 12.2.0 (x86_64, posix threads, seh, release
  12.2.0-rt_v10-rev2, https://github.com/niXman/mingw-builds-binaries; GCC's own sources are at
  https://ftp.gnu.org/gnu/gcc/gcc-12.2.0/). The library, the module and the mex files link parts of libstdc++ and
  libgcc statically. The wheels (`cpptiff/`) and `windows/` carry `libgcc_s_seh-1.dll` and `libgomp-1.dll`.

## mingw-w64 winpthreads

Source: https://sourceforge.net/projects/mingw-w64/files/mingw-w64/mingw-w64-release/ (mingw-w64 v10, as built by
MinGW-builds 12.2.0-rt_v10-rev2)

The POSIX threads library of the Windows toolchain, which GCC's runtime libraries use. The Windows wheels
(`cpptiff/`) and `windows/` carry it as `libwinpthread-1.dll`.

## mingw-w64 runtime

Source: https://sourceforge.net/projects/mingw-w64/files/mingw-w64/mingw-w64-release/ (mingw-w64 v10, as built by
MinGW-builds 12.2.0-rt_v10-rev2, which ships this notice as `licenses/mingw-w64/COPYING.MinGW-w64-runtime.txt`)

The C runtime startup code and import libraries of the Windows toolchain, linked statically into every Windows binary
(the library, the module and the mex files), so it is in the Windows wheels and `windows/`. Some of its parts require
their notices in binary distributions; the file lists them.
