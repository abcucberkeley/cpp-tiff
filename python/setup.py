import os
import platform
from setuptools import setup, find_packages, Extension
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sysconfig
import pybind11
import subprocess
import shutil

# Determine OS and shared libraries
system = platform.system()
shared_libraries = []
if system == "Linux":
    shared_libraries.append("*.so*")
elif system == "Windows":
    shared_libraries.append("*.dll")
elif system == "Darwin":
    shared_libraries.append("*.dylib")


class BuildExt(build_ext):
    def run(self):
        # Get the directory of the current file (setup.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define paths relative to the current directory
        cmake_source_dir = os.path.join(current_dir)
        cmake_build_dir = os.path.join(current_dir, "build")
        install_dir = os.path.join(cmake_build_dir, "install")

        # Build and install CMake-based dependency
        os.makedirs(cmake_build_dir, exist_ok=True)

        # Run cmake configuration and build commands
        cmake_cmd = [
            "cmake",
            cmake_source_dir,
            "-B", cmake_build_dir,
            f"-DCMAKE_INSTALL_PREFIX={install_dir}"
        ]

        # Configure and build CMake project
        try:
            subprocess.check_call(cmake_cmd)
            subprocess.check_call(["cmake", "--build", cmake_build_dir, "--parallel", "--target", "install"])
        except subprocess.CalledProcessError as e:
            print("Error during CMake build and install:", e)
            raise

        # Run the regular build_ext to compile the Python extension
        super().run()


ext_modules = [
    Extension(
        "cpptiff.cpptiff",
        include_dirs=['build/install/include',pybind11.get_include()],
        library_dirs=['build/install/lib64','build/install/lib'],
        libraries=['cppTiff'],
        sources=["cpptiff.cpp"],
        # The module's static C++ runtime stays private to it. On macOS the module exports its entry point alone:
        # the loader merges the exported weak definitions of one name across every image a process loads, so that
        # next to another GCC-built Python module (cpp-zarr's 1.7.3 wheel), whichever of the two was loaded second
        # ran the other's copy of pybind11's std::call_once code, on the other's runtime, and its first write crashed
        extra_link_args=['-static-libstdc++'] + (['-Wl,--exclude-libs,ALL'] if system == "Linux" else
                                                 ['-Wl,-exported_symbol,_PyInit_cpptiff'] if system == "Darwin" else [])
    ),
]


# Packaging for PyPI
setup(
    name="cpp-tiff",
    version="1.7.3",
    description="Python wrappers for cpp-tiff",
    url='https://github.com/abcucberkeley/cpp-tiff',
    author='Matthew Mueller',
    author_email='matthewmueller@berkeley.edu',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt} if system=="Linux" else {"build_ext": build_ext},
    install_requires=['numpy>=1.20'],
    # (reads into GPU memory: nvCOMP, which the library loads at run time; macOS has no NVIDIA GPUs)
    extras_require={'gpu': ['nvidia-libnvcomp-cu12; platform_system != "Darwin"']},
    package_dir={'cpp-tiff': 'cpp-tiff'},
    package_data={"": shared_libraries} if system!="Linux" else {},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9, <3.15',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
