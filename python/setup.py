import os
import platform
from setuptools import setup, find_packages, Extension
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sysconfig
import pybind11

# Determine OS and shared libraries
system = platform.system()
shared_libraries = []
if system == "Linux":
    shared_libraries.append("*.so*")
elif system == "Windows":
    shared_libraries.append("*.dll")
elif system == "Darwin":
    shared_libraries.append("*.dylib")

ext_modules = [
    Extension(
        "cpptiff.cpptiff",
        include_dirs=['build/install/include',pybind11.get_include()],
        library_dirs=['build/install/lib64','build/install/lib'],
        libraries=['cppTiff'],
        sources=["cpptiff.cpp"],
	extra_link_args=['-static-libstdc++']
    ),
]


# Packaging for PyPI
setup(
    name="cpp-tiff",
    version="1.0.0",
    description="Python bindings for cpp-tiff",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=['numpy'],
    package_dir={'cpp-tiff': 'cpp-tiff'},
    package_data={"": shared_libraries},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
