cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

project(
  cpp-tiff
  VERSION 1.5.0
  LANGUAGES CXX)

# Set the RPATH to $ORIGIN when installing if applicable
set(CMAKE_INSTALL_RPATH $ORIGIN)

# OpenMP is Required
find_package(OpenMP REQUIRED)

include(FetchContent)

# zlib's static target builds libz.a on Unix; MinGW/MSYS uses a different name.
if(UNIX)
set(ZLIB_STATIC_NAME "libz.a")
else()
set(ZLIB_STATIC_NAME "libzlibstatic.a")
endif()

set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)

# zlib: fetched at configure time instead of vendored. Building it in-tree renames
# the source zconf.h -> zconf.h.included on every configure, dirtying the repo.
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zlib
    URL      https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
)
FetchContent_MakeAvailable(zlib)
set_property(TARGET zlibstatic PROPERTY POSITION_INDEPENDENT_CODE ON)

# libdeflate: fetched too. We only need the static library for libtiff -- not its
# gzip CLI (whose install rule fails) or its shared lib.
set(LIBDEFLATE_BUILD_GZIP OFF CACHE BOOL "" FORCE)
set(LIBDEFLATE_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(libdeflate
    URL      https://github.com/ebiggers/libdeflate/releases/download/v1.25/libdeflate-1.25.tar.gz
    URL_HASH SHA256=fed5cd22f00f30cc4c2e5329f94e2b8a901df9fa45ee255cb70e2b0b42344477
)
FetchContent_MakeAvailable(libdeflate)
set_property(TARGET libdeflate_static PROPERTY POSITION_INDEPENDENT_CODE ON)

# zstd: fetched at configure time instead of vendored. Its CMake lives under build/cmake.
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
FetchContent_Declare(zstd
    URL      https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
    URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
    SOURCE_SUBDIR build/cmake
)
FetchContent_MakeAvailable(zstd)
set(ZSTD_DIR "${zstd_SOURCE_DIR}")
set_property(TARGET libzstd_static PROPERTY POSITION_INDEPENDENT_CODE ON)
# libtiff's ZSTDCodec.cmake enables zstd when the target zstd::libzstd_static exists,
# and ALIASes it to ZSTD::ZSTD. Otherwise it runs a configure-time link test against
# libzstd.a, which isn't built yet in a clean build. Provide it as an imported target
# (a plain alias can't be re-aliased) pointing at the static lib FetchContent builds.
if(NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static STATIC IMPORTED GLOBAL)
    set_target_properties(zstd::libzstd_static PROPERTIES
        IMPORTED_LOCATION "${zstd_BINARY_DIR}/lib/libzstd.a"
        INTERFACE_INCLUDE_DIRECTORIES "${zstd_SOURCE_DIR}/lib")
endif()

# Add libtiff (fetched at configure time; finds its deps via the cache vars below)
include_directories(${ZSTD_DIR}/lib)
# zconf.h is generated into the binary dir, so list both dirs for libtiff's FindZLIB.
set(ZLIB_INCLUDE_DIR  "${zlib_SOURCE_DIR};${zlib_BINARY_DIR}" CACHE STRING "" FORCE)
set(ZLIB_LIBRARY  ${zlib_BINARY_DIR}/${ZLIB_STATIC_NAME} CACHE STRING "" FORCE)
set(Deflate_INCLUDE_DIR  ${libdeflate_SOURCE_DIR} CACHE STRING "" FORCE)
set(Deflate_LIBRARY  ${libdeflate_BINARY_DIR}/libdeflate.a CACHE STRING "" FORCE)
set(ZSTD_INCLUDE_DIR  ${ZSTD_DIR}/lib CACHE STRING "" FORCE)
set(ZSTD_LIBRARY  ${zstd_BINARY_DIR}/lib/libzstd.a CACHE STRING "" FORCE)
set(tiff-tools OFF CACHE BOOL "" FORCE)
set(tiff-tests OFF CACHE BOOL "" FORCE)
set(tiff-contrib OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Disable jpeg and lzma until I can add the dependencies
set(jpeg OFF CACHE BOOL "" FORCE)
set(lzma OFF CACHE BOOL "" FORCE)
set(jbig OFF CACHE BOOL "" FORCE)
set(webp OFF CACHE BOOL "" FORCE)
set(lerc OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tiff
    URL      https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz
    URL_HASH SHA256=f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69
)
FetchContent_MakeAvailable(tiff)
set(LIBTIFF_DIR "${tiff_SOURCE_DIR}")
include_directories(${tiff_BINARY_DIR}/libtiff)
set_property(TARGET tiff PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(tiff zlibstatic)
add_dependencies(tiff libdeflate_static)
add_dependencies(tiff libzstd_static)


set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(cppTiff_shared SHARED ${SRC_DIR}/helperfunctions.cpp  ${SRC_DIR}/lzwencode.cpp ${SRC_DIR}/parallelreadtiff.cpp ${SRC_DIR}/parallelwritetiff.cpp)
set_target_properties(cppTiff_shared PROPERTIES OUTPUT_NAME cppTiff)
add_dependencies(cppTiff_shared zlibstatic)
add_dependencies(cppTiff_shared libdeflate_static)
add_dependencies(cppTiff_shared libzstd_static)
add_dependencies(cppTiff_shared tiff)

file(GLOB LIBTIFF_HEADERS "${LIBTIFF_DIR}/libtiff/*.h")

set_target_properties(cppTiff_shared PROPERTIES PUBLIC_HEADER "${LIBTIFF_HEADERS};${tiff_BINARY_DIR}/libtiff/tiffconf.h;${tiff_BINARY_DIR}/libtiff/tiffvers.h;${ZSTD_DIR}/lib/zstd.h;${SRC_DIR}/helperfunctions.h;${SRC_DIR}/lzwencode.h;${SRC_DIR}/parallelreadtiff.h;${SRC_DIR}/parallelwritetiff.h")

# Include the header files
target_include_directories(cppTiff_shared PRIVATE
    ${zlib_SOURCE_DIR}
	${zlib_BINARY_DIR}
	${libdeflate_SOURCE_DIR}
	${ZSTD_DIR}/lib   
	${LIBTIFF_DIR}/libtiff
)

# Link against the libraries
target_link_libraries(cppTiff_shared PRIVATE
    ${tiff_BINARY_DIR}/libtiff/libtiff.a
    ${zlib_BINARY_DIR}/${ZLIB_STATIC_NAME}
    ${libdeflate_BINARY_DIR}/libdeflate.a
    ${zstd_BINARY_DIR}/lib/libzstd.a
    -static-libgcc
    -static-libstdc++
    OpenMP::OpenMP_CXX
)

install(TARGETS cppTiff_shared
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Optional benchmark executables for comparing reader/writer versions (not installed).
# Enable with -DBUILD_BENCH=ON. See benchmarks/run_bench.sh.
option(BUILD_BENCH "Build the read/write benchmark executables" OFF)
if(BUILD_BENCH)
    add_executable(benchRead ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchread.cpp)
    set_target_properties(benchRead PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(benchRead PRIVATE ${SRC_DIR})
    target_link_libraries(benchRead PRIVATE cppTiff_shared OpenMP::OpenMP_CXX)

    add_executable(benchWrite ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchwrite.cpp)
    set_target_properties(benchWrite PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(benchWrite PRIVATE ${SRC_DIR})
    target_link_libraries(benchWrite PRIVATE cppTiff_shared OpenMP::OpenMP_CXX)
endif()

# Round-trip smoke tests: write small random data, read it back, verify.
# Enable with -DBUILD_TESTS=ON, then run `ctest` in the build dir.
option(BUILD_TESTS "Build and register the round-trip smoke tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_executable(roundtripTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.cpp)
    set_target_properties(roundtripTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(roundtripTest PRIVATE ${SRC_DIR})
    target_link_libraries(roundtripTest PRIVATE cppTiff_shared OpenMP::OpenMP_CXX)
    # Match libcppTiff's runtime linkage so std::string/exceptions cross the DLL
    # boundary with one ABI (avoids a MinGW libstdc++ mismatch on Windows).
    target_link_options(roundtripTest PRIVATE -static-libgcc -static-libstdc++)
    add_test(NAME roundtrip COMMAND roundtripTest ${CMAKE_CURRENT_BINARY_DIR})
endif()
