cmake_minimum_required(VERSION 3.9)

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
  VERSION 1.1.0
  LANGUAGES CXX)

# OpenMP is Required
find_package(OpenMP REQUIRED)

if(UNIX)
set(ZLIB_STATIC_NAME "libz.a")
else()
set(ZLIB_STATIC_NAME "libzlib.a")
endif()

set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
set(ZLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zlib-1.2.8")
add_subdirectory(${ZLIB_DIR})
set_property(TARGET zlibstatic PROPERTY POSITION_INDEPENDENT_CODE ON)

set(LIBDEFLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libdeflate-1.18")
add_subdirectory(${LIBDEFLATE_DIR})
set_property(TARGET libdeflate_static PROPERTY POSITION_INDEPENDENT_CODE ON)

set(ZSTD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zstd-1.5.5")
add_subdirectory(${ZSTD_DIR}/build/cmake)
set_property(TARGET zstd PROPERTY POSITION_INDEPENDENT_CODE ON)

# Add libtiff
include_directories(${ZSTD_DIR}/lib)
set(ZLIB_INCLUDE_DIR  ${ZLIB_DIR} CACHE STRING "" FORCE)
set(ZLIB_LIBRARY  ${CMAKE_BINARY_DIR}/dependencies/zlib-1.2.8/${ZLIB_STATIC_NAME} CACHE STRING "" FORCE)
set(Deflate_INCLUDE_DIR  ${LIBDEFLATE_DIR} CACHE STRING "" FORCE)
set(Deflate_LIBRARY  ${CMAKE_BINARY_DIR}/dependencies/libdeflate-1.18/libdeflate.a CACHE STRING "" FORCE)
set(ZSTD_INCLUDE_DIR  ${ZSTD_DIR}/lib CACHE STRING "" FORCE)
set(ZSTD_LIBRARY  ${CMAKE_BINARY_DIR}/dependencies/zstd-1.5.5/build/cmake/lib/libzstd.a CACHE STRING "" FORCE)
set(LIBTIFF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/tiff-4.6.0")
set(tiff-tools OFF CACHE BOOL "" FORCE)
set(tiff-tests OFF CACHE BOOL "" FORCE)
set(tiff-contrib OFF CACHE BOOL "" FORCE)
add_subdirectory(${LIBTIFF_DIR})
set_property(TARGET tiff PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(tiff zlibstatic)
add_dependencies(tiff libdeflate_static)
add_dependencies(tiff zstd)


set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(cppTiff_shared SHARED ${SRC_DIR}/helperfunctions.cpp  ${SRC_DIR}/lzwencode.cpp ${SRC_DIR}/parallelreadtiff.cpp ${SRC_DIR}/parallelwritetiff.cpp)
set_target_properties(cppTiff_shared PROPERTIES OUTPUT_NAME cppTiff)
add_dependencies(cppTiff_shared zlibstatic)
add_dependencies(cppTiff_shared libdeflate_static)
add_dependencies(cppTiff_shared zstd)
add_dependencies(cppTiff_shared tiff)

set_target_properties(cppTiff_shared PROPERTIES PUBLIC_HEADER "${SRC_DIR}/helperfunctions.h;${SRC_DIR}/lzwencode.h;${SRC_DIR}/parallelreadtiff.h;${SRC_DIR}/parallelwritetiff.h")

# Include the header files from blosc, blosc2, and json
target_include_directories(cppTiff_shared PRIVATE
    ${ZLIB_DIR}
	${LIBDEFLATE_DIR} 
	${ZSTD_DIR}/lib   
	${LIBTIFF_DIR}/libtiff
)

# Link against the blosc and blosc2 libraries
target_link_libraries(cppTiff_shared PRIVATE
    ${CMAKE_BINARY_DIR}/dependencies/zlib-1.2.8/${ZLIB_STATIC_NAME}
    ${CMAKE_BINARY_DIR}/dependencies/libdeflate-1.18/libdeflate.a
    ${CMAKE_BINARY_DIR}/dependencies/zstd-1.5.5/build/cmake/lib/libzstd.a
    ${CMAKE_BINARY_DIR}/dependencies/tiff-4.6.0/libtiff/libtiff.a
    -static-libgcc
    -static-libstdc++
    OpenMP::OpenMP_CXX
)

install(TARGETS cppTiff_shared
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
