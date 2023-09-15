cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

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

#set(ZSTD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zstd-1.5.5")
#add_subdirectory(${ZSTD_DIR}/build/cmake)
#set_property(TARGET zstd PROPERTY POSITION_INDEPENDENT_CODE ON)

# Run the script to build ZSTD
execute_process(
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/dependecies/build_zstd.cmake
)

#set_property(TARGET libtiff PROPERTY POSITION_INDEPENDENT_CODE ON)
#add_dependencies(libtiff ZSTD::ZSTD)
# Create a custom target for libtiff to check for ZSTD
#add_custom_target(libtiff_check_zstd DEPENDS ZSTD::ZSTD)

# Add libtiff
include_directories(${ZSTD_DIR}/lib)
set(ZLIB_INCLUDE_DIR  ${ZLIB_DIR} CACHE STRING "" FORCE)
set(ZLIB_LIBRARY  ${CMAKE_BINARY_DIR}/dependencies/zlib-1.2.8/${ZLIB_STATIC_NAME} CACHE STRING "" FORCE)
set(Deflate_INCLUDE_DIR  ${LIBDEFLATE_DIR} CACHE STRING "" FORCE)
set(Deflate_LIBRARY  ${CMAKE_BINARY_DIR}/dependencies/libdeflate-1.18/libdeflate.a CACHE STRING "" FORCE)
set(ZSTD_INCLUDE_DIR  ${ZSTD_DIR}/lib CACHE STRING "" FORCE)
set(ZSTD_LIBRARY  ${CMAKE_BINARY_DIR}/dependencies/zstd-1.5.5/libzstd.a CACHE STRING "" FORCE)
set(LIBTIFF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/tiff-4.6.0")
add_subdirectory(${LIBTIFF_DIR})
set_property(TARGET libtiff PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_target_properties(libtiff PROPERTIES CMAKE_ORDER_DEPEND libtiff_check_zstd)
#add_dependencies(libtiff ZSTD::ZSTD)
# tiff
#set_property(TARGET libtiff PROPERTY POSITION_INDEPENDENT_CODE ON)

set(CBLOSC2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/c-blosc2-2.10.2")
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CBLOSC2_DIR})
set_property(TARGET blosc2_static PROPERTY POSITION_INDEPENDENT_CODE ON)

set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
set(ZLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zlib-1.2.8")
add_subdirectory(${ZLIB_DIR})
set_property(TARGET zlibstatic PROPERTY POSITION_INDEPENDENT_CODE ON)


set(CBLOSC_INCLUDE_DIR "${CBLOSC_DIR}/blosc")
set(CBLOSC2_INCLUDE_DIR "${CBLOSC2_DIR}/include")
set(JSON_INCLUDE_DIR "${JSON_DIR}/include")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(cppZarr_shared SHARED ${SRC_DIR}/zarr.cpp ${SRC_DIR}/helperfunctions.cpp ${SRC_DIR}/parallelreadzarr.cpp ${SRC_DIR}/parallelwritezarr.cpp)
set_target_properties(cppZarr_shared PROPERTIES OUTPUT_NAME cppZarr)
add_dependencies(cppZarr_shared blosc_static)
add_dependencies(cppZarr_shared blosc2_static)
add_dependencies(cppZarr_shared zlibstaticCPPZARR)
if(UNIX)
add_dependencies(cppZarr_shared uuid_external)
endif()

#add_library(cppZarrStatic STATIC ${SRC_DIR}/zarr.cpp ${SRC_DIR}/helperfunctions.cpp ${SRC_DIR}/parallelreadzarr.cpp ${SRC_DIR}/parallelwritezarr.cpp)
#set_property(TARGET cppZarr_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(cppZarr_shared PROPERTIES PUBLIC_HEADER "${SRC_DIR}/zarr.h;${SRC_DIR}/helperfunctions.h;${SRC_DIR}/parallelreadzarr.h;${SRC_DIR}/parallelwritezarr.h")

#message(NOTICE "${CBLOSC_INCLUDE_DIR}")
#message(NOTICE "${CBLOSC2_INCLUDE_DIR}")
#message(NOTICE "${JSON_INCLUDE_DIR}")
#message(NOTICE "${ZLIB_DIR}")

# Include the header files from blosc, blosc2, and json
target_include_directories(cppZarr_shared PRIVATE
    ${CBLOSC_INCLUDE_DIR}
    ${CBLOSC2_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
    ${ZLIB_DIR}
)


#target_include_directories(cppZarrStatic PRIVATE
#    ${CBLOSC_INCLUDE_DIR}
#    ${CBLOSC2_INCLUDE_DIR}
#    ${JSON_INCLUDE_DIR}
#    ${ZLIB_DIR}
#)

# Link against the blosc and blosc2 libraries
target_link_libraries(cppZarr_shared PRIVATE
    #"-Wl,--allow-multiple-definition"  
    ${CMAKE_BINARY_DIR}/dependencies/c-blosc2-2.10.2/blosc/libblosc2.a
    ${CMAKE_BINARY_DIR}/dependencies/c-blosc-1.21.5/blosc/libblosc.a
    ${CMAKE_BINARY_DIR}/dependencies/zlib-1.2.8/${ZLIB_STATIC_NAME}
    -static-libgcc
    -static-libstdc++
    OpenMP::OpenMP_CXX
)

install(TARGETS cppZarr_shared
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
