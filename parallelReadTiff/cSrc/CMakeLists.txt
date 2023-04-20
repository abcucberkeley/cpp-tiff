cmake_minimum_required (VERSION 2.8.12)

# Setup cmake policies.
foreach(p
    CMP0012
    CMP0013
    CMP0014
    CMP0022 # CMake 2.8.12
    CMP0025 # CMake 3.0
    CMP0053 # CMake 3.1
    CMP0054 # CMake 3.1
    CMP0074 # CMake 3.12
    CMP0075 # CMake 3.12
    CMP0083 # CMake 3.14
    CMP0093 # CMake 3.15
  )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

project(OMP_TIFF C)

# Find OpenMP
option(USE_SYSTEM_OPENMP "Use system-installed OpenMP." ON)
if(USE_SYSTEM_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
      add_definitions(-DENABLE_OPENMP=1)
      set(ENABLE_OPENMP 1)
      set(OPENMP_LIBRARIES "${OpenMP_C_LIBRARIES}")
  else()
      message(FATAL_ERROR "OpenMP not found")
  endif()
endif()

# Find LibTIFF
option(USE_LIB_TIFF "Enable LibTiff." ON)
if(USE_LIB_TIFF)
  find_package(TIFF REQUIRED)
  if(TIFF_FOUND)
    # Add the LibTIFF include directory to the include path
    include_directories(${TIFF_INCLUDE_DIRS})
    add_library(omp_tiff parallelReadTiff.c)
    target_compile_options(omp_tiff PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(omp_tiff PUBLIC ${OpenMP_C_LIBRARIES})
    target_link_libraries(omp_tiff PUBLIC ${TIFF_LIBRARIES})
    target_include_directories(omp_tiff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    
    add_executable(main main.c)
    target_link_libraries(main ${TIFF_LIBRARIES})
    target_link_libraries(main omp_tiff)
  else()
    message(WARNING "LibTiff not found, ignore building the executables which requires LibTiff support.")
  endif()
endif()