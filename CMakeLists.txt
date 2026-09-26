cmake_minimum_required(VERSION 3.18)
project(ccann)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
option(CCANN_BUILD_PYTHON "Build the Python extension" OFF)
option(CCANN_TIMING "Collect internal timing statistics" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
	${PROJECT_SOURCE_DIR}/third_party/thread-pool/include
)

if(WIN32)
  include_directories(BEFORE ${PROJECT_SOURCE_DIR}/include/win_compat)
else()
add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
)
if(CCANN_BUILD_PYTHON)
  add_compile_options(-g -O3 -mtune=generic -ftree-vectorize -funroll-loops)
else()
  add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops)
endif()
endif()

if(CCANN_TIMING)
    add_definitions(-DANN_TIMING)
endif()

if(NOT WIN32)
include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
	${PROJECT_SOURCE_DIR}/third_party/thread-pool/include
	${PROJECT_SOURCE_DIR}/third_party/userspace-rcu/build/include
)
link_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src
)

link_libraries(uring)

message(STATUS "Link with URCU")
link_libraries(
	${PROJECT_SOURCE_DIR}/third_party/userspace-rcu/build/lib/liburcu.so
)

link_libraries(
	tbb
)
endif()

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
if(NOT WIN32)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
try_run(
	RUN_RESULT
	COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp"
	COMPILE_OUTPUT_VARIABLE COMPILE_OUT
	RUN_OUTPUT_VARIABLE RUN_OUT
)

if(COMPILE_RESULT AND RUN_RESULT EQUAL 3) # AVX512 + AVX2
	set(CPU_SUPPORTS_AVX512 TRUE)
	message(STATUS "CPU supports AVX512")
elseif(COMPILE_RESULT AND RUN_RESULT EQUAL 1) # AVX2
	set(CPU_SUPPORTS_AVX2 TRUE)
	message(STATUS "CPU supports AVX2")
else()
	message(STATUS "CPU does not support AVX512 or detection failed")
endif()
endif()

if(CCANN_BUILD_PYTHON)
    message(STATUS "Building portable Python extension without host-specific ISA")
elseif(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU supports -mavx512f. Enabling AVX512F.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU supports -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
endif()

#OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(NOT WIN32)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(NOT WIN32)
find_package(BLAS)
if(BLAS_FOUND)
    message(STATUS "BLAS found")
    link_libraries(${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS is required but was not found.")
endif()
endif()

add_subdirectory(src)
if(NOT WIN32)
  add_subdirectory(tests)
  add_subdirectory(tests/utils)
endif()
if(CCANN_BUILD_PYTHON)
  add_subdirectory(python)
endif()
