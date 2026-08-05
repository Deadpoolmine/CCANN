cmake_minimum_required(VERSION 3.0)
project(ccann)

set(CMAKE_STANDARD 17)
option(USE_AIO "Use AIO instead of liburing" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
)

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
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops )
# add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fsanitize=address)
# add_compile_options(-g -O0 -march=native -mtune=native -ftree-vectorize -funroll-loops)
# add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -mno-avx512f -mno-avx512vl -mno-avx512bw -mno-avx512cd)

message("Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	# -DNDEBUG
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	# -DDIRECT_READ_CC # disable opt #1.1
	# -DBG_IO_THREAD # enable opt #1.2
	# -DDELTA_PRUNING # enable opt #2

	# -DNO_POLLUTE_ORIGINAL

	# optimizations.
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # starling: must disable this!
	-DOVERLAP_INIT
	-DDYN_PIPE_WIDTH
	
	# policy.
	# -DSTATIC_POLICY

	# unoptimizations.
	# -DNAIVE_PIPE

	-DUSE_BS_THREAD_POOL
)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
	${PROJECT_SOURCE_DIR}/third_party/pmdk/src/include
	${PROJECT_SOURCE_DIR}/third_party/thread-pool/include
	${PROJECT_SOURCE_DIR}/third_party/userspace-rcu/build/include
)
link_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src
	${PROJECT_SOURCE_DIR}/third_party/pmdk/src
)

if(USE_AIO)
	message(STATUS "Using AIO for asynchronous I/O, compatible but slower.")
	link_libraries(aio)
	add_definitions(-DUSE_AIO)
else()
	message(STATUS "Using liburing for asynchronous I/O, faster but requires kernel support.")
	link_libraries(uring)
endif()

message(STATUS "Link with PMDK")
link_libraries(
	${PROJECT_SOURCE_DIR}/third_party/pmdk/src/nondebug/libpmem.so
)

message(STATUS "Link with URCU")
link_libraries(
	${PROJECT_SOURCE_DIR}/third_party/userspace-rcu/build/lib/liburcu.so
)

link_libraries(
	jemalloc
	tbb
)
# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)
# add_compile_options(-fsanitize=undefined -lubsan)
# add_link_options(-fsanitize=undefined)

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
# test_x86_avx.cpp returns a bitmask: bit0=AVX2, bit1=AVX512F, bit2=AVX512VPOPCNTDQ
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
set(CPU_SUPPORTS_AVX512_VPOPCNTDQ FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
	try_run(
		RUN_RESULT
		COMPILE_RESULT
		"${CMAKE_BINARY_DIR}"
		SOURCES "${PROJECT_SOURCE_DIR}/tests/test_x86_avx.cpp"
		COMPILE_OUTPUT_VARIABLE COMPILE_OUT
		RUN_OUTPUT_VARIABLE RUN_OUT
	)
else()
	set(RUN_RESULT 0)
	set(COMPILE_RESULT FALSE)
endif()

# RUN_RESULT is a bitmask: bit0=AVX2, bit1=AVX512F, bit2=AVX512VPOPCNTDQ
if(COMPILE_RESULT)
	math(EXPR HAS_AVX2 "${RUN_RESULT} & 1")
	math(EXPR HAS_AVX512F "${RUN_RESULT} & 2")
	math(EXPR HAS_VPOPCNTDQ "${RUN_RESULT} & 4")
	if(HAS_AVX2)
		set(CPU_SUPPORTS_AVX2 TRUE)
		message(STATUS "CPU supports AVX2")
	endif()
	if(HAS_AVX512F)
		set(CPU_SUPPORTS_AVX512 TRUE)
		message(STATUS "CPU supports AVX512F")
	endif()
	if(HAS_VPOPCNTDQ)
		set(CPU_SUPPORTS_AVX512_VPOPCNTDQ TRUE)
		message(STATUS "CPU supports AVX512VPOPCNTDQ")
	endif()
else()
	message(STATUS "CPU feature detection failed")
endif()

if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512_VPOPCNTDQ)
	message(STATUS "Enabling AVX512 + VPOPCNTDQ.")
	add_definitions(-DUSE_AVX512)
	add_definitions(-DUSE_AVX512_VPOPCNTDQ)
elseif(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU support -mavx512f. Enabling AVX512.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU support -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
endif()

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

	
find_package(BLAS)
if(BLAS_FOUND)
    message(STATUS "BLAS found")
    link_libraries(${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS is required but was not found.")
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
# add_subdirectory(python)
