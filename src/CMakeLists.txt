set(CMAKE_CXX_STANDARD 17)

if(WIN32)
  file(GLOB PROJECT_CPP *.cpp update/*.cpp utils/*.cpp)
  list(REMOVE_ITEM PROJECT_CPP
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/linux_aligned_file_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/windows_aligned_file_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/windows_cblas.cpp
  )
  list(APPEND PROJECT_CPP
    search/beam_search.cpp
    utils/windows_aligned_file_io.cpp
    utils/windows_cblas.cpp
  )
else()
  file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp)
  list(REMOVE_ITEM PROJECT_CPP
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/windows_aligned_file_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/windows_cblas.cpp
  )
endif()
add_library(${PROJECT_NAME} ${PROJECT_CPP})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WIN32)
  target_compile_options(${PROJECT_NAME} PRIVATE
    "/FI${PROJECT_SOURCE_DIR}/include/platform_compat.h")
endif()
