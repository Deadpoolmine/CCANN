set(CMAKE_CXX_STANDARD 17)

file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
