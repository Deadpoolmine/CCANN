# pip install "pybind11[global]"

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native pybind.cpp)
target_link_libraries(_native PRIVATE ccann)
set_target_properties(_native PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
