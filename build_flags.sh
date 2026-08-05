export ADDITIONAL_DEFINITIONS=$1

mkdir -p build
cd build
cmake ..
make -j56
