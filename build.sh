# -DBG_IO_THREAD Do not use background I/O
export ADDITIONAL_DEFINITIONS="-DBATCH_PRUNING"

mkdir -p build
cd build
cmake ..
make -j56