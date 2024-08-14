apt-get -y install libopenmpi-dev libomp-dev

export HIPCC_BIN=/opt/rocm/bin
export MPI_INCLUDE=/usr/lib/openmpi/include

export OPT="-O3 -g -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO"
export OMP="-fopenmp"

export HIPCC_FLAGS="-v -ffast_math -DINLINE_FORCE -I${MPI_INCLUDE}"
export HIPCC_FLAGS="-v -I${MPI_INCLUDE} -I/opt/rocm/hip/include -I/opt/rocm/hcc-1.0/include"

export HACC_PLATFORM="hip"
export HACC_OBJDIR="${HACC_PLATFORM}"

export HACC_CFLAGS="$OPT $OMP $HIPCC_FLAGS"
export HACC_CC="${HIPCC_BIN}/hcc -x c -Xclang -std=c99"

export HACC_CXXFLAGS="$OPT $OMP $HIPCC_FLAGS"
export HACC_CXX="${HIPCC_BIN}/hipcc -Xclang"

export HACC_LDFLAGS="-lm -lrt"

# USE_SERIAL_COSMO must be set to avoid building the code with MPI, which isn't
# supported on the GPU model in gem5.
export USE_SERIAL_COSMO="1"
export HACC_NUM_CUDA_DEV="1"
export HACC_MPI_CFLAGS="$OPT $OMP $HIPCC_FLAGS"
export HACC_MPI_CC="${HIPCC_BIN}/hcc -x c -Xclang -std=c99 -Xclang -pthread"

export HACC_MPI_CXXFLAGS="$OPT $OMP $HIPCC_FLAGS"
export HACC_MPI_CXX="${HIPCC_BIN}/hipcc -Xclang -pthread"
export HACC_MPI_LD="${HIPCC_BIN}/hipcc -Xclang -pthread"

export HACC_MPI_LDFLAGS="-lm -lrt"
