SOURCE_DIR?=.
BIN_DIR?=bin
MKDIR=mkdir -p
ROCM_PATH?=/opt/rocm

#hipcc
HIPCC=${ROCM_PATH}/hip/bin/hipcc

#BLAS
ROCBLAS_LIB=rocblas
ROCBLAS_INC=${ROCM_PATH}/rocblas/include

#CONV
MIOPEN_LIB?=MIOpen
MIOPEN_INC=${ROCM_PATH}/miopen/include

#GEMM
MIOPENGEMM_LIB=miopengemm

#DeepBench
DEEPBENCH_INC=${SOURCE_DIR}/../kernels

all: conv rnn gemm 

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-g

GEM5_PATH ?= ../../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(DEEPBENCH_INC) -I$(MIOPEN_INC) -I$(ROCBLAS_INC) -l$(MIOPEN_LIB) -l$(ROCBLAS_LIB) -l$(MIOPENGEMM_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx801

rnn:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_rocm.cpp -o $(BIN_DIR)/rnn_bench $(CFLAGS) -I$(DEEPBENCH_INC) -I$(MIOPEN_INC) -I$(ROCBLAS_INC) $(LDFLAGS) -l$(MIOPEN_LIB) -l$(ROCBLAS_LIB) -l$(MIOPENGEMM_LIB) $(OPT) -std=c++11 -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive --amdgpu-target=gfx801

rnn_sync:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_sync.cpp -o $(BIN_DIR)/rnn_bench_sync $(CFLAGS) -I$(DEEPBENCH_INC) -I$(MIOPEN_INC) -I$(ROCBLAS_INC) $(LDFLAGS) -l$(MIOPEN_LIB) -l$(ROCBLAS_LIB) -l$(MIOPENGEMM_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx801

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(DEEPBENCH_INC) -I$(ROCBLAS_INC) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx801

clean:
	rm -rf $(BIN_DIR)
