GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
matmul_m: matmul_m.hip.cpp 
	hipcc -o matmul_m.out matmul_m.hip.cpp --amdgpu-target=gfx801

matmul: matmul.hip.cpp
	hipcc -o matmul.out matmul.hip.cpp --amdgpu-target=gfx801

gemm_tam: gemm.cpp
	hipcc -o gemm.out gemm.cpp --amdgpu-target=gfx801

gemm: gemm.hip.cpp
	hipcc -o gemm.out gemm.hip.cpp --amdgpu-target=gfx801
