GEM5_PATH ?= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = main.cpp kernel/kernel_gpu_hip_wrapper.cpp util/device/device.cpp util/num/num.c util/timer/timer.c
EXECUTABLE = lavaMD
# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 
else
	OPTS += -O3
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
# HIP technically only supports C++, so need to tell hipcc we're compiling
# HIP code
#HIPCC = $(HIP_PATH)/bin/hipcc -x hip
HIPCC = $(HIP_PATH)/bin/hipcc

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(KERNEL_DIM) $(OUTPUT) $(CXXFLAGS) $(INC_FLAGS) $(CPPSRC) $(LDFLAGS) -o $(EXECUTABLE) $(CFLAGS)

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o
	$(C_C)	$(KERNEL_DIM) main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o \
			-lm \
			-L$(CUDA_LIB_DIR) \
			-lcudart \
			$(OMP_LIB) \
			-lstdc++ \
			-o lavaMD

# delete all object and executable files
clean:
	rm -f *.o $(EXECUTABLE)

.PHONY: lavaMD clean

