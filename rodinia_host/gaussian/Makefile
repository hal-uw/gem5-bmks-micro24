GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = gaussian.cpp
EXE = gaussian
# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 
else
	OPTS += -O3
endif

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC = $(HIP_PATH)/bin/hipcc

KERNEL_DIM := -DRD_WG_SIZE_0=128 -DRD_WG_SIZE_1=16

all: $(EXE)

$(EXE): $(CPPSRC)
	$(HIPCC) $(KERNEL_DIM) $(OPTS) --amdgpu-target=gfx801,gfx803 $(CXXFLAGS) $(INC_FLAGS) $(CPPSRC) $(LDFLAGS) -o $(EXE) $(CFLAGS)

clean:
	rm -f *.o *~ $(EXE)

.PHONY: gaussian clean
