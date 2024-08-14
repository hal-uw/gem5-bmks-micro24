GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = main.cpp AVI/avilib.c AVI/avimod.c define.c setdevice.cpp
EXE = heartwall
# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 
else
	OPTS += -O3
endif

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
# HIP technically only supports C++, so need to tell hipcc we're compiling
# HIP code
HIPCC = $(HIP_PATH)/bin/hipcc

INC_FLAGS = -I./AVI

ifdef OUTPUT
override OUTPUT = -DOUTPUT
OPTS += -DOUTPUT
endif

all: $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(INC_FLAGS) $(CXXFLAGS) $(CPPSRC) $(LDFLAGS) -o $(EXE)

# delete all object files
clean:
	rm -f *.o AVI/*.o $(EXE) *.linkinfo

.PHONY: heartwall clean
