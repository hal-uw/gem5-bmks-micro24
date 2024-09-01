GEM5_PATH ?= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = main.c
EXECUTABLE = b+tree.out
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
#HIPCC = $(HIP_PATH)/bin/hipcc -x hip
HIPCC = $(HIP_PATH)/bin/hipcc

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(CXXFLAGS) $(INC_FLAGS) $(CPPSRC) $(LDFLAGS) -o $(EXECUTABLE) $(CFLAGS)

clean:
	rm -f *.o *~ $(EXECUTABLE)

.PHONY: b+tree clean
