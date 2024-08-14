GEM5_PATH= ../../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = lud.cpp ../common/common.c
EXECUTABLE = lud_hip

# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 -Wall
else
	OPTS += -O3 -Wall
endif

DEFS += \
	-DGPU_TIMER \
	$(SPACE)

LDFLAGS += -lm
CFLAGS += -I../common \
	  -Wall \
	  $(SPACE)

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
# HIP technically only supports C++, so need to tell hipcc we're compiling
# HIP code
#HIPCC = $(HIP_PATH)/bin/hipcc -x hip
HIPCC = $(HIP_PATH)/bin/hipcc

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(KERNEL_DIM) $(CXXFLAGS) $(INC_FLAGS) $(DEFS) $(CPPSRC) $(LDFLAGS) -o $(EXECUTABLE) $(CFLAGS)

clean:
	rm -f *.o *~ $(EXECUTABLE)

.PHONY: lud clean