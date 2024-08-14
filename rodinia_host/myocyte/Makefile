CPPSRC = main.cpp
EXECUTABLE = myocyte.out

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
HIPCC = $(HIP_PATH)/bin/hipcc -x hip

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(CXXFLAGS) $(INC_FLAGS) $(CPPSRC) -o $(EXECUTABLE) $(CFLAGS) -lm

clean:
	rm -f *.o *~ $(EXECUTABLE)

.PHONY: myocyte clean
