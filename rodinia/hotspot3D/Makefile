GEM5_PATH ?= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
SRC = 3D.cpp
EXE = 3D 

OUTPUT = *.out

# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 
else
	OPTS += -O3
endif

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC = $(HIP_PATH)/bin/hipcc

all: $(EXE)

$(EXE): $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(CXXFLAGS) $(INC_FLAGS) $(SRC) $(LDFLAGS) -o $(EXE) $(CFLAGS)

clean:
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 

.PHONY: 3D clean

