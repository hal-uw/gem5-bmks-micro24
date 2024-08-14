GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = backprop.c facetrain.c imagenet.c backprop_hip.c
EXECUTABLE = backprop
# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 
else
	OPTS += -O3
endif

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC = $(HIP_PATH)/bin/hipcc

BIN_DIR := ./

all: $(EXECUTABLE)

$(EXECUTABLE): $(CPPSRC) $(BIN_DIR)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(CXXFLAGS) $(INC_FLAGS) $(CPPSRC) $(LDFLAGS) -o $(BIN_DIR)/$(EXECUTABLE) $(CFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -f *.o *~ $(EXECUTABLE)

.PHONY: backprop clean
