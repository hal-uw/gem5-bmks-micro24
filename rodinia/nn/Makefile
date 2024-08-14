GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
CPPSRC = nn_hip.cpp
EXE = nn

# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 -Wall
else
	OPTS += -O3 -Wall
endif

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC = $(HIP_PATH)/bin/hipcc
LOCAL_CC = gcc -g -O3 -Wall

all: $(EXE) hurricane_gen

$(EXE): $(CPPSRC)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801 $(CXXFLAGS) $(INC_FLAGS) $(CPPSRC) $(LDFLAGS) -o $(EXE) $(CFLAGS)

clean:
	rm -f *.o *~ $(EXE) hurricane_gen

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh

.PHONY: nn clean

