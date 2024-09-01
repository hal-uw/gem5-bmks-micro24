GEM5_PATH?= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
SRC = main_test_hip.cpp cpuencode.cpp stats_logger.cpp
EXE = pavle
# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	OPTS += -g -O0 
else
	OPTS += -O3
endif

OPTS += -Wall -Wextra -m64

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC = $(HIP_PATH)/bin/hipcc

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

all: $(SRC)
	$(HIPCC) $(TESTING) $(CACHECWLUT) $(OPTS) --amdgpu-target=gfx801 $(CXXFLAGS) $(INC_FLAGS) $(SRC) $(LDFLAGS) -o $(EXE) $(CFLAGS)

clean:
	rm -f *.o $(EXE) 
