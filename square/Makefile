GEM5_PATH ?= ../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
ifeq (,$(HIP_PATH))
	HIP_PATH=/opt/rocm
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

ifeq (${HIP_PLATFORM}, nvcc)
	SOURCES=square.cu
else
	SOURCES=square.cpp
endif

all: square.out

# Step
square.cpp: square.cu
	$(HIP_PATH)/bin/hipify-perl square.cu > square.cpp

square.out: $(SOURCES)
	$(HIPCC) $(CXXFLAGS) $(SOURCES) -o bin/$@

square_m: square_m.hip.cpp
	hipcc $(CFLAGS) -o square_m square_m.hip.cpp $(LDFLAGS) --amdgpu-target=gfx801

square_case2: square_m.hip.cpp
	hipcc $(CFLAGS) -o bin/square_case2 square_case2.hip.cpp $(LDFLAGS) --amdgpu-target=gfx801

square_case3: square_m.hip.cpp
	hipcc $(CFLAGS) -o bin/square_case3 square_case3.hip.cpp $(LDFLAGS) --amdgpu-target=gfx801

square_m_2: square_m.hip.cpp
	hipcc -o bin/square_m_2.out square_m_2.hip.cpp --amdgpu-target=gfx801

clean:
	rm -f *.o *.out square.cpp

microbench: microbench.cpp
	mkdir -p bin
	hipcc $(CFLAGS) -o bin/microbench microbench.cpp  $(LDFLAGS) --amdgpu-target=gfx801
