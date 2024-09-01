ifeq (,$(HIP_PATH))
	HIP_PATH=/opt/rocm
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

GEM5_PATH?= ../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5


SOURCES=BabelStream.cpp

all: BabelStreamN BabelStreamC 

GEM5_PATH= ../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5

BabelStreamN:
	hipcc $(CFLAGS) -o babelstream BabelStream.cpp $(LDFLAGS) --amdgpu-target=gfx801

BabelStreamC:
	hipcc $(CFLAGS) -o BabelStream_lk BabelStream_lk.cpp $(LDFLAGS) --amdgpu-target=gfx801

clean:
	rm -f *.o *.out
