GEM5_PATH ?= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
EXECUTABLE := dwt2d

# Debug options
dbg=0
performancetest=1

# Files
CPPFILES := main.cpp dwt.cpp components.cpp dwt_hip/fdwt53.cpp dwt_hip/fdwt97.cpp dwt_hip/common.cpp dwt_hip/rdwt97.cpp dwt_hip/rdwt53.cpp

# Common flags
CFLAGS += $(COMMONFLAGS)

# Warning flags (from cuda common.mk)
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CFLAGS += $(CWARN_FLAGS)
CXXFLAGS += $(CXXWARN_FLAGS)

# Debug/release flags
ifeq ($(dbg),1)
    COMMONFLAGS += -g 
    CXXFLAGS    += -D_DEBUG
    CFLAGS      += -D_DEBUG
else 
    COMMONFLAGS += -O3 
    CXXFLAGS    += -fno-strict-aliasing
    CFLAGS      += -fno-strict-aliasing
endif

#ifeq ($(performancetest),1)
#    COMMONFLAGS += -DGPU_DWT_TESTING
#endif

#ifdef OUTPUT
#	override OUTPUT = -DOUTPUT
#endif

OUTPUT = -DOUTPUT

# Compilers
ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC = $(HIP_PATH)/bin/hipcc

$(EXECUTABLE): $(CPPFILES)
	$(HIPCC) $(COMMONFLAGS) --amdgpu-target=gfx801 $(CXXFLAGS) $(CFLAGS) $(CPPFILES) $(LDFLAGS) -o $(EXECUTABLE)

clean:
	rm -f $(COBJS) $(CXXOBJS) $(CUOBJS) $(EXECUTABLE)
	rm -f *.bmp.dwt.*		
