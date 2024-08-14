GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
GEM5_PATH ?= ../../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5

BIN_DIR := ./bin

# File names
EXEC = gmm_scoring

gmm_scoring: gmm_scoring.hip.cpp $(BIN_DIR)
	hipcc $(CFLAGS) -o $(BIN_DIR)/$@ $< $(LDFLAGS) --amdgpu-target=gfx801

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# To remove generated files
clean:
	rm -rf $(BIN_DIR)
