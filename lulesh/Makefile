BIN_DIR?= ./bin
GEM5_PATH= ../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5

all: $(BIN_DIR)
	hipcc src/lulesh.hip.cc $(CFLAGS) -o $(BIN_DIR)/lulesh  $(LDFLAGS) --amdgpu-target=gfx801

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
