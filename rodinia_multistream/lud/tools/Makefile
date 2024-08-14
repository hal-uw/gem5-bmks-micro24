GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
gen_input: gen_input.c
	gcc -o gen_input -O3 gen_input.c -fopenmp
