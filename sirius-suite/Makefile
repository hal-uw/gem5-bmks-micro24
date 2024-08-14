GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
KERNELS = gmm dnn-asr fe fd crf regex stemmer

.PHONY: all $(KERNELS)

SUBCLEAN = $(addsuffix .clean, $(KERNELS))
.PHONY: clean $(SUBCLEAN)

SUBTEST = $(addsuffix .test, $(KERNELS))
.PHONY: test $(SUBTEST)

all: $(KERNELS)
$(KERNELS):
	$(MAKE) -C $@

clean: $(SUBCLEAN)
$(SUBCLEAN): %.clean:
	$(MAKE) -C $* clean

test: $(SUBTEST)
$(SUBTEST): %.test:
	@$(MAKE) -s -C $* test
