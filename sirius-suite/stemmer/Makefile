GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
PLATFORMS = baseline pthread gpu
ifeq ($(shell which nvcc),)
PLATFORMS = baseline pthread
endif

.PHONY: all $(PLATFORMS)

SUBCLEAN = $(addsuffix .clean, $(PLATFORMS))
.PHONY: clean $(SUBCLEAN)

SUBTEST = $(addsuffix .test, $(PLATFORMS))
.PHONY: test $(SUBTEST)

all: $(PLATFORMS)
$(PLATFORMS):
	$(MAKE) -C $@

clean: $(SUBCLEAN)
$(SUBCLEAN): %.clean:
	$(MAKE) -C $* clean

test: $(SUBTEST)
$(SUBTEST): %.test:
	$(MAKE) -s -C $* test
