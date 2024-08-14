GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5
SUBDIRS= nvidia osu_allreduce baidu_allreduce

subdirs: $(SUBDIRS)

$(SUBDIRS):
	mkdir -p bin
	$(MAKE) -C $@
	cp $@/bin/* bin
	rm -rf $@/bin

.PHONY: subdirs $(SUBDIRS)

clean: 
	rm -rf bin
