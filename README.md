# CPCoh Benchmarks

Status:

- [X] Square
- [ ] Rodinia
    - [?] Btree -> Has more than 4 args
    - [X] Gaussian
    - [X] Backprop
    - [?] BFS -> Has 6 args
    - [ ] DW2D
    - [X] Hotspot
    - [?] Hotspot3D -> No hipLaunchGGL/__global__
    - [ ] Huffman
    - [ ] LavaMD
    - [ ] LUD
    - [ ] NN
    - [ ] NW
    - [X] Pathfinder
    - [ ] SRAD
    - [ ] StreamCluster

## Setup

```bash
cd /nobackup/
mkdir -p $USER
cd $USER
loginctl enable-linger
systemctl --user enable docker.service # CSL docker settings
git clone --branch HAL_CpCoh https://github.com/hal-uw/gem5_multigpu/
cd gem5_multigpu
cd util/m5
scons build/x86/out/m5
cd ../../
git clone --branch cpcoh_1 https://github.com/hal-uw/multigpu_benchmarks
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh scons build/GCN3_X86/gem5.opt -j$(nproc)
# For debugging: docker run -it --volume $(pwd):$(pwd) -w $(pwd) dalmia/cpcoh /bin/bash
```

## Square

```bash
docker run --rm -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/square/ pdalmia/cpcoh mkdir bin
docker run --rm -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/square/ pdalmia/cpcoh make square_m
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh build/GCN3_X86/gem5.opt  --outdir=square_test configs/example/apu_se.py -n3 --benchmark-root=multigpu_benchmarks/square/bin -c square_m --options="16384 1 1 0 64 256 0" --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RR --num-compute-units=60 --cu-per-sa=15 --num-gpu-complex=4 --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --num-dirs=64 --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800
```

## Porting

```cpp
#include <gem5/m5ops.h>
m5_getKernelArg(reinterpret_cast<uintptr_t>(), reinterpret_cast<uintptr_t>(), 0, 0, 3, 2);
```

```bash
# Makefile
GEM5_PATH= ../../../
CFLAGS += -I$(GEM5_PATH)/include
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5

# find . -type f -name 'Makefile' -exec sed -i '1s#^#GEM5_PATH= ../../../\nCFLAGS += -I$(GEM5_PATH)/include\nLDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5\n#' {} \;

$(CFLAGS) $(LDFLAGS)

```

- hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1); -> hipStreamCreate(&hip_stream[i]); `find ./ -type f -exec sed -i -e 's#hipStreamCreateWithFlags(\&hip_stream\[i\], 0x01, -1)#hipStreamCreateWithFlags(\&hip_stream\[i\])#g' {} \;`
- hipStreamCreateWithFlags(&hip_stream[i], 0x01); -> hipStreamCreate(&hip_stream[i]); `find ./ -type f -exec sed -i -e 's#hipStreamCreateWithFlags(\&hip_stream\[i\], 0x01)#hipStreamCreateWithFlags(\&hip_stream\[i\])#g' {} \;`
- Remove hipHccModuleRingDoorbell(hip_stream[i]);
- Add < numStreams if (individualGpus) hipSetDevice(i % numGpus); // Not needed
- hipLaunchKernelGGL_lk -> hipLaunchKernelGGL `find ./ -type f -exec sed -i -e 's/hipLaunchKernelGGL_lk/hipLaunchKernelGGL/g' {} \;`
