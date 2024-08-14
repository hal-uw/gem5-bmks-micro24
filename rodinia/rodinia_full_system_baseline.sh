#!/bin/bash

# BTREE
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_btree configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/b+tree/  -c b+tree.out --options="file multigpu_benchmarks/rodinia/b+tree/mil.txt command multigpu_benchmarks/rodinia/b+tree/command.txt 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel |& tee btree_baseline_large.log  &

# BFS
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_bfs configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/bfs/  -c bin/bfs --options="multigpu_benchmarks/rodinia/bfs/graph65536.txt 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee bfs_baseline_large.log  &

# BACKPROP
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_backprop configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/backprop/  -c bin/backprop --options="65536 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee backprop_baseline_large.log  &

# DWT2D
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_dwt2d configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/dwt2d/  -c dwt2d --options="multigpu_benchmarks/rodinia/dwt2d/192.bmp -d 192x192 -f -5 -l 3 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee dwt2d_baseline_large.log  &

# GAUSSIAN
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_gaussian configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/gaussian/  -c gaussian --options="-f multigpu_benchmarks/rodinia/gaussian/matrix208.txt 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee gaussian_baseline_large.log  &

# # HOTSPOT
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_hotspot configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/hotspot/  -c hotspot --options="512 2 2 multigpu_benchmarks/rodinia/hotspot/temp_512 multigpu_benchmarks/rodinia/hotspot/power_512 multigpu_benchmarks/rodinia/hotspot/output.out 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee hotspot_baseline_large.log  &

# HOTSPOT3D
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_hotspot3D configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/hotspot3D/  -c 3D --options="512 8 100 multigpu_benchmarks/rodinia/hotspot3D/power_512x8 multigpu_benchmarks/rodinia/hotspot/temp_512x8 multigpu_benchmarks/rodinia/hotspot/output.out"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee hotspot3D_baseline_large.log  &

# # HUFFMAN
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_huffman configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/huffman/  -c pavle --options="multigpu_benchmarks/rodinia/huffman/test1024_H2.206587175259.in 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee huffman_baseline_large.log  &

# LAVAMD
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_lavamd configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/lavaMD/  -c lavaMD --options="-boxes1d 10"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee lavamd_baseline_large.log  &

# LUD
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_lud configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/lud/hip/  -c lud_hip --options="-i multigpu_benchmarks/rodinia/lud/512.dat 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee lud_baseline_large.log  &

# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_nn configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/nn/  -c nn --options="multigpu_benchmarks/rodinia/nn/filelist.txt -r 5 -lat 30 -lng 90 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee nn_baseline_large.log  &

docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_nw configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/nw/  -c needle --options="2048 10"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee nw_baseline_large.log  &

# PATHFINDER
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_pathfinder configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/pathfinder/  -c pathfinder --options="100000 100 20 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee pathfinder_baseline_large.log  &

# STREAM CLUSTER
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_streamcluster configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/streamcluster/  -c sc_gpu --options="10 20 256 65536 65536 1000 none output.txt 1"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee streamcluster_baseline_large.log  &

# SRAD v2
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_srad_v2 configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/srad/srad_v2/  -c srad --options="2048 2048 0 127 0 127 0.5 2 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee srad_v2_baseline_large.log  &


# SRAD v1
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) pdalmia/cpcoh_kr:v1 build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_srad_v1 configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/srad/srad_v1/  -c srad --options="100 0.5 502 458"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel  |& tee srad_v1_baseline_large.log  &