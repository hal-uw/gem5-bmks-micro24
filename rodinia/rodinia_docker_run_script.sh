#!/bin/bash

# BTREE
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_btree configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/b+tree/  -c b+tree.out --options="file multigpu_benchmarks/rodinia/b+tree/mil.txt command multigpu_benchmarks/rodinia/b+tree/command.txt 1 1"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 |& tee btree_cpcoh.log  &

# BFS
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_bfs configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/bfs/  -c bin/bfs --options="multigpu_benchmarks/rodinia/bfs/graph65536.txt 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee bfs_cpcoh.log  &

# BACKPROP
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_backprop configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/backprop/  -c bin/backprop --options="65536 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee backprop_cpcoh.log  &

# DWT2D
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_dwt2d configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/dwt2d/  -c dwt2d --options="multigpu_benchmarks/rodinia/dwt2d/192.bmp -d 192x192 -f -5 -l 3 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee dwt2d_cpcoh.log  &

# GAUSSIAN
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_gaussian configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/gaussian/  -c gaussian --options="-f multigpu_benchmarks/rodinia/gaussian/matrix4.txt 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee gaussian_cpcoh.log  &

# HOTSPOT
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_hotspot configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/hotspot/  -c hotspot --options="512 2 2 multigpu_benchmarks/rodinia/hotspot/temp_512 multigpu_benchmarks/rodinia/hotspot/power_512 multigpu_benchmarks/rodinia/hotspot/output.out 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee hotspot_cpcoh.log  &

# HOTSPOT3D
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_hotspot3D configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/hotspot3D/  -c 3D --options="512 8 100 multigpu_benchmarks/rodinia/hotspot3D/power_512x8 multigpu_benchmarks/rodinia/hotspot/temp_512x8 multigpu_benchmarks/rodinia/hotspot/output.out"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee hotspot3D_cpcoh.log  &

# HUFFMAN
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_huffman configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/huffman/  -c pavle --options="multigpu_benchmarks/rodinia/huffman/test1024_H2.206587175259.in 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee huffman_cpcoh.log  &

# LAVAMD
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_lavamd configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/lavaMD/  -c lavaMD --options="-boxes1d 10"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee lavamd_cpcoh.log  &

# LUD
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_lud configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/lud/hip/  -c lud_hip --options="-i multigpu_benchmarks/rodinia/lud/256.dat 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee lud_cpcoh.log  &

docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_nn configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/nn/  -c nn --options="multigpu_benchmarks/rodinia/nn/filelist.txt -r 5 -lat 30 -lng 90 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee nn_cpcoh.log  &

docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_nw configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/nw/  -c needle --options="2048 10"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee nw_cpcoh.log  &

# PATHFINDER
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_pathfinder configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/pathfinder/  -c pathfinder --options="100000 100 20 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee pathfinder_cpcoh.log  &

# STREAM CLUSTER
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_streamcluster configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/streamcluster/  -c sc_gpu --options="10 20 256 65536 65536 1000 none output.txt 1"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee streamcluster_cpcoh.log  &

# SRAD v2
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_srad_v2 configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/srad/srad_v2/  -c srad --options="2048 2048 0 127 0 127 0.5 2 1 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee srad_v2_cpcoh.log  &


# SRAD v1
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_srad_v1 configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/rodinia/srad/srad_v1/  -c srad --options="100 0.5 502 458"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10  |& tee srad_v1_cpcoh.log  &