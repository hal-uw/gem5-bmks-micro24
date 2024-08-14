#!/bin/bash

# BC
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_bc configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/pannotia/bc/bin/  -c bc.gem5 --options="gem5-gpu-benchmark-suite/benchmarks/pannotia/bc/1k_128k.gr"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --default-acq-rel |& tee bc_cpcoh.log  &

# FW
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_fw configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/pannotia/bc/bin/  -c fw_hip.gem5 --options="gem5-gpu-benchmark-suite/benchmarks/pannotia/fw/1k_128k.gr"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --default-acq-rel |& tee fw_cpcoh.log  &

# Color
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_color configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/pannotia/color/bin/  -c color_max.gem5 --options="gem5-gpu-benchmark-suite/benchmarks/pannotia/color/1k_128k.gr 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --default-acq-rel |& tee color_cpcoh.log  &

# MIS
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_mis configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/pannotia/bc/bin/  -c mis_hip.gem5 --options="gem5-gpu-benchmark-suite/benchmarks/pannotia/mis/1k_128k.gr 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --default-acq-rel |& tee mis_cpcoh.log  &

# SSSP
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_sssp configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/pannotia/sssp/bin/  -c sssp.gem5 --options="gem5-gpu-benchmark-suite/benchmarks/pannotia/sssp/1k_128k.gr 0"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --default-acq-rel |& tee sssp_cpcoh.log  &

# Pagerank
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpcoh build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh,GPUDisp --debug-file=run_cpcoh.log --outdir=results_cpcoh_pagerank configs/example/apu_se.py -n16  --benchmark-root=multigpu_benchmarks/pannotia/pagerank/bin/  -c pagerank.gem5 --options="gem5-gpu-benchmark-suite/benchmarks/pannotia/pagerank/coAuthorsDBLP.graph 1"  --num-hw-queues=40 --num-gpus=4 --reg-alloc-policy=dynamic --gs-policy=GSP_RRCS --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --default-acq-rel |& tee pagerank_cpcoh.log  &