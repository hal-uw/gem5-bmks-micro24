#!/bin/bash

cd /MIOpen/src/kernels

# 64 1 1 lstm
mkdir -p /.cache/miopen/1.7.0/6cae50be837953aae4f49f0c41fc3fc6
/opt/rocm/bin/clang-ocl  -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=1024 -mcpu=gfx801 -Wno-everything MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/6cae50be837953aae4f49f0c41fc3fc6/MIOpenSubTensorOpWithScalarKernel.cl.o

mkdir -p /.cache/miopen/1.7.0/6dbe32a0a86f23a3b1fd9fb933448156
/opt/rocm/bin/clang-ocl  -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -mcpu=gfx801 -Wno-everything MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/6dbe32a0a86f23a3b1fd9fb933448156/MIOpenSubTensorOpWithScalarKernel.cl.o

mkdir -p /.cache/miopen/1.7.0/47099a6ce6ea23780cd06deb32e26f03
/opt/rocm/bin/clang-ocl  -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -mcpu=gfx801 -Wno-everything MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/47099a6ce6ea23780cd06deb32e26f03/MIOpenSubTensorOpWithSubTensorKernel.cl.o

mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl  -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 -Wno-everything MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o

mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl  -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 -Wno-everything MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o

mkdir -p /.cache/miopen/1.7.0/dda78bb575e349c38d3ee181f838df6e
/opt/rocm/bin/clang-ocl  -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=16 -DREAD_TYPE=float4 -DBETA -DBIAS -DMAX_NUM_WG=4096 -mcpu=gfx801 -Wno-everything MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/dda78bb575e349c38d3ee181f838df6e/MIOpenTensorKernels.cl.o

mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl  -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 -Wno-everything MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o

mkdir -p /.cache/miopen/1.7.0/b373949ce8b09d00dadc7d5a49456f5f
/opt/rocm/bin/clang-ocl  -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=16 -DREAD_TYPE=float4 -DBIAS -DMAX_NUM_WG=4096 -mcpu=gfx801 -Wno-everything MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/b373949ce8b09d00dadc7d5a49456f5f/MIOpenTensorKernels.cl.o

mkdir -p /.cache/miopen/1.7.0/f54682734788df0eaf6dfc7c53e0a122
/opt/rocm/bin/clang-ocl  -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=16 -DREAD_TYPE=float4 -DBIAS -DMAX_NUM_WG=4096 -mcpu=gfx801 -Wno-everything MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/f54682734788df0eaf6dfc7c53e0a122/MIOpenTensorKernels.cl.o
