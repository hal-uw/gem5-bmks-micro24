#!/bin/bash

cd /MIOpen/src/kernels

# DeepBench 128 1 1 lstm
# Missing the first kernel, whoops

mkdir -p /.cache/miopen/1.7.0/71ff09ee368a7657071714e18c2f1a18/
/opt/rocm/bin/clang-ocl  -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -mcpu=gfx801  -Wno-everything MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/71ff09ee368a7657071714e18c2f1a18/MIOpenSubTensorOpWithScalarKernel.cl.o

mkdir -p  /.cache/miopen/1.7.0/1ea2111c5cc9df0d43e6c8b9712fd005/
/opt/rocm/bin/clang-ocl  -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -mcpu=gfx801  -Wno-everything MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/1ea2111c5cc9df0d43e6c8b9712fd005/MIOpenSubTensorOpWithSubTensorKernel.cl.o

mkdir -p /.cache/miopen/1.7.0/e2b4e55b04ad05d0e642dc5fdf7d1f43
/opt/rocm/bin/clang-ocl  -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DBIAS -DMAX_NUM_WG=4096 -mcpu=gfx801  -Wno-everything MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/e2b4e55b04ad05d0e642dc5fdf7d1f43/MIOpenTensorKernels.cl.o

# 256 8 1 vanilla
mkdir -p /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4096 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 256 8 2 vanilla
mkdir -p /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/ebd4bba7c276783c6776d5a29844495b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4096 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/ebd4bba7c276783c6776d5a29844495b/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 256 8 4 vanilla
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 256 8 8 vanilla
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 128 4 1 lstm
mkdir -p /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o

# 128 4 2 lstm
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/1d73f8dfe1c58d912f6012e4d31406f2
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/1d73f8dfe1c58d912f6012e4d31406f2/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 128 4 4 lstm
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 128 4 8 lstm
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# 128 4 1 gru
mkdir -p /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4096 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o

# 128 4 2 gru
mkdir -p /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/1d73f8dfe1c58d912f6012e4d31406f2
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/1d73f8dfe1c58d912f6012e4d31406f2/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4096 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc/MIOpenSubTensorOpWithScalarKernel.cl.o

# 128 4 4 gru
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b/MIOpenSubTensorOpWithScalarKernel.cl.o

# 128 4 8 gru
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o


# vanilla
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4096 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/b689dad086da5ae0795a8de6696edb95/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/17c27bd8d4bf01de06c93a62f766222e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/35d7f3e3161567193e3e3caffbb7ec9b/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a959f9d1ddd3fbdeb14239e02965398a/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=3 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/8ee45353ec460a44a6fe7a42e586b3bd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=2048 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/bb463432cf5cc33896d929c89238e855/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/72189144e09bdc13e774473ddbafb143/MIOpenSubTensorOpWithSubTensorKernel.cl.o

# gru
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4096 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/ac0046008721a79b06896f9a5a3ca2cc/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16384 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/e736d631316700e9b4c954aed5d60e61/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenAdd -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/56d3d32eb9aaf7ce07782a359c35295e/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/aacafe3a883afa8aeb56932a374115ab/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o

# lstm
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=8192 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/93d8cb87d5b4a198524bec6a557d339b/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=16 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/0d77c06cefc58e9a324c619e0683898e/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/d7a1890fe38e8c500ce350d4842262b8/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=64 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/f8666c00a334ff3df03c9302ef2dd629/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=256 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/cfcbc4683c0f9e4c2db9aa76a97cb499/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32768 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/a8ef6200bc476547cd8819f4d7328d47/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=32 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/55e7af84ee985483c35435560211ccfd/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=65536 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/2f00bddb394ecbfd4d236df217e5bcb4/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=512 -mcpu=gfx801 MIOpenSubTensorOpWithScalarKernel.cl -o /.cache/miopen/1.7.0/27d056f96f195163c6d3a7593ac845d7/MIOpenSubTensorOpWithScalarKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=128 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/30144dc379b7e72463a7d64b2bd1bedf/MIOpenSubTensorOpWithSubTensorKernel.cl.o
mkdir -p /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/4c8258ae4791bf376648d3f20a4a197e/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd
/opt/rocm/bin/clang-ocl -DLITE -DMIOPEN_READ_UNIT=4 -DMIOPEN_READ_TYPE=_FLOAT4 -DMIOPEN_NRN_OP_ID=2 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -mcpu=gfx801 MIOpenNeuron.cl -o /.cache/miopen/1.7.0/3a7d5800c50a09bc37f0e81566ba5dcd/MIOpenNeuron.cl.o
mkdir -p /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64
/opt/rocm/bin/clang-ocl -DMIOPEN_TYPE=float -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DMIOPEN_TENSOR_OP=miopenMul -DUSE_2D_TENSOR_LITE -DRD_BLCK=4 -DMAP_RD=32 -DREAD_TYPE=float4 -DBETA -DMAX_NUM_WG=4096 -mcpu=gfx801 MIOpenTensorKernels.cl -o /.cache/miopen/1.7.0/d72de5bb4db116e22a10a63569632f64/MIOpenTensorKernels.cl.o
mkdir -p /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde
/opt/rocm/bin/clang-ocl -DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1 -DWORK_LENGTH_0=4 -DWORK_LENGTH_1=128 -mcpu=gfx801 MIOpenSubTensorOpWithSubTensorKernel.cl -o /.cache/miopen/1.7.0/7129e79f97bbf34ff9571fe52d55bdde/MIOpenSubTensorOpWithSubTensorKernel.cl.o
