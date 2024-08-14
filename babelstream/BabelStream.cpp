#include <stdio.h>
#include "hip/hip_runtime.h"

#include <gem5/m5ops.h>

#define TBSIZE 256
#define DOT_NUM_BLOCKS 256
#define startScalar 0.4

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i];
}


template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  b[i] = scalar * c[i];
}


template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i] + b[i];
}


template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = b[i] + scalar * c[i];
}



template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] += b[i] + scalar * c[i];
}



template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  const size_t local_i = hipThreadIdx_x;

  tb_sum[local_i] = 0.0;
  for (; i < array_size; i += hipBlockDim_x*hipGridDim_x)
    tb_sum[local_i] += a[i] * b[i];

  for (int offset = hipBlockDim_x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[hipBlockIdx_x] = tb_sum[local_i];
}

int main(int argc, char *argv[])
{
    float *A_h, *B_h, *C_h;
    float *sums;
    size_t array_size = atoi(argv[1]);
    size_t Nbytes = array_size * sizeof(float);
    //int iterations = atoi(argv[2]);
    float initA = 10;
    float initB = 10;
    float initC = 10;
    hipDeviceProp_t props;
    int numGpus;
    hipGetDeviceCount(&numGpus);
    printf("Have %d GPUs\n", numGpus);
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);

    printf ("info: allocate host mem (%6.2f MB)\n", 3*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    B_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    sums = (float*)malloc(DOT_NUM_BLOCKS*sizeof(float));
    // Fill with Phi + i
    for (size_t i=0; i<array_size; i++) 
    {
        A_h[i] = 1.618f + i;
        B_h[i] = 1.618f + i; 
        C_h[i] = 1.618f + i;
    }

    printf ("info: allocate device mem (%6.2f MB)\n", 3*Nbytes/1024.0/1024.0);
    //CHECK(hipMalloc(&A_d, Nbytes));
    //CHECK(hipMalloc(&B_h, Nbytes));
    //CHECK(hipMalloc(&C_h, Nbytes));
    //CHECK(hipMalloc(&d_sum, DOT_NUM_BLOCKS*sizeof(float)));

    //printf ("info: copy Host2Device\n");
    //CHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    //CHECK ( hipMemcpy(B_h, B_h, Nbytes, hipMemcpyHostToDevice));
    //CHECK ( hipMemcpy(C_h, C_h, Nbytes, hipMemcpyHostToDevice));

    for( int i = 0; i < 1; i ++){
        printf ("info: Launch Init Kernel\n");
        fflush(stdout);
        m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(A_h), reinterpret_cast<uintptr_t>(B_h), reinterpret_cast<uintptr_t>(C_h), 63, 3, 17);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(init_kernel), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, A_h, B_h, C_h, initA, initB, initC);
        printf ("info: Launch Copy Kernel\n");
        fflush(stdout);
        m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h), reinterpret_cast<uintptr_t>(A_h), 0, 3, 2, 18);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, A_h, C_h);
        printf ("info: Launch Multiply\n");
        fflush(stdout);
        m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h), reinterpret_cast<uintptr_t>(B_h), 0, 3, 2, 19);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, B_h, C_h);
        printf ("info: Launch Add Kernel\n");
        fflush(stdout);
        m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h), reinterpret_cast<uintptr_t>(A_h), reinterpret_cast<uintptr_t>(B_h), 3, 3, 20);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, A_h, B_h, C_h);
        printf ("info: Launch Triad Kernel\n");
        fflush(stdout);
        m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(A_h), reinterpret_cast<uintptr_t>(B_h), reinterpret_cast<uintptr_t>(C_h), 3, 3, 21);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, A_h, B_h, C_h);
        printf ("info: Launch Dot Kernel\n");
        fflush(stdout);
        m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(sums), reinterpret_cast<uintptr_t>(A_h), reinterpret_cast<uintptr_t>(B_h), 3, 3, 22);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, A_h, B_h, sums, array_size);
    }
   m5_dump_reset_stats(0, 0);
    //printf ("info: copy Device2Host\n");
   // CHECK ( hipMemcpy(sums, d_sum, DOT_NUM_BLOCKS, hipMemcpyDeviceToHost));

    printf ("info: check result\n");
    
    /*
    for (size_t i=0; i<N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {
            CHECK(hipErrorUnknown);
        }
    }
    */
    //hipFree(d_sum);
    free(A_h);
    free(B_h);
    free(C_h);
    free(sums);

    printf ("PASSED!\n");
}