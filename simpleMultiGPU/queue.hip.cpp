// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <hip/hip_runtime.h>
#include <gem5/m5ops.h>
#include <timer.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <limits.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

const int MAX_GPU_COUNT = 7;
const int DATA_N        = 128;

__global__ void reduceKernel(float *d_Result, float *d_Input, int N)
{
    const int     tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int threadN = hipGridDim_x * hipBlockDim_x;
    float sum = 0;

    for (int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}

template <typename T>
__global__ void vector_square(T* C_d, const T* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

template <typename T>
__global__ void vector_add(T* C_d, const T* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] += C_d[i];
    }
}

int main(int argc, char **argv)
{
    TGPUplan      plan[MAX_GPU_COUNT];

    //GPU reduction results
    float     h_SumGPU[MAX_GPU_COUNT];

    float sumGPU;
    double sumCPU, diff;
    int i, j, gpuBase, GPU_N;
    int THREAD_N = 64;
    //int  BLOCK_N = DATA_N/THREAD_N;
	int BLOCK_N;
    //int  ACCUM_N = BLOCK_N * THREAD_N;
	int ACCUM_N = DATA_N;

    float *A_h, *C_h;
    size_t N = THREAD_N;
    size_t Nbytes = N * sizeof(int);
    A_h = (float*)malloc(Nbytes);
    C_h = (float*)malloc(Nbytes);
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    printf("Starting simpleMultiGPU\n");
    hipGetDeviceCount(&GPU_N);

    if (GPU_N > MAX_GPU_COUNT)
    {
        GPU_N = MAX_GPU_COUNT;
    }

    printf("CUDA-capable device count: %i\n", GPU_N);

    printf("Generating input data...\n\n");

    for (i = 0; i < GPU_N; i++)
    {
        plan[i].dataN = DATA_N / GPU_N;
    }

    gpuBase = 0;

    for (i = 0; i < GPU_N; i++)
    {
        plan[i].h_Sum = h_SumGPU + i;
        gpuBase += plan[i].dataN;
    }

    for (i = 0; i < GPU_N; i++)
    {
        hipSetDevice(i);
        hipStreamCreate(&plan[i].stream);
        plan[i].h_Sum_from_device = (float*) malloc(ACCUM_N * sizeof(float));
        plan[i].h_Data = (float*) malloc(plan[i].dataN * sizeof(float));
        for (j = 0; j < plan[i].dataN; j++)
        {
            plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
        }
    }

    for (i = 0; i < GPU_N; i++)
    {
        hipSetDevice(i);
		BLOCK_N = plan[i].dataN/THREAD_N;
        m5_getKernelArg(reinterpret_cast<uintptr_t>(plan[i].h_Sum_from_device), reinterpret_cast<uintptr_t>(plan[i].h_Data), reinterpret_cast<uintptr_t>(plan[i].dataN), 3, 3, 1);
        hipLaunchKernelGGL((reduceKernel), dim3(BLOCK_N), dim3(THREAD_N), 0, plan[i].stream, plan[i].h_Sum_from_device, plan[i].h_Data, plan[i].dataN);
    }

    //Process GPU results
    for (i = 0; i < GPU_N; i++)
    {
        float sum;
        hipSetDevice(i);
        hipStreamSynchronize(plan[i].stream);
        sum = 0;
        for (j = 0; j < ACCUM_N; j++)
        {
            sum += plan[i].h_Sum_from_device[j];
        }
        *(plan[i].h_Sum) = (float)sum;
        free(plan[i].h_Sum_from_device);
        hipStreamDestroy(plan[i].stream);
    }

	BLOCK_N = N/THREAD_N;
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h), reinterpret_cast<uintptr_t>(A_h), 0, 3, 2, 2);
    hipLaunchKernelGGL(vector_square, dim3(BLOCK_N), dim3(THREAD_N), 0, 0, C_h, A_h, N);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h), reinterpret_cast<uintptr_t>(A_h), 0, 3, 2, 3);
    hipLaunchKernelGGL(vector_add, dim3(BLOCK_N), dim3(THREAD_N), 0, 0, C_h, A_h, N);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h), reinterpret_cast<uintptr_t>(A_h), 0, 3, 2, 4);
    hipLaunchKernelGGL(vector_add, dim3(BLOCK_N), dim3(THREAD_N), 0, 0, C_h, A_h, N);

	hipStreamSynchronize(0);

	free(A_h);
	free(C_h);

    // Cleanup and shutdown
    for (i = 0; i < GPU_N; i++)
    {
        hipSetDevice(i);
        free(plan[i].h_Data);
        hipDeviceReset();
    }

}
