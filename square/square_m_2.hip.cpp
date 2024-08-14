/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include "hip/hip_runtime.h"

#define task 1

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void
vector_square(T *C_d, const T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

template <typename T>
__global__ void 
vector_add(T* C_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] += C_d[i];
    }
}


__host__ void call_vector_square(unsigned blocks, unsigned threadsPerBlock, float* C_d, float* A_d, size_t N, hipStream_t stream, uint32_t lastKernel) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks), dim3(threadsPerBlock), 0, stream, C_d, A_d, N);
}

__host__ void call_vector_add(unsigned blocks, unsigned threadsPerBlock, float* C_d, size_t N, hipStream_t stream, uint32_t lastKernel) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_add), dim3(blocks), dim3(threadsPerBlock), 0, stream, C_d, N);
}


int main(int argc, char *argv[])
{
    float *A_h, *C_h, *D_h, *E_h;
    size_t N = 64 * 256;
    size_t Nbytes = N * sizeof(float);

    hipStream_t hip_stream[task];

    for (int i = 0; i < task; i++) {
        hipStreamCreate(&hip_stream[i]);
        //hipStreamCreateWithPriority(&hip_stream[i], hipStreamDefault, Kalmar::priority_high);
    }

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    #ifdef __HIP_PLATFORM_HCC__
      printf ("info: architecture on AMD GPU device is: %d\n",props.gcnArch);
    #endif
    printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    D_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    E_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    // Fill with Phi + i
    for (size_t i=0; i<N; i++)
    {
        A_h[i] = 1.618f + i;
    }

    const unsigned blocks =64;
    const unsigned threadsPerBlock = 256;

    printf ("info: launch 'vector_square' kernel\n");

    for(int i = 0; i < task; i++)
    {
        call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, hip_stream[i], 0);
        call_vector_square(blocks, threadsPerBlock, D_h, A_h, N, hip_stream[i], 0);
        call_vector_square(blocks, threadsPerBlock, E_h, A_h, N, hip_stream[i], 1);

        hipDeviceSynchronize();
        printf("Check stuff\n");
        for (size_t i = 0; i < N; i++) {
            if (C_h[i] != A_h[i] * A_h[i] ||
                D_h[i] != A_h[i] * A_h[i] ||
                E_h[i] != A_h[i] * A_h[i]) {
                printf("Err: C: %f D: %f E: %f A*A: %f\n", C_h[i], D_h[i], E_h[i], A_h[i]*A_h[i]);
                CHECK(hipErrorUnknown);
                return -1;
            }
        }
        printf("Done with check\n");

    }

    free(A_h);
    free(C_h);
    free(D_h);
    free(E_h);

    return 0;
}
