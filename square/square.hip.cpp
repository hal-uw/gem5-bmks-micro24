/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
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

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(T* C_d, const T* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

template <typename T>
__global__ void vector_add(T* C_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] += C_d[i];
    }
}

__host__ void call_vector_square(unsigned blocks, unsigned threadsPerBlock, float* C_d, const float* A_d, size_t N, uint32_t lastKernel) {
	hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks), dim3(threadsPerBlock), 0, 0, lastKernel, C_d, A_d, N);
}

__host__ void call_vector_add(unsigned blocks, unsigned threadsPerBlock, float* C_d, size_t N, uint32_t lastKernel) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_add), dim3(blocks), dim3(threadsPerBlock), 0, 0, lastKernel, C_d, N);
}

int main(int argc, char* argv[]) {
    float *A_h, *C_h;
    size_t N = 8 * 256;
    size_t Nbytes = N * sizeof(int);
    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_HCC__
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
#endif
    printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (float*)malloc(Nbytes);
    C_h = (float*)malloc(Nbytes);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    const unsigned blocks = 8;
    const unsigned threadsPerBlock = 256;

    printf("info: launch 'vector_square' kernel\n");

    call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, 0);

    call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, 0);

    call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, 0);

    call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, 0);

    call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, 0);

	call_vector_square(blocks, threadsPerBlock, C_h, A_h, N, 1);

    hipDeviceSynchronize();

    /*printf("info: check result\n");
    for (size_t i = 0; i < N; i++) {
        if (C_h[i] != (A_h[i] * A_h[i] * 4)) {
            CHECK(hipErrorUnknown);
        }
    }
    printf("PASSED!\n");*/
 
    free(A_h);
    free(C_h);
}
