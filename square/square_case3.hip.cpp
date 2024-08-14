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

#include <gem5/m5ops.h>
#include <stdio.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "hip/hip_runtime.h"

#define CHECK(cmd)                                                        \
    {                                                                     \
        hipError_t error = cmd;                                           \
        if (error != hipSuccess) {                                        \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                 \
                    hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(T *C_d, const T *A_d, size_t N,
                              size_t numRepeats) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t j = 0; j < numRepeats; j++) {
        for (size_t i = offset; i < N; i += stride) {
            C_d[i] += A_d[i] * A_d[i];
        }
    }
}

template <typename T>
__global__ void vector_add(T *C_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] += C_d[i];
    }
}

__host__ void call_vector_square(unsigned blocks, unsigned threadsPerBlock,
                                 float *C_d, float *A_d, size_t N,
                                 size_t repeats, hipStream_t stream,
                                 uint32_t lastKernel) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks),
                       dim3(threadsPerBlock), 0, stream, C_d, A_d, N, repeats);
}

__host__ void call_vector_add(unsigned blocks, unsigned threadsPerBlock,
                              float *C_d, size_t N, hipStream_t stream,
                              uint32_t lastKernel) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_add), dim3(blocks),
                       dim3(threadsPerBlock), 0, stream, C_d, N);
}

int main(int argc, char *argv[]) {
    assert(argc == 8);
    size_t N = atoi(argv[1]);
    size_t Nbytes = N * sizeof(float);

    size_t numStreams = atoi(argv[2]);
    size_t numKernels = atoi(argv[3]);
    size_t numRepeats = atoi(argv[4]);
    unsigned blocks = atoi(argv[5]);
    unsigned threadsPerBlock = atoi(argv[6]);
    bool individualGpus = atoi(argv[7]);

    int numGpus;
    hipGetDeviceCount(&numGpus);
    printf("Have %d GPUs\n", numGpus);

    std::vector<float *> A_h(numStreams);
    std::vector<float *> C_h(numStreams);

    hipStream_t hip_stream[numStreams];

    for (int i = 0; i < numStreams; i++) {
        if (individualGpus) hipSetDevice(i % numGpus);
        hipStreamCreate(&hip_stream[i]);
    }

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0 /*deviceID*/));
    printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_HCC__
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
#endif
    printf("info: allocate host mem (%6.2f MB)\n",
           2 * Nbytes / 1024.0 / 1024.0);

    // In case we want to allocate A for each stream
    // Fill with Phi + i
    for (size_t i = 0; i < numStreams; i++) {
        A_h[i] = (float *)malloc(Nbytes);
        C_h[i] = (float *)malloc(Nbytes);
        CHECK(A_h[i] == 0 ? hipErrorMemoryAllocation : hipSuccess);
        for (size_t j = 0; j < N; j++) {
            A_h[i][j] = 1.618f + ((i * N + j) / (float)N);
            C_h[i][j] = 0;
        }
    }

    printf("info: launch 'vector_square' kernel\n");

    m5_dump_reset_stats(0, 0);

    for (int i = 0; i < numStreams; i++) {
        for (int j = 0; j < numKernels; j++) {
            if (individualGpus) hipSetDevice(i % numGpus);
            if (j % 2 == 0) {
                m5_getKernelArg(reinterpret_cast<uintptr_t>(A_h[i]),
                                reinterpret_cast<uintptr_t>(C_h[i]), 0, 0, 12,
                                2);
                call_vector_square(blocks, threadsPerBlock, C_h[i], A_h[i], N,
                                   numRepeats, hip_stream[i], 1);
            } else {
                m5_getKernelArg(reinterpret_cast<uintptr_t>(C_h[i]),
                                reinterpret_cast<uintptr_t>(A_h[i]), 0, 0, 12,
                                2);
                call_vector_square(blocks, threadsPerBlock, A_h[i], C_h[i], N,
                                   numRepeats, hip_stream[i], 1);
            }
        }
    }

    for (int i = 0; i < numStreams; i++) {
        if (individualGpus) hipSetDevice(i % numGpus);
        hipStreamSynchronize(hip_stream[i]);
    }
    m5_dump_reset_stats(0, 0);

/*
    printf("Checking 100 numbers\n");
    for (int i = 0; i < numStreams; i++) {
        for (int k = 0; k < numKernels; k++) {
            int c_idx = i;  // * numKernels + k;
            for (size_t j = 0; j < N; j += (N / 100)) {
                if (std::abs(C_h[c_idx][j] -
                             numRepeats * A_h[i][j] * A_h[i][j]) > 0.001) {
                    printf(
                        "Err: C[%d][%zu]: %f, numRepeats*A*A: %f, delta: %f\n",
                        c_idx, j, C_h[c_idx][j],
                        numRepeats * A_h[i][j] * A_h[i][j],
                        std::abs(C_h[c_idx][j] -
                                 numRepeats * A_h[i][j] * A_h[i][j]));
                    return -1;
                }
            }
        }
    }
    printf("Done with check\n");
*/

    free(A_h[i]);
    free(C_h[i]);

    return 0;
}
