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

#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <time.h>
#include <vector>
#include "hip/hip_runtime.h"

#include <gem5/m5ops.h>

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
vector_square(T *C_d, const T *A_d, size_t N, size_t numRepeats)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        for (size_t j = 0; j < numRepeats; j++) {
            C_d[i] += A_d[i] * A_d[i];
        }
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


__host__ void call_vector_square(unsigned blocks, unsigned threadsPerBlock, float* C_d, float* A_d, size_t N, size_t repeats, hipStream_t stream) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_d), reinterpret_cast<uintptr_t>(A_d), 0, 3, 2, 1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks), dim3(threadsPerBlock), 0, stream, 0, C_d, A_d, N, repeats);
}

__host__ void call_vector_add(unsigned blocks, unsigned threadsPerBlock, float* C_d, size_t N, hipStream_t stream, uint32_t lastKernel) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_d), 0, 0, 3, 1, 1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_add), dim3(blocks), dim3(threadsPerBlock), 0, stream, 0, C_d, N);
}

struct csvData
{
    float deadline; // in ms
    float delay; // in ms
    std::vector<int> kern_runtime; // in ms

};
std::ostream &operator<<(std::ostream &os, const csvData &_csvData) {
    os << "Deadline: " << _csvData.deadline <<  " Delay: " << _csvData.delay << " Kernel runtimes:";
    for (auto i : _csvData.kern_runtime) {
        os << " " << i;
    }
    return os;
}

static long spinloop(volatile long l) {
    while (l--);
    return 0;
}

void spin(float ms) {
    ms *= (83333. + (1./3.));
    spinloop(ms);
}

int main(int argc, char *argv[])
{
    assert(argc==3);
    // 1 second in ticks: 1000000000000
    // 1 ms in ticks: 1000000000
    const size_t one_sec = 1000000000000;
    const size_t one_ms = 1000000000;

    std::vector<csvData> stream_info;
    bool individualGpus = atoi(argv[2]);
    // Args: <csv file> <individual GPUs>

    // Parse the CSV file

    std::fstream file (std::string(argv[1]), std::ios::in);
    if (file.is_open()) {
        std::string line, field;
        while (std::getline(file, line)) {
            csvData dat;
            std::stringstream sstr(line);
            if (std::getline(sstr, field, ',')) {
                dat.deadline = std::stof(field);
            }
            if (std::getline(sstr, field, ',')) {
                dat.delay = std::stof(field);
            }
            while (std::getline(sstr, field, ',')) {
                dat.kern_runtime.push_back(std::stoi(field));
            }
            stream_info.push_back(dat);
        }
    }

    for (auto i : stream_info) {
        std::cout << i << std::endl;
    }

    int numGpus;
    hipGetDeviceCount(&numGpus);
    printf("Have %d GPUs\n", numGpus);

    int num_streams = stream_info.size();
    int num_kerns;
    for (auto i : stream_info) {
        num_kerns += i.kern_runtime.size();
    }

    hipStream_t hip_stream[num_streams];
    for (int i = 0; i < num_streams; i++) {
        if (individualGpus) {
            hipSetDevice(i % numGpus);
        }
        hipStreamCreateWithFlags(&hip_stream[i]); // , 0x01, (size_t)((double)stream_info[i].deadline * one_ms));
    }

    std::vector<float *> A_h (num_streams);
    std::vector<float *> C_h (num_kerns);
    // Tune these I guess
    size_t N = 98304;
    size_t Nbytes = N * sizeof(float);
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    #ifdef __HIP_PLATFORM_HCC__
      printf ("info: architecture on AMD GPU device is: %d\n",props.gcnArch);
    #endif
    printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);


    for (int i = 0; i < num_kerns; i++) {
        C_h[i] = (float *)calloc(N, sizeof(float));
        CHECK(C_h[i] == 0 ? hipErrorMemoryAllocation: hipSuccess);
    }

    for (size_t i = 0; i < num_streams; i++) {
        A_h[i] = (float *)malloc(Nbytes);
        CHECK(A_h[i] == 0 ? hipErrorMemoryAllocation : hipSuccess);
        for (size_t j = 0; j < N; j++) {
            A_h[i][j] = 1.618f + ((i*N + j) / (float)N);
        }
    }

    unsigned large_blocks[5] = {32, 64, 128, 256, 512};
    unsigned blocks[5] = {2, 4, 6, 8, 10};
    unsigned threadsPerBlock = 256;
    unsigned large_numRepeats[5] = {11, 45, 72, 100, 130};
    unsigned numRepeats[5] = {10, 55, 130, 195, 130};

    printf("Warm-up\n");

    hipStream_t warmup_stream;
    float *warmup_C = (float *)calloc(N, sizeof(float));
    float *warmup_A = (float *)malloc(Nbytes);
    for(int i = 0; i < N; i++){
        warmup_A[i] = 1.618f + i;
    }
    hipStreamCreateWithFlags(&warmup_stream); // , 0x01, -1);

    for(int i = 0; i < 4; i++) {
        call_vector_square(blocks[i], threadsPerBlock, warmup_C, warmup_A, N, numRepeats[i], warmup_stream);
    }

    // hipHccModuleRingDoorbell(warmup_stream);
    hipStreamSynchronize(warmup_stream);

    printf("Launching kernels\n");

    m5_dump_reset_stats(0, 0);
    //auto start = std::chrono::high_resolution_clock::now();
    int count = 0;
    for (int i = 0; i < num_streams; i++) {
        for (auto runtime : stream_info[i].kern_runtime) {
            call_vector_square(blocks[runtime-1], threadsPerBlock, C_h[count], A_h[i], N, numRepeats[runtime-1], hip_stream[i]);
            count++;
        }
    }

    // for (int i = 0; i < num_streams; i++) {
    //     spin(stream_info[i].delay);
    //     hipHccModuleRingDoorbell(hip_stream[i]);
    // }

    printf("After ring doorbell\n");

    for (int i = 0; i < num_streams; i++) {
        hipStreamSynchronize(hip_stream[i]);
    }
    //auto stop= std::chrono::high_resolution_clock::now();
    m5_dump_reset_stats(0, 0);

    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Kernel runtime: " << duration.count() << std::endl;
    printf("After synchronize\n");

    for (int i = 0; i < num_streams; i++)
        free(A_h[i]);
    for (int i = 0; i < num_kerns; i++)
        free(C_h[i]);
}
