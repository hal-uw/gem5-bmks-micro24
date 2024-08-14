#include "hip/hip_runtime.h"
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include <iostream>
#include <stdio.h>
#include <gem5/m5ops.h>

#define CHECK(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true)
{
    if(code != hipSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",hipGetErrorString(code),file, line);
        if(abort) exit(code);
    }
}

using namespace std;

#define SIZE    (100*1024*1024)


__global__ void histo_kernel( unsigned char *buffer,
        long size,
        unsigned int *histo ) {

    __shared__  unsigned int temp[256];

    temp[hipThreadIdx_x] = 0;
    __syncthreads();

    int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int offset = hipBlockDim_x * hipGridDim_x;
    while (i < size) {
        atomicAdd( &temp[buffer[i]], 1 );
        i += offset;
    }

    __syncthreads();
    atomicAdd( &(histo[hipThreadIdx_x]), temp[hipThreadIdx_x] );
}

int runHisto(char* file, unsigned int* freq, unsigned int memSize, unsigned int *source) {

    FILE *f = fopen(file,"rb");
    if (!f) {perror(file); exit(1);}
    fseek(f,0,SEEK_SET);
    size_t result = fread(source,1,memSize,f);
    if(result != memSize) fputs("Cannot read input file", stderr);

    fclose(f);

    unsigned char *buffer = (unsigned char*)source;

    hipDeviceProp_t  prop;
    ( hipGetDeviceProperties( &prop, 0 ) );
    int blocks = prop.multiProcessorCount;
    /*
      AMD supports multiple streams, don't need to check and not supported
    if(!prop.deviceOverlap)
    {
        cout << "No overlaps, so no speedup from streams" << endl;
        return 0;
    }
    */

    // allocate memory on the GPU for the file's data
    int partSize = memSize/32;
    int totalNum = memSize/sizeof(unsigned int);
    int partialNum = partSize/sizeof(unsigned int);

    unsigned char *dev_buffer0; 
    unsigned char *dev_buffer1;
    unsigned int *dev_histo;
    hipMalloc( (void**)&dev_buffer0, partSize ) ;
    hipMalloc( (void**)&dev_buffer1, partSize ) ;
    hipMalloc( (void**)&dev_histo,
            256 * sizeof( int ) ) ;
    hipMemset( dev_histo, 0,
            256 * sizeof( int ) ) ;
    hipStream_t stream0, stream1;
    hipStreamCreateWithFlags(&stream0, 0x01, -1);
    hipStreamCreateWithFlags(&stream1, 0x01, -1);
    hipEvent_t     start, stop;
    ( hipEventCreate( &start ) );
    ( hipEventCreate( &stop ) );
    ( hipEventRecord( start, 0 ) );
    int count = 1;
    for(int i = 0; i < totalNum; i+=partialNum*2)
    {

        CHECK(hipMemcpyAsync(dev_buffer0, buffer+i, partSize, hipMemcpyHostToDevice,stream0));
        CHECK(hipMemcpyAsync(dev_buffer1, buffer+i+partialNum, partSize, hipMemcpyHostToDevice,stream1));

        m5_getKernelArg(reinterpret_cast<uintptr_t>(dev_buffer0), reinterpret_cast<uintptr_t>(dev_histo), 0, 12, 2, count++);
        // kernel launch - 2x the number of mps gave best timing
        hipLaunchKernelGGL(histo_kernel, dim3(blocks*2), dim3(256), 0, stream0,  dev_buffer0, partSize, dev_histo );
        hipHccModuleRingDoorbell(stream0);
hipStreamSynchronize(stream0);
m5_dump_reset_stats(0, 0);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(dev_buffer1), reinterpret_cast<uintptr_t>(dev_histo), 0, 12, 2, count++);
        hipLaunchKernelGGL(histo_kernel, dim3(blocks*2), dim3(256), 0, stream1,  dev_buffer1, partSize, dev_histo );
        hipHccModuleRingDoorbell(stream1);
hipStreamSynchronize(stream1);
m5_dump_reset_stats(0, 0);
    }
    hipMemcpy( freq, dev_histo, 256 * sizeof( int ), hipMemcpyDeviceToHost );
    ( hipEventRecord( stop, 0 ) );
    ( hipEventSynchronize( stop ) );
    float   elapsedTime;
    ( hipEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );



    hipFree( dev_histo );
    hipFree( dev_buffer0 );
    hipFree( dev_buffer1 );
    return 0;
}
