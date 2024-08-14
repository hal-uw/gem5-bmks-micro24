/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA. Main file.
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */

#include "hip/hip_runtime.h"
size_t numStreams;
bool individualGpus;
hipStream_t *hip_stream;

#include "stdafx.h"
#include "hip_helpers.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"
#include <sys/time.h>
#include "vlc_kernel_sm64huff.h"
#include "scan.h"
#include "pack_kernels.h"
#include "cpuencode.h"
#include <gem5/m5ops.h>

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);

int main(int argc, char* argv[]){
  if(!InitHIP()) { return 0;   }
  unsigned int num_block_threads = 256;
  numStreams = atoi(argv[2]);
  individualGpus = atoi(argv[3]);
  hip_stream = new hipStream_t [numStreams]; 
  
  int numGpus;
  hipGetDeviceCount(&numGpus);

  for (int i = 0; i < numStreams; i++) {
    if (individualGpus) {
      hipSetDevice(i % numGpus);
    }
    hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1);
  }
  if (argc > 1) {
    for (int i=1; i<argc; i++) {
      runVLCTest(argv[i], num_block_threads);
    }
  }
  else {        runVLCTest(NULL, num_block_threads, 1024);      }
  hipDeviceReset();
  return 0;
}

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks) {
    printf("HIP! Starting VLC Tests!\n");
    unsigned int num_elements; //uint num_elements = num_blocks * num_block_threads; 
    unsigned int mem_size; //uint mem_size = num_elements * sizeof(int); 
    unsigned int symbol_type_size = sizeof(int);
    //////// LOAD DATA ///////////////
    double H; // entropy
    initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size, symbol_type_size);
    printf("Parameters: num_elements: %d, num_blocks: %d, num_block_threads: %d\n----------------------------\n", num_elements, num_blocks, num_block_threads);
    ////////LOAD DATA ///////////////
    uint        *sourceData =   (uint*) malloc(mem_size);
    uint        *destData   =   (uint*) malloc(mem_size);
    uint        *crefData   =   (uint*) malloc(mem_size);

    uint        *codewords         = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
    uint        *codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

    uint        *cw32 =         (uint*) malloc(mem_size);
    uint        *cw32len =      (uint*) malloc(mem_size);
    uint        *cw32idx =      (uint*) malloc(mem_size);

    uint        *cindex2=       (uint*) malloc(num_blocks*sizeof(int));

    memset(sourceData,   0, mem_size);
    memset(destData,     0, mem_size);
    memset(crefData,     0, mem_size);
    memset(cw32,         0, mem_size);
    memset(cw32len,      0, mem_size);
    memset(cw32idx,      0, mem_size);
    memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
    memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);
    memset(cindex2, 0, num_blocks*sizeof(int));
    //////// LOAD DATA ///////////////
    loadData(file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);

    //////// LOAD DATA ///////////////

    unsigned int        **d_sourceData, **d_destData, **d_destDataPacked;
    unsigned int        **d_codewords, **d_codewordlens;
    unsigned int        **d_cw32, **d_cw32len, **d_cw32idx, **d_cindex, **d_cindex2;

    hipHostMalloc((void**) &d_sourceData, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_sourceData[i],              mem_size);
    }   

    hipHostMalloc((void**) &d_destData, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_destData[i],              mem_size);
    }
    hipHostMalloc((void**) &d_destDataPacked, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_destDataPacked[i],          mem_size);
    }
    hipHostMalloc((void**) &d_codewords, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_codewords[i],               NUM_SYMBOLS*symbol_type_size);
    }

    hipHostMalloc((void**) &d_codewordlens, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_codewordlens[i],            NUM_SYMBOLS*symbol_type_size);
    }

    hipHostMalloc((void**) &d_cw32, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_cw32[i],                            mem_size);
    }

    hipHostMalloc((void**) &d_cw32len, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_cw32len[i],                         mem_size);
    }
    hipHostMalloc((void**) &d_cw32idx, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**) &d_cw32idx[i],                         mem_size);
    }
    hipHostMalloc((void**) &d_cindex, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**)&d_cindex[i],         num_blocks*sizeof(unsigned int));
    }
    hipHostMalloc((void**) &d_cindex2, sizeof(unsigned int*)*numStreams);
    for (int i = 0; i < numStreams; ++i) { 
        hipHostMalloc((void**)&d_cindex2[i],        num_blocks*sizeof(unsigned int));
    }

    for (int i = 0; i < numStreams; ++i) { 
        hipMemcpy(d_sourceData[i],              sourceData,             mem_size,               hipMemcpyHostToDevice);
        hipMemcpy(d_codewords[i],               codewords,              NUM_SYMBOLS*symbol_type_size,   hipMemcpyHostToDevice);
        hipMemcpy(d_codewordlens[i],    codewordlens,   NUM_SYMBOLS*symbol_type_size,   hipMemcpyHostToDevice);
        hipMemcpy(d_destData[i],                destData,               mem_size,               hipMemcpyHostToDevice);
    }
    dim3 grid_size(num_blocks,1,1);
    dim3 block_size(num_block_threads, 1, 1);
    unsigned int sm_size; 

    int NT = 10; //number of runs for each execution time

    //////////////////* CPU ENCODER *///////////////////////////////////
    unsigned int refbytesize;
    long long timer = get_time();
    cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
    float msec = (float)((get_time() - timer)/1000.0);
    printf("CPU Encoding time (CPU): %f (ms)\n", msec);
    printf("CPU Encoded to %d [B]\n", refbytesize);
    unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);
    //////////////////* END CPU *///////////////////////////////////

    //////////////////* SM64HUFF KERNEL *///////////////////////////////////
    grid_size.x         = num_blocks;
    block_size.x        = num_block_threads;
    sm_size                     = block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
    sm_size                     = 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
    hipEvent_t     start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    int count = 1;
    hipEventRecord( start, 0 );
    for (int i=0; i<NT; i++) {
      
        for (int j = 0; j < numStreams; ++j) { 
        m5_getKernelArg(reinterpret_cast<uintptr_t>(d_sourceData[j]), reinterpret_cast<uintptr_t>(d_codewords[j]), reinterpret_cast<uintptr_t>(d_codewordlens[j]), 0, 3, count);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(d_destData[j]), reinterpret_cast<uintptr_t>(d_cindex[j]), 0, 15, 2, count++);
        hipLaunchKernelGGL(vlc_encode_kernel_sm64huff, dim3(grid_size), dim3(block_size),  /*sm_size,*/ 0, hip_stream[j], /*0,*/ d_sourceData[j], d_codewords[j], d_codewordlens[j],  
#ifdef TESTING
                         d_cw32[j], d_cw32len[j], d_cw32idx[j], 
#endif
                         d_destData[j], d_cindex[j]); //testedOK2
        }
       for (int j = 0; j < numStreams; ++j) { 
        hipHccModuleRingDoorbell(hip_stream[j]);
        hipStreamSynchronize(hip_stream[j]);
        m5_dump_reset_stats(0, 0);
        }
    }
    // hipDeviceSynchronize();
    hipEventRecord( stop, 0 ) ;
    hipEventSynchronize( stop ) ;
    float   elapsedTime;
    hipEventElapsedTime( &elapsedTime,
            start, stop ) ;

    CUT_CHECK_ERROR("Kernel execution failed\n");
    printf("GPU Encoding time (SM64HUFF): %f (ms)\n", elapsedTime/NT);
    //////////////////* END KERNEL *///////////////////////////////////

#ifdef TESTING
    unsigned int num_scan_elements = grid_size.x;
    preallocBlockSums(num_scan_elements);
    hipMemset(d_destDataPacked, 0, mem_size);
    printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);
    prescanArray(d_cindex2, d_cindex, num_scan_elements);

        for (int j = 0; j < numStreams; ++j) { 
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_destData[j]), reinterpret_cast<uintptr_t>(d_cindex[j]), reinterpret_cast<uintptr_t>(d_cindex2[j]), 0, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_destDataPacked[j]), 0, 0, 3, 1, count++);
    hipLaunchKernelGGL(pack2, dim3(num_scan_elements/16), dim3(16), 0, hip_stream[j], /*0,*/ (unsigned int*)d_destData[j], d_cindex[j], d_cindex2[j], (unsigned int*)d_destDataPacked[j], num_elements/num_scan_elements);
        }
        for (int j = 0; j < numStreams; ++j) { 
            hipHccModuleRingDoorbell(hip_stream[j]);
        hipStreamSynchronize(hip_stream[j]);
        m5_dump_reset_stats(0, 0);
        }
    CUT_CHECK_ERROR("Pack2 Kernel execution failed\n");
    deallocBlockSums();

    hipMemcpy(destData, d_destDataPacked[0], mem_size, hipMemcpyDeviceToHost);
    compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
#endif 

    free(sourceData);
    free(destData);
    free(codewords);
    free(codewordlens);
    free(cw32);
    free(cw32len);
    free(crefData); 
        for (int j = 0; j < numStreams; ++j) { 
    hipFree(d_sourceData[j]);
    hipFree(d_destData[j]);
    hipFree(d_destDataPacked[j]);
    hipFree(d_codewords[j]);
    hipFree(d_codewordlens[j]);
    hipFree(d_cw32[j]);
    hipFree(d_cw32len[j]);
    hipFree(d_cw32idx[j]);
    hipFree(d_cindex[j]);
    hipFree(d_cindex2[j]);
        }
    free(cindex2);
}

