#include "hip/hip_runtime.h"
/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _PRESCAN_H_
#define _PRESCAN_H_

// includes, kernels
#include "scanLargeArray_kernel.h"
#include <assert.h>
#include <stdio.h>
#include "hiputil.h"
#include <gem5/m5ops.h>

inline bool
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

static unsigned int*** g_scanBlockSums;
static unsigned int g_numEltsAllocated = 0;
static unsigned int g_numLevelsAllocated = 0;

static void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    //g_scanBlockSums = (unsigned int***) malloc(level * sizeof(unsigned int**));
    CUDA_SAFE_CALL(hipHostMalloc((void***) &g_scanBlockSums, sizeof(unsigned int**)*numStreams));
    //hipHostMalloc((void**) &g_scanBlockSums, sizeof(unsigned int**)* level);
    //g_numLevelsAllocated = level;
    //numElts = maxNumElements;
    //level = 0;
    
    for (int j = 0; j < numStreams; ++j) {
    g_numLevelsAllocated = level;
    numElts = maxNumElements;
    level = 0;
    CUDA_SAFE_CALL(hipHostMalloc((void**) &g_scanBlockSums[j], sizeof(unsigned int*)*level));
    do {       
        unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) { 
        //g_scanBlockSums[level] = (unsigned int**) malloc(numStreams * sizeof(unsigned int*));
        //hipHostMalloc((void**) &g_scanBlockSums[level], sizeof(unsigned int*)*numStreams);
            printf("%p\n", &g_scanBlockSums[j][level]);
            CUDA_SAFE_CALL(hipHostMalloc((void**) &g_scanBlockSums[j][level], numBlocks * sizeof(unsigned int)));
            level ++;
        }
        //hipHostMalloc((void**) &g_scanBlockSums[j][level++], numBlocks * sizeof(unsigned int));
        numElts = numBlocks;
    } while (numElts > 1);
    }

    CUT_CHECK_ERROR("preallocBlockSums");
}

static void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
    {
        for (int j = 0; j < numStreams; ++j) {
            hipFree(g_scanBlockSums[i][j]);
        }
    }

    CUT_CHECK_ERROR("deallocBlockSums");
    
    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}

static void prescanArrayRecursive(unsigned int **outArray, 
                           unsigned int **inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(unsigned int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(unsigned int) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // make sure there are no CUDA errors before we start
    CUT_CHECK_ERROR("prescanArrayRecursive before kernels");
    int count = 1;
    // execute the scan
    if (numBlocks > 1)
    {
        for (int j = 0; j < numStreams; ++j) {
        m5_getKernelArg(reinterpret_cast<uintptr_t>(outArray[j]), reinterpret_cast<uintptr_t>(inArray[j]), 0, 3, 2, count++);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prescan<true, false>), dim3(grid), dim3(threads), sharedMemSize, /*0,*/ hip_stream[j], outArray[j], 
                                                                 inArray[j], 
                                                                 g_scanBlockSums[level][j],
                                                                 numThreads * 2, 0, 0);
        }
        for (int j = 0; j < numStreams; ++j) {
            hipHccModuleRingDoorbell(hip_stream[j]);
            hipStreamSynchronize(hip_stream[j]);
            m5_dump_reset_stats(0, 0);
        }

        CUT_CHECK_ERROR("prescanWithBlockSums");
        if (np2LastBlock)
        {
            for (int j = 0; j < numStreams; ++j) {
                m5_getKernelArg(reinterpret_cast<uintptr_t>(outArray[j]), reinterpret_cast<uintptr_t>(inArray[j]), 0, 3, 2, count++);
                hipLaunchKernelGGL(HIP_KERNEL_NAME(prescan<true, true>), dim3(1), dim3(numThreadsLastBlock), /*sharedMemLastBlock*/ 0 , hip_stream[j], outArray[j], inArray[j], g_scanBlockSums[level][j], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
        }
            for (int j = 0; j < numStreams; ++j) {
                hipHccModuleRingDoorbell(hip_stream[j]);
                hipStreamSynchronize(hip_stream[j]);
                m5_dump_reset_stats(0, 0);
            }

            CUT_CHECK_ERROR("prescanNP2WithBlockSums");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        for (int j = 0; j < numStreams; ++j) {
        m5_getKernelArg(reinterpret_cast<uintptr_t>(outArray[j]), 0, 0, 3, 1, count++);
        hipLaunchKernelGGL(uniformAdd, dim3(grid), dim3(threads ), 0, hip_stream[j], /*0,*/ outArray[j], 
                                        g_scanBlockSums[level][j], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        }
        
        for (int j = 0; j < numStreams; ++j) {
            hipHccModuleRingDoorbell(hip_stream[j]);
                hipStreamSynchronize(hip_stream[j]);
                m5_dump_reset_stats(0, 0);
            }
        CUT_CHECK_ERROR("uniformAdd");
        if (np2LastBlock)
        {
        for (int j = 0; j < numStreams; ++j) {
            m5_getKernelArg(reinterpret_cast<uintptr_t>(outArray[j]), 0, 0, 3, 1, count++);
            hipLaunchKernelGGL(uniformAdd, dim3(1), dim3(numThreadsLastBlock ), 0, hip_stream[j], /*0,*/ outArray[j], 
                                                     g_scanBlockSums[level][j], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
        }
         for (int j = 0; j < numStreams; ++j) {
            hipHccModuleRingDoorbell(hip_stream[j]);
                hipStreamSynchronize(hip_stream[j]);
                m5_dump_reset_stats(0, 0);
            }
            CUT_CHECK_ERROR("uniformAdd");
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        for (int j = 0; j < numStreams; ++j) {
            m5_getKernelArg(reinterpret_cast<uintptr_t>(outArray[j]), reinterpret_cast<uintptr_t>(inArray[j]), 0, 3, 2, count++);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(prescan<false, false>), dim3(grid), dim3(threads), sharedMemSize , hip_stream[j], outArray[j], inArray[j],
                                                                  (unsigned int*)NULL, numThreads * 2, 0, 0);
        }
        for (int j = 0; j < numStreams; ++j) {
            hipHccModuleRingDoorbell(hip_stream[j]);
                hipStreamSynchronize(hip_stream[j]);
                m5_dump_reset_stats(0, 0);
            }
        CUT_CHECK_ERROR("prescan");
    }
    else
    {
        for (int j = 0; j < numStreams; ++j) {
            m5_getKernelArg(reinterpret_cast<uintptr_t>(outArray[j]), reinterpret_cast<uintptr_t>(inArray[j]), 0, 3, 2, count++);
         hipLaunchKernelGGL(HIP_KERNEL_NAME(prescan<false, true>), dim3(grid), dim3(threads), sharedMemSize, hip_stream[j], outArray[j], inArray[j], 
                                                                  (unsigned int*)NULL, numElements, 0, 0);
        }
        for (int j = 0; j < numStreams; ++j) {
            hipHccModuleRingDoorbell(hip_stream[j]);
                hipStreamSynchronize(hip_stream[j]);
                m5_dump_reset_stats(0, 0);
        }
         CUT_CHECK_ERROR("prescanNP2");
    }
}

static void prescanArray(unsigned int **outArray, unsigned int **inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}

#endif // _PRESCAN_H_
