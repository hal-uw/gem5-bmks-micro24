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

#ifndef _SCAN_BEST_KERNEL_H_
#define _SCAN_BEST_KERNEL_H_

#include "hip/hip_runtime.h"

// Define this to more rigorously avoid bank conflicts, 
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance 
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS 

// 32 banks of LDS
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// http://www.cs.unc.edu/~prins/Classes/203/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// excellent paper "Prefix sums and their applications".
// http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//

template <bool isNP2>
__device__ static void loadSharedChunkFromMem(unsigned int *s_data,
                                       const unsigned int *g_idata,
                                       int n, int baseIndex,
                                       int& ai, int& bi, 
                                       int& mem_ai, int& mem_bi, 
                                       int& bankOffsetA, int& bankOffsetB)
{
    int thid = hipThreadIdx_x;
    mem_ai = baseIndex + hipThreadIdx_x;
    mem_bi = mem_ai + hipBlockDim_x;

    ai = thid;
    bi = thid + hipBlockDim_x;

    // compute spacing to avoid bank conflicts
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    // pad values beyond n with zeros
    s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
    
    if (isNP2) // compile-time decision
    {
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    }
    else
    {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}

template <bool isNP2>
__device__ static void storeSharedChunkToMem(unsigned int* g_odata, 
                                      const unsigned int* s_data,
                                      int n, 
                                      int ai, int bi, 
                                      int mem_ai, int mem_bi,
                                      int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    // write results to global memory
    g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
    if (isNP2) // compile-time decision
    {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
    else
    {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}

template <bool storeSum>
__device__ static void clearLastElement(unsigned int* s_data, 
                                 unsigned int *g_blockSums, 
                                 int blockIndex)
{
    if (hipThreadIdx_x == 0)
    {
        int index = (hipBlockDim_x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        
        if (storeSum) // compile-time decision
        {
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}

__device__ static unsigned int buildSum(unsigned int *s_data)
{
    int thid = hipThreadIdx_x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = hipBlockDim_x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
	    int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__ static void scanRootToLeaves(unsigned int *s_data, unsigned int stride)
{
    int thid = hipThreadIdx_x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= (int)hipBlockDim_x; d *= 2)
    {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
	    int i  = __mul24(__mul24(2, stride), thid);
	    int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum>
__device__ static void prescanBlock(unsigned int *data, int blockIndex, unsigned int *blockSums)
{
    int stride = buildSum(data);               // build the sum in place up the tree
    clearLastElement<storeSum>(data, blockSums, 
                               (blockIndex == 0) ? hipBlockIdx_x : blockIndex);
    scanRootToLeaves(data, stride);            // traverse down tree to build the scan 
}

template <bool storeSum, bool isNP2>
__global__ void prescan(unsigned int *g_odata, 
                        const unsigned int *g_idata, 
                        unsigned int *g_blockSums, 
                        int n, 
                        int blockIndex, 
                        int baseIndex)
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    HIP_DYNAMIC_SHARED( unsigned int, s_data)

    // load data into shared memory
    loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, 
                                  (baseIndex == 0) ? 
                                  __mul24(hipBlockIdx_x, (hipBlockDim_x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi, 
                                  bankOffsetA, bankOffsetB);
    // scan the data in each block
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    // write results to device memory
    storeSharedChunkToMem<isNP2>(g_odata, s_data, n, 
                                 ai, bi, mem_ai, mem_bi, 
                                 bankOffsetA, bankOffsetB);  
}

__global__ void uniformAdd(unsigned int *g_data, 
                           unsigned int *uniforms, 
                           int n, 
                           int blockOffset, 
                           int baseIndex)
{
    __shared__ unsigned int uni;
    if (hipThreadIdx_x == 0) {
        uni = uniforms[hipBlockIdx_x + blockOffset];
    }

    unsigned int address = __mul24(hipBlockIdx_x, (hipBlockDim_x << 1)) + baseIndex + hipThreadIdx_x; 

    __syncthreads();
    
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + hipBlockDim_x] += ((int)hipThreadIdx_x + (int)hipBlockDim_x < n) * uni;
}

#endif // #ifndef _SCAN_BEST_KERNEL_H_
