#ifndef KERNEL_GPU_HIP_WRAPPER_2_HIP_H_
#define KERNEL_GPU_HIP_WRAPPER_2_HIP_H_

/*
#ifdef __cplusplus
extern "C" {
#endif
*/

//========================================================================================================================================================================================================200
//      INCLUDE
//========================================================================================================================================================================================================200
#include "hip/hip_runtime.h"
#include <gem5/m5ops.h>

//======================================================================================================================================================150
//      COMMON
//======================================================================================================================================================150
//#include "../common.h"                                                     // (in the main program folder) needed to recognized input parameters

//======================================================================================================================================================150
//      UTILITIES
//======================================================================================================================================================150
// (in library path specified to compiler)      needed by for device functions
#include "../util/timer/timer.h"                                           // (in library path specified to compiler)      needed by timer
#include "../util/hip/hip.h"

//======================================================================================================================================================150
//      KERNEL
//======================================================================================================================================================150
#include "./kernel_gpu_hip_2.h"                                            // (in the current directory)   GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//======================================================================================================================================================150
//      HEADER
//======================================================================================================================================================150
#include "./kernel_gpu_hip_wrapper_2.h"                                    // (in the current directory)

//========================================================================================================================================================================================================200
//      FUNCTION
//========================================================================================================================================================================================================200
void 
kernel_gpu_hip_wrapper_2(knode *knodes,
                          long knodes_elem,
                          long knodes_mem,

                          int order,
                          long maxheight,
                          int count,

                          long *currKnode,
                          long *offset,
                          long *lastKnode,
                          long *offset_2,
                          int *start,
                          int *end,
                          int *recstart,
                          int *reclength)
{

  //======================================================================================================================================================150
  //    CPU VARIABLES
  //======================================================================================================================================================150

  // timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time();

  //======================================================================================================================================================150
  //    GPU SETUP
  //======================================================================================================================================================150

  //====================================================================================================100
  //    INITIAL DRIVER OVERHEAD
  //====================================================================================================100
  hipDeviceSynchronize();

  //====================================================================================================100
  //    EXECUTION PARAMETERS
  //====================================================================================================100
  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

  time1 = get_time();

  //======================================================================================================================================================150
  //    GPU MEMORY                              MALLOC
  //======================================================================================================================================================150

  //====================================================================================================100
  //    DEVICE IN
  //====================================================================================================100

  //==================================================50
  //    knodesD
  //==================================================50
  knode *knodesD = (knode *) malloc (knodes_mem);

  //==================================================50
  //    currKnodeD
  //==================================================50
  long *currKnodeD = (long *) malloc (count*sizeof(long));

  //==================================================50
  //    offsetD
  //==================================================50
  long *offsetD = (long *) malloc (count*sizeof(long));

  //==================================================50
  //    lastKnodeD
  //==================================================50
  long *lastKnodeD= (long *) malloc (count*sizeof(long)) ;

  //==================================================50
  //    offset_2D
  //==================================================50
  long *offset_2D= (long *) malloc (count*sizeof(long)) ;

  //==================================================50
  //    startD
  //==================================================50
  int *startD= (int *) malloc (count*sizeof(int));

  //==================================================50
  //    endD
  //==================================================50
  int *endD= (int *) malloc (count*sizeof(int));

  //====================================================================================================100
  //    DEVICE IN/OUT
  //====================================================================================================100

  //==================================================50
  //    ansDStart
  //==================================================50
  int *ansDStart= (int *) malloc (count*sizeof(int));
  // hipHostMalloc((void**)&ansDStart, count*sizeof(int));
  // checkHIPError("hipHostMalloc ansDStart");

  //==================================================50
  //    ansDLength
  //==================================================50
  int *ansDLength= (int *) malloc (count*sizeof(int));
  // hipHostMalloc((void**)&ansDLength, count*sizeof(int));
  // checkHIPError("hipHostMalloc ansDLength");

  time2 = get_time();

  //======================================================================================================================================================150
  //    GPU MEMORY                      COPY
  //======================================================================================================================================================150

  //====================================================================================================100
  //    DEVICE IN
  //====================================================================================================100

  //==================================================50
  //    knodesD
  //==================================================50
  memcpy(knodesD, knodes, knodes_mem);

  //==================================================50
  //    currKnodeD
  //==================================================50
  memcpy(currKnodeD, currKnode, count*sizeof(long));

  //==================================================50
  //    offsetD
  //==================================================50
  memcpy(offsetD, offset, count*sizeof(long));

  //==================================================50
  //    lastKnodeD
  //==================================================50
  memcpy(lastKnodeD, lastKnode, count*sizeof(long));

  //==================================================50
  //    offset_2D
  //==================================================50
  memcpy(offset_2D, offset_2, count*sizeof(long));

  //==================================================50
  //    startD
  //==================================================50
  memcpy(startD, start, count*sizeof(int));

  //==================================================50
  //    endD
  //==================================================50
  memcpy(endD, end, count*sizeof(int));

  //====================================================================================================100
  //    DEVICE IN/OUT
  //====================================================================================================100

  //==================================================50
  //    ansDStart
  //==================================================50
  memcpy(ansDStart, recstart, count*sizeof(int));

  //==================================================50
  //    ansDLength
  //==================================================50
  memcpy(ansDLength, reclength, count*sizeof(int));

  time3 = get_time();

  //======================================================================================================================================================150
  //    KERNEL
  //======================================================================================================================================================150

  // [GPU] findRangeK kernel
  m5_getKernelArg(reinterpret_cast<uintptr_t>(knodesD), reinterpret_cast<uintptr_t>(currKnodeD), reinterpret_cast<uintptr_t>(offsetD), 60, 3, 1);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(offset_2D), reinterpret_cast<uintptr_t>(startD), reinterpret_cast<uintptr_t>(endD), 3, 3, 1);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(ansDLength),reinterpret_cast<uintptr_t>(lastKnodeD), reinterpret_cast<uintptr_t>(ansDStart), 63, 3, 1);
  hipLaunchKernelGGL(findRangeK, dim3(numBlocks), dim3(threadsPerBlock), 0, 0,    maxheight,
                                                knodesD, //r
                                                knodes_elem,

                                                currKnodeD, //w
                                                offsetD, // w
                                                lastKnodeD, //w
                                                offset_2D, // w
                                                startD, //r 
                                                endD, //r 
                                                ansDStart, //w
                                                ansDLength); //w
  hipDeviceSynchronize();
  checkHIPError("findRangeK");

  time4 = get_time();

  //======================================================================================================================================================150
  //    GPU MEMORY                      COPY (CONTD.)
  //======================================================================================================================================================150

  //====================================================================================================100
  //    DEVICE IN/OUT
  //====================================================================================================100

  //==================================================50
  //    ansDStart
  //==================================================50

  memcpy(recstart, ansDStart, count*sizeof(int));
  checkHIPError("memcpy ansDStart");

  //==================================================50
  //    ansDLength
  //==================================================50

  memcpy(reclength, ansDLength, count*sizeof(int));
  checkHIPError("memcpy ansDLength");

  time5 = get_time();

  //======================================================================================================================================================150
  //    GPU MEMORY DEALLOCATION
  //======================================================================================================================================================150
  free(knodesD);

  free(currKnodeD);
  free(offsetD);
  free(lastKnodeD);
  free(offset_2D);
  free(startD);
  free(endD);
  free(ansDStart);
  free(ansDLength);

  time6 = get_time();

  //======================================================================================================================================================150
  //    DISPLAY TIMING
  //======================================================================================================================================================150
  printf("Time spent in different stages of GPU_HIP KERNEL:\n");

  printf("%15.12f s, %15.12f percent : GPU: SET DEVICE / DRIVER INIT\n",
	 (float) (time1-time0) / 1000000,
	 (float) (time1-time0) / (float) (time6-time0) * 100);
  printf("%15.12f s, %15.12f percent : GPU MEM: ALO\n",
	 (float) (time2-time1) / 1000000,
	 (float) (time2-time1) / (float) (time6-time0) * 100);
  printf("%15.12f s, %15.12f percent : GPU MEM: COPY IN\n",
	 (float) (time3-time2) / 1000000,
	 (float) (time3-time2) / (float) (time6-time0) * 100);

  printf("%15.12f s, %15.12f percent : GPU: KERNEL\n",
	 (float) (time4-time3) / 1000000,
	 (float) (time4-time3) / (float) (time6-time0) * 100);

  printf("%15.12f s, %15.12f percent : GPU MEM: COPY OUT\n",
	 (float) (time5-time4) / 1000000,
	 (float) (time5-time4) / (float) (time6-time0) * 100);
  printf("%15.12f s, %15.12f percent : GPU MEM: FRE\n",
	 (float) (time6-time5) / 1000000,
	 (float) (time6-time5) / (float) (time6-time0) * 100);

  printf("Total time:\n");
  printf("%.12f s\n", (float) (time6-time0) / 1000000);

}

  //========================================================================================================================================================================================================200
  //    END
  //========================================================================================================================================================================================================200

/*
#ifdef __cplusplus
}
#endif
*/

#endif
