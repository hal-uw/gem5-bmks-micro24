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
  knode **knodesD;
  hipHostMalloc((void**)&knodesD, sizeof(knode*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&knodesD[i], knodes_mem);
  }
  checkHIPError("hipHostMalloc  recordsD");

  //==================================================50
  //    currKnodeD
  //==================================================50
  long **currKnodeD;
  hipHostMalloc((void**)&currKnodeD, sizeof(long*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&currKnodeD[i], count*sizeof(long));
  }
  checkHIPError("hipHostMalloc  currKnodeD");

  //==================================================50
  //    offsetD
  //==================================================50
  long **offsetD;
  hipHostMalloc((void**)&offsetD, sizeof(long*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&offsetD[i], count*sizeof(long));
  }
  checkHIPError("hipHostMalloc  offsetD");

  //==================================================50
  //    lastKnodeD
  //==================================================50
  long **lastKnodeD;
  hipHostMalloc((void**)&lastKnodeD, sizeof(long*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&lastKnodeD[i], count*sizeof(long));
  }
  checkHIPError("hipHostMalloc  lastKnodeD");

  //==================================================50
  //    offset_2D
  //==================================================50
  long **offset_2D;
  hipHostMalloc((void**)&offset_2D, sizeof(long*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&offset_2D[i], count*sizeof(long));
  }
  checkHIPError("hipHostMalloc  offset_2D");

  //==================================================50
  //    startD
  //==================================================50
  int **startD;
  hipHostMalloc((void**)&startD, sizeof(int*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&startD[i], count*sizeof(int));
  }
  checkHIPError("hipHostMalloc startD");

  //==================================================50
  //    endD
  //==================================================50
  int **endD;
  hipHostMalloc((void**)&endD, sizeof(int*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&endD[i], count*sizeof(int));
  }
  checkHIPError("hipHostMalloc endD");

  //====================================================================================================100
  //    DEVICE IN/OUT
  //====================================================================================================100

  //==================================================50
  //    ansDStart
  //==================================================50
  int **ansDStart;
  hipHostMalloc((void**)&ansDStart, sizeof(int*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&ansDStart[i], count*sizeof(int));
  }
  checkHIPError("hipHostMalloc ansDStart");

  //==================================================50
  //    ansDLength
  //==================================================50
  int **ansDLength;
  hipHostMalloc((void**)&ansDLength, sizeof(int*)*numStreams);
  for (int i = 0; i < numStreams; ++i) { 
    hipHostMalloc((void**)&ansDLength[i], count*sizeof(int));
  }
  checkHIPError("hipHostMalloc ansDLength");

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

  for (int i = 0; i < numStreams; ++i) { 
    hipMemcpy(knodesD[i], knodes, knodes_mem, hipMemcpyHostToDevice);
    checkHIPError("hipHostMalloc hipMemcpy memD");

    //==================================================50
    //    currKnodeD
    //==================================================50
    hipMemcpy(currKnodeD[i], currKnode, count*sizeof(long), hipMemcpyHostToDevice);
    checkHIPError("hipHostMalloc hipMemcpy currKnodeD");

    //==================================================50
    //    offsetD
    //==================================================50
    hipMemcpy(offsetD[i], offset, count*sizeof(long), hipMemcpyHostToDevice);
    checkHIPError("hipHostMalloc hipMemcpy offsetD");

    //==================================================50
    //    lastKnodeD
    //==================================================50
    hipMemcpy(lastKnodeD[i], lastKnode, count*sizeof(long), hipMemcpyHostToDevice);
    checkHIPError("hipHostMalloc hipMemcpy lastKnodeD");

    //==================================================50
    //    offset_2D
    //==================================================50
    hipMemcpy(offset_2D[i], offset_2, count*sizeof(long), hipMemcpyHostToDevice);
    checkHIPError("hipHostMalloc hipMemcpy offset_2D");

    //==================================================50
    //    startD
    //==================================================50
    hipMemcpy(startD[i], start, count*sizeof(int), hipMemcpyHostToDevice);
    checkHIPError("hipMemcpy startD");

    //==================================================50
    //    endD
    //==================================================50
    hipMemcpy(endD[i], end, count*sizeof(int), hipMemcpyHostToDevice);
    checkHIPError("hipMemcpy endD");

    //====================================================================================================100
    //    DEVICE IN/OUT
    //====================================================================================================100

    //==================================================50
    //    ansDStart
    //==================================================50
    hipMemcpy(ansDStart[i], recstart, count*sizeof(int), hipMemcpyHostToDevice);
    checkHIPError("hipMemcpy ansDStart");

    //==================================================50
    //    ansDLength
    //==================================================50
    hipMemcpy(ansDLength[i], reclength, count*sizeof(int), hipMemcpyHostToDevice);
    checkHIPError("hipMemcpy ansDLength");
  }
  time3 = get_time();

  //======================================================================================================================================================150
  //    KERNEL
  //======================================================================================================================================================150
  int countKernel = 1;
  // [GPU] findRangeK kernel
  for (int i = 0; i < numStreams; ++i) { 
  m5_getKernelArg(reinterpret_cast<uintptr_t>(knodesD[i]), reinterpret_cast<uintptr_t>(currKnodeD[i]), reinterpret_cast<uintptr_t>(offsetD[i]), 60, 3, countKernel);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(offset_2D[i]), reinterpret_cast<uintptr_t>(startD[i]), reinterpret_cast<uintptr_t>(endD[i]), 48, 3, countKernel);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(ansDLength[i]),reinterpret_cast<uintptr_t>(lastKnodeD[i]), reinterpret_cast<uintptr_t>(ansDStart[i]), 51, 3, countKernel++);
  hipLaunchKernelGGL(findRangeK, dim3(numBlocks), dim3(threadsPerBlock), 0, hip_stream[i], /*0,*/    maxheight,
                                                knodesD[i],
                                                knodes_elem,

                                                currKnodeD[i],
                                                offsetD[i],
                                                lastKnodeD[i],
                                                offset_2D[i],
                                                startD[i],
                                                endD[i],
                                                ansDStart[i],
                                                ansDLength[i]);
  }
  
  for (int i = 0; i < numStreams; ++i) {
    hipHccModuleRingDoorbell(hip_stream[i]); 
    hipStreamSynchronize(hip_stream[i]);
    m5_dump_reset_stats(0, 0);
  }
  // hipDeviceSynchronize();
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

  hipMemcpy(recstart, ansDStart[0], count*sizeof(int), hipMemcpyDeviceToHost);
  checkHIPError("hipMemcpy ansDStart");

  //==================================================50
  //    ansDLength
  //==================================================50

  hipMemcpy(reclength, ansDLength[0], count*sizeof(int), hipMemcpyDeviceToHost);
  checkHIPError("hipMemcpy ansDLength");

  time5 = get_time();

  //======================================================================================================================================================150
  //    GPU MEMORY DEALLOCATION
  //======================================================================================================================================================150
  for (int i = 0; i < numStreams; ++i) { 
  hipFree(knodesD[i]);

  hipFree(currKnodeD[i]);
  hipFree(offsetD[i]);
  hipFree(lastKnodeD[i]);
  hipFree(offset_2D[i]);
  hipFree(startD[i]);
  hipFree(endD[i]);
  hipFree(ansDStart[i]);
  hipFree(ansDLength[i]);
  }
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
