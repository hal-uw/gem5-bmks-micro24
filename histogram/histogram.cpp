#include <hip/hip_runtime.h>
#include <stdio.h>
#include "randoms.h"
#include <math.h>
#include <time.h>
#include<iostream>
#include<string.h>
#include<stdlib.h>
#include <gem5/m5ops.h>

using namespace std;

    
 __global__ void kernel_getHist(int* array, long size,  int* histo, int buckets)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid>=size)   return;

     int value = array[tid];

    int bin = value % buckets;

    atomicAdd(&histo[bin],1);
    //__syncthreads();
}

  __global__ void CalHistKernel(int* array , long size,  int* histo ,int buckets)
{
extern __shared__ int _bins[];

int tx = hipThreadIdx_x;
int idx = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;//hipBlockDim_y=1

if(tx< hipBlockDim_x)
{

    for (int i = 0; i < buckets/hipBlockDim_x ; i++)
     _bins[i*hipBlockDim_x+tx]=0;     

}
__syncthreads();

if(idx<size)
{       
    atomicAdd((int*)&_bins[array[idx] % buckets],1);     

}
__syncthreads();
for (int i = 0; i < buckets/hipBlockDim_x; i++)
atomicAdd((int*)&histo[i*hipBlockDim_x+tx],_bins[i*hipBlockDim_x+tx]);
}

void histogram256CPU(int *h_Histogram, int *histo, long long size, int buckets)
 {

     for (long long i = 0; i < buckets ; i++)
     {

          histo[i] = 0;

     }

     printf("\n \n Running CPU Function \n \n ");
     for (long long i = 0; i < size ; i++)
     {
        int bin = h_Histogram[i] % buckets;

          histo[bin]++;

     }


 }




int main(int argc, char *argv[]) {
    
    int buckets = 256;
    int seed = 1;
    
    long size = 3840 * 2160;      // 4k
   // long size = 2048*1080;    // 4k
  // long size = 512;
    const int threadsPerBlock = 256;
    int *hA = new int[size];
    int *hB = new int[buckets];
    int *histo = new int[buckets];
    
    random_ints(hA, 0, 10000, size, seed);
    random_ints(hB, 0, 0,  buckets, seed); 
/*

     for (long long i = 0; i < size ; i++)
     {
               hA[i] = 2;
               
     }
*/
    
    
       int* dArray;
    hipMalloc(&dArray,size * sizeof(int));
    hipMemcpy(dArray,hA,size * sizeof(int) ,hipMemcpyHostToDevice);

     int* dHist;
    hipMalloc(&dHist,buckets * sizeof(int));
    hipMemset(dHist,0,buckets * sizeof(int));

    dim3 block(threadsPerBlock);
    dim3 grid((size + block.x - 1)/block.x);
   hipStream_t hip_stream;
    hipStreamCreateWithFlags(&hip_stream, 0x01, -1);
    int count = 257;
    m5_dump_reset_stats(0, 0);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(dArray), reinterpret_cast<uintptr_t>(dHist), 0, 12, 2, count++);
   hipLaunchKernelGGL((CalHistKernel), dim3(grid), dim3(block), buckets * sizeof(int), 0, dArray, size, dHist, buckets);

    hipHccModuleRingDoorbell(hip_stream);
    hipStreamSynchronize(hip_stream);
	m5_dump_reset_stats(0, 0);
   
   hipDeviceSynchronize();
    hipMemcpy(histo,dHist,buckets * sizeof(int),hipMemcpyDeviceToHost);

    histogram256CPU(hA, hB, size, buckets);
   
   
   

    
    printf("\n Final histogram Value \n \n");
    
    for (int j = 0 ; j < buckets; j++){
        //printf(" %d, ", hB[j]);
        if (histo[j] != hB[j])
        { 
            printf(" \n \n , ");
            printf("Failed at index = %d \n", j);
            printf("CPU value = %d \t", hB[j]);
            printf("GPU value = %d \n", histo[j]);
            //return 0;
        
        }
    }
            
    printf("\n Passed \n");


   hipFree(dArray);
   hipFree(dHist);
   delete[] hA;
   delete[] hB;
   delete[] histo ;

   
    
    return 0;
    
}

