#ifndef SIMPLEMULTIGPU_H
#define SIMPLEMULTIGPU_H

typedef struct
{
    //Host-side input data
    int dataN;
    float *h_Data;

    //Partial sum for this GPU
    float *h_Sum;

    //Device buffers
    float *d_Data,*d_Sum;

    //Reduction copied back from GPU
    float *h_Sum_from_device;

    //Stream for asynchronous command execution
    hipStream_t stream;

} TGPUplan;

extern "C" 
void launch_reduceKernel(float *d_Result, float *d_Input, int N, int BLOCK_N, int THREAD_N, hipStream_t &s);

#endif
