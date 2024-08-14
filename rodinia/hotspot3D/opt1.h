#ifndef OPT1_H_
#define OPT1_H_

#include "hip/hip_runtime.h"
#include <gem5/m5ops.h>

size_t numStreams;
bool individualGpus;

hipStream_t *hip_stream;

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float* tIn, float *tOut, float sdc,
                            int nx, int ny, int nz,
                            float ce, float cw, 
                            float cn, float cs,
                            float ct, float cb, 
                            float cc) 
{
    float amb_temp = 80.0;

    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;  
    int j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut,
                  int nx, int ny, int nz,
                  float Cap, 
                  float Rx, float Ry, float Rz, 
                  float dt, int numiter) 
{
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(float) * nx * ny * nz;  
    float  **tIn_d, **tOut_d, **p_d;
    hipHostMalloc((void**)&p_d,sizeof(float*)*numStreams);
    hipHostMalloc((void**)&tIn_d,sizeof(float*)*numStreams);
    hipHostMalloc((void**)&tOut_d,sizeof(float*)*numStreams);
    for (int i = 0; i < numStreams; i++) {
        hipHostMalloc((void**)&p_d[i],s);
        hipHostMalloc((void**)&tIn_d[i],s);
        hipHostMalloc((void**)&tOut_d[i],s);
        hipMemcpy(tIn_d[i], tIn, s, hipMemcpyHostToDevice);
        hipMemcpy(p_d[i], p, s, hipMemcpyHostToDevice);
    }
    
    
    hipFuncSetCacheConfig(reinterpret_cast<const void*>(hotspotOpt1), hipFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);
    int count = 257;
    long long start = get_time();
    for (int i = 0; i < numiter; ++i) {
        for (int j = 0; j < numStreams; j++) {
            m5_getKernelArg(reinterpret_cast<uintptr_t>(p_d[j]), reinterpret_cast<uintptr_t>(tIn_d[j]), reinterpret_cast<uintptr_t>(tOut_d[j]), 48, 3, count++);
            hipLaunchKernelGGL(hotspotOpt1, dim3(grid_dim), dim3(block_dim), 0, hip_stream[j], /*0,*/ p_d[j], tIn_d[j], tOut_d[j], stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
            float *t = tIn_d[j];
            tIn_d[j] = tOut_d[j];
            tOut_d[j] = t;
        }
        for (int j = 0; j < numStreams; j++) {
            hipHccModuleRingDoorbell(hip_stream[j]);
            hipStreamSynchronize(hip_stream[j]);
            m5_dump_reset_stats(0, 0);
        }
    }
    
    //hipDeviceSynchronize();

    long long stop = get_time();
    float time = (float)((stop - start)/(1000.0 * 1000.0));
    printf("Time: %.3f (s)\n",time);
    hipMemcpy(tOut, tOut_d[0], s, hipMemcpyDeviceToHost);
    for (int j = 0; j < numStreams; j++) {
        hipFree(p_d[j]);
        hipFree(tIn_d[j]);
        hipFree(tOut_d[j]);
    }
    return;
}

#endif
