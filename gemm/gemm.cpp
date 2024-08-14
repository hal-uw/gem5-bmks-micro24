#include "hip/hip_runtime.h"
/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <gem5/m5ops.h>

#define BLOCK_SIZE 16

/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; 
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/*
*********************************************************************
function name: gpu_square_matrix_mult
description: dot product of two matrix (not only square) in GPU
parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = hipBlockIdx_y * BLOCK_SIZE + hipThreadIdx_y;
    int col = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < hipGridDim_x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + hipThreadIdx_x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[hipThreadIdx_y][hipThreadIdx_x] = 0;
        }
        else
        {
            tile_a[hipThreadIdx_y][hipThreadIdx_x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + hipThreadIdx_y) * n + col;
        if(idx >= n*n)
        {
            tile_b[hipThreadIdx_y][hipThreadIdx_x] = 0;
        }  
        else
        {
            tile_b[hipThreadIdx_y][hipThreadIdx_x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[hipThreadIdx_y][k] * tile_b[k][hipThreadIdx_x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose
description: matrix transpose
parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult
description: dot product of two matrix (not only square) in CPU, 
             for validating GPU results
parameters: 
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C) 
            to store the result
return: none
*********************************************************************
*/
void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k) {
    printf("Hello\n");
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

/*
*********************************************************************
function name: main
description: test and compare
parameters: 
            none
return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(123);
    printf("please type in m n and k\n");
    m = 64;
    n = 32;
    k = 64;

    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c, *h_cc;
    h_a = (float*) malloc(sizeof(float)*m*n);
    h_b = (float*) malloc(sizeof(float)*n*k);
    h_c = (float*) malloc(sizeof(float)*m*k);
    h_cc = (float*) malloc(sizeof(float)*m*k);

    // random initialize matrix A
    double range = double(10) + 1.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = float(range*rand()/(RAND_MAX + 1.0));
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = float(range*rand()/(RAND_MAX + 1.0));
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // some events to count the execution time
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // start to count execution time of GPU version
    hipEventRecord(start, 0);
    // Allocate memory space on the device 
    float *d_a, *d_b, *d_c;
    hipMalloc((void **) &d_a, sizeof(float)*m*n);
    hipMalloc((void **) &d_b, sizeof(float)*n*k);
    hipMalloc((void **) &d_c, sizeof(float)*m*k);

    // copy matrix A and B from host to device memory
    hipMemcpy(d_a, h_a, sizeof(float)*m*n, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, sizeof(float)*n*k, hipMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // Launch kernel 
    if(m == n && n == k)
    {
        m5_getKernelArg(reinterpret_cast<uintptr_t>(d_a), reinterpret_cast<uintptr_t>(d_b), reinterpret_cast<uintptr_t>(d_c), 48, 3, 1);
        hipLaunchKernelGGL((gpu_square_matrix_mult), dim3(dimGrid), dim3(dimBlock), 0, 0, d_a, d_b, d_c, n);    
    }
    else
    {
        m5_getKernelArg(reinterpret_cast<uintptr_t>(d_a), reinterpret_cast<uintptr_t>(d_b), reinterpret_cast<uintptr_t>(d_c), 48, 3, 2);
        hipLaunchKernelGGL((gpu_matrix_mult), dim3(dimGrid), dim3(dimBlock), 0, 0, d_a, d_b, d_c, m, n, k);    
    }
    // Transefr results from device to host 
    hipMemcpy(h_c, d_c, sizeof(float)*m*k, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    // time counting terminate
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // compute time elapse on GPU computing
    hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

    // start the CPU version
    hipEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

    // free memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipHostFree(h_a);
    hipHostFree(h_b);
    hipHostFree(h_c);
    hipHostFree(h_cc);
    return 0;
}