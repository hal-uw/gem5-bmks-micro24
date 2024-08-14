#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <gem5/m5ops.h>

#define task 2

#define BLOCK_SIZE 16

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

template <typename T>
__global__ void
vector_square(T *C_d, const T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

template <typename T>
__global__ void 
vector_add(T* C_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] += C_d[i];
    }
}


__host__ void call_vector_square(unsigned blocks, unsigned threadsPerBlock, float* C_d, float* A_d, size_t N, hipStream_t stream, uint32_t lastKernel) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_d), reinterpret_cast<uintptr_t>(A_d), 0, 3, 2, 1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks), dim3(threadsPerBlock), 0, stream, lastKernel, C_d, A_d, N);
}

__host__ void call_vector_add(unsigned blocks, unsigned threadsPerBlock, float* C_d, size_t N, hipStream_t stream, uint32_t lastKernel) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(C_d), 0, 0, 3, 1, 2);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_add), dim3(blocks), dim3(threadsPerBlock), 0, stream, lastKernel, C_d, N);
}

__host__ void call_gemm(dim3 dimGrid, dim3 dimBlock, float* M_a, float* M_b, float* M_c, size_t m, size_t n, size_t k, hipStream_t stream, uint32_t lastKernel)
{
    m5_getKernelArg(reinterpret_cast<uintptr_t>(M_a), reinterpret_cast<uintptr_t>(M_b), reinterpret_cast<uintptr_t>(M_c), 48, 3, 3);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_matrix_mult), dim3(dimGrid), dim3(dimBlock), 0, stream, lastKernel, M_a, M_b, M_c, m, n, k);
}


int main(int argc, char const *argv[])
{

    /*hipStream_t hip_stream[task];
    hipStream_t hip_stream2[task];

    for (int i = 0; i < task; i++) {
        hipStreamCreateWithPriority(&hip_stream[i], hipStreamDefault, Kalmar::priority_low);
        hipStreamCreateWithPriority(&hip_stream2[i], hipStreamDefault, Kalmar::priority_high);
    }*/

    hipStream_t hip_stream;
    hipStream_t hip_stream2;
    hipStream_t hip_stream3;
    hipStream_t hip_stream4;

    //hipStreamCreateWithPriority(&hip_stream, hipStreamDefault, Kalmar::priority_low);
    hipStreamCreateWithPriority(&hip_stream2, hipStreamDefault, Kalmar::priority_normal);
    hipStreamCreateWithPriority(&hip_stream3, hipStreamDefault, Kalmar::priority_low);
    hipStreamCreateWithPriority(&hip_stream4, hipStreamDefault, Kalmar::priority_high);

    /******************************************************************************************/

    size_t N = 4 * 256;
    size_t Nbytes = N * sizeof(float);
    const unsigned blocks = 4;
    const unsigned threadsPerBlock = 256;

    float *A_h, *C_h;
    A_h = (float*)malloc(Nbytes);
    C_h = (float*)malloc(Nbytes);

    // Fill with Phi + i
    for (size_t i=0; i<N; i++)
    {
        A_h[i] = 1.618f + i;
    }

    /******************************************************************************************/

    int m, n, k;
    /* Fixed seed for illustration */
    srand(123);
    m = 128;
    n = 64;
    k = 128;

    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c, *h_d, *h_e, *h_f, *h_cc;
    h_a = (float*) malloc(sizeof(float)*m*n);
    h_b = (float*) malloc(sizeof(float)*n*k);
    h_c = (float*) malloc(sizeof(float)*m*k);
    h_d = (float*) malloc(sizeof(float)*m*k);
    h_e = (float*) malloc(sizeof(float)*m*k);
    h_f = (float*) malloc(sizeof(float)*m*k);
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

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE; //8
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE; //8
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    /******************************************************************************************/
   
    // Launch kernel 
    call_gemm(dimGrid, dimBlock, h_a, h_b, h_c, m, n, k, hip_stream2, 0);
    call_gemm(dimGrid, dimBlock, h_a, h_b, h_d, m, n, k, hip_stream2, 1);
    call_gemm(dimGrid, dimBlock, h_a, h_b, h_e, m, n, k, hip_stream3, 1);
    call_gemm(dimGrid, dimBlock, h_a, h_b, h_f, m, n, k, hip_stream4, 1);


    /******************************************************************************************/

    /*cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

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
        printf("all results are correct!!!\n");
    }
    else
    {
        printf("incorrect results\n");
    }*/

    //hipDeviceSynchronize();

    //free(A_h);
    //free(C_h);
    //free(h_a);
    //free(h_b);
    //free(h_c);
    //free(h_d);
    //free(h_e);
    //free(h_f);
    //free(h_cc);
    return 0;
}
