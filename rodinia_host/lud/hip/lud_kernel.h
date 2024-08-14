#ifndef LUD_KERNEL_H_
#define LUD_KERNEL_H_

#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <gem5/m5ops.h>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

__global__ void 
lud_diagonal(float *m, int matrix_dim, int offset)
{
  int i,j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i][hipThreadIdx_x]=m[array_offset+hipThreadIdx_x];
    array_offset += matrix_dim;
  }
  __syncthreads();
  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (hipThreadIdx_x>i){
      for(j=0; j < i; j++)
        shadow[hipThreadIdx_x][i] -= shadow[hipThreadIdx_x][j]*shadow[j][i];
      shadow[hipThreadIdx_x][i] /= shadow[i][i];
    }

    __syncthreads();
    if (hipThreadIdx_x>i){

      for(j=0; j < i+1; j++)
        shadow[i+1][hipThreadIdx_x] -= shadow[i+1][j]*shadow[j][hipThreadIdx_x];
    }
    __syncthreads();
  }

  /* 
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+hipThreadIdx_x]=shadow[i][hipThreadIdx_x];
    array_offset += matrix_dim;
  }
}

__global__ void
lud_perimeter(float *m, int matrix_dim, int offset)
{
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  if (hipThreadIdx_x < BLOCK_SIZE) {
    idx = hipThreadIdx_x;
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx]=m[array_offset+(hipBlockIdx_x+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = hipThreadIdx_x-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(hipBlockIdx_x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
  }
  __syncthreads();

/* this version works ok on hardware, but not gpgpusim
 **************************************************************
  if (hipThreadIdx_x < BLOCK_SIZE) { //peri-row
    idx=hipThreadIdx_x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }

    
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(hipBlockIdx_x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=hipThreadIdx_x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }

    __syncthreads();
    
    array_offset = (offset+(hipBlockIdx_x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
***************************************************************/
  if (hipThreadIdx_x < BLOCK_SIZE) { //peri-row
    idx=hipThreadIdx_x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }
  } else { //peri-col
    idx=hipThreadIdx_x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();
    
  if (hipThreadIdx_x < BLOCK_SIZE) { //peri-row
    idx=hipThreadIdx_x;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(hipBlockIdx_x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=hipThreadIdx_x - BLOCK_SIZE;
    array_offset = (offset+(hipBlockIdx_x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }

}

__global__ void
lud_internal(float *m, int matrix_dim, int offset)
{
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  int global_row_id = offset + (hipBlockIdx_y+1)*BLOCK_SIZE;
  int global_col_id = offset + (hipBlockIdx_x+1)*BLOCK_SIZE;

  peri_row[hipThreadIdx_y][hipThreadIdx_x] = m[(offset+hipThreadIdx_y)*matrix_dim+global_col_id+hipThreadIdx_x];
  peri_col[hipThreadIdx_y][hipThreadIdx_x] = m[(global_row_id+hipThreadIdx_y)*matrix_dim+offset+hipThreadIdx_x];

  __syncthreads();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[hipThreadIdx_y][i] * peri_row[i][hipThreadIdx_x];
  m[(global_row_id+hipThreadIdx_y)*matrix_dim+global_col_id+hipThreadIdx_x] -= sum;


}

void lud_hip(float *m, int matrix_dim)
{
  int i=0;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  //float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      m5_getKernelArg(reinterpret_cast<uintptr_t>(m), 0, 0, 3, 1, 1);
      hipLaunchKernelGGL(lud_diagonal, dim3(1), dim3(BLOCK_SIZE), 0, 0, m, matrix_dim, i);
      m5_getKernelArg(reinterpret_cast<uintptr_t>(m), 0, 0, 3, 1, 2); 
      hipLaunchKernelGGL(lud_perimeter, dim3((matrix_dim-i)/BLOCK_SIZE-1), dim3(BLOCK_SIZE*2), 0, 0, m, matrix_dim, i);
      dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
      m5_getKernelArg(reinterpret_cast<uintptr_t>(m), 0, 0, 3, 1, 3);
      hipLaunchKernelGGL(lud_internal, dim3(dimGrid), dim3(dimBlock), 0, 0, m, matrix_dim, i); 
  }
  m5_getKernelArg(reinterpret_cast<uintptr_t>(m), 0, 0, 3, 1, 4);
  hipLaunchKernelGGL(lud_diagonal, dim3(1), dim3(BLOCK_SIZE), 0, 0, m, matrix_dim, i);
}

#endif
