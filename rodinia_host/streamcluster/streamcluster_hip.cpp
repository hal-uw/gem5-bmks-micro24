/***********************************************
        streamcluster_hip.cpp
        : parallelized code of streamcluster
        
        - original code from PARSEC Benchmark Suite
        - parallelization with HIP API has been applied by
        
        Shawn Sang-Ha Lee - sl4ge@virginia.edu
        University of Virginia
        Department of Electrical and Computer Engineering
        Department of Computer Science
        
***********************************************/
#include "hip/hip_runtime.h"
#include "streamcluster_header.h"
#include <gem5/m5ops.h>
using namespace std;

// AUTO-ERROR CHECK FOR ALL HIP FUNCTIONS
#define HIP_SAFE_CALL( call) do {                                      \
    hipError_t err = call;                                               \
    if( hipSuccess != err) {                                           \
      fprintf(stderr, "Hip error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, hipGetErrorString( err) );           \
      exit(EXIT_FAILURE);                                               \
    } } while (0)

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
#define HIPTIME

// host memory
float *work_mem_h;
float *coord_h;

// device memory
float *work_mem_d;
float *coord_d;
int   *center_table_d;
bool  *switch_membership_d;
Point *p;

static int iter = 0;            // counter for total# of iteration

//=======================================
// Euclidean Distance
//=======================================
__device__ float
d_dist(int p1, int p2, int num, int dim, float *coord_d)
{
  float retval = 0.0;
  for(int i = 0; i < dim; i++){
    float tmp = coord_d[(i*num)+p1] - coord_d[(i*num)+p2];
    retval += tmp * tmp;
  }
  return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
__global__ void
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
                    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d)
{
  // block ID and global thread ID
  const int bid  = hipBlockIdx_x + hipGridDim_x * hipBlockIdx_y;
  const int tid = hipBlockDim_x * bid + hipThreadIdx_x;

  if(tid < num)
  {
    float *lower = &work_mem_d[tid*stride];
                
    // cost between this point and point[x]: euclidean distance multiplied by weight
    float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;
                
    // if computed cost is less then original (it saves), mark it as to reassign
    if ( x_cost < p[tid].cost )
    {
      switch_membership_d[tid] = 1;
      lower[K] += x_cost - p[tid].cost;
    }
    // if computed cost is larger, save the difference
    else
    {
      lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
    }
  }
}

//=======================================
// Allocate Device Memory
//=======================================
void allocDevMem(int num, int dim)
{
  center_table_d = (int *) malloc (num * sizeof(int));
  switch_membership_d = (bool *) malloc (num * sizeof(bool));
  p = (Point *) malloc (num * sizeof(Point));
  coord_d = (float *) malloc (num * dim * sizeof(float));
}

//=======================================
// Allocate Host Memory
//=======================================
void allocHostMem(int num, int dim)
{
  coord_h       = (float*) malloc( num * dim * sizeof(float) );
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem()
{
  free(center_table_d);
  free(switch_membership_d);
  free(p);
  free(coord_d);
}

//=======================================
// Free Host Memory
//=======================================
void freeHostMem()
{
  free(coord_h);
}

//=======================================
// pgain Entry - HIP SETUP + HIP CALL
//=======================================
float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged,
             double *serial_t, double *cpu_to_gpu_t, double *gpu_to_cpu_t, double *alloc_t, double *kernel_t, double *free_t)
{       
#ifdef HIPTIME
  float tmp_t;
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
        
  hipEventRecord(start, 0);
#endif

  hipError_t error;
        
  int stride    = *numcenters + 1;                      // size of each work_mem segment
  int K         = *numcenters ;                         // number of centers
  int num               =  points->num;                         // number of points
  int dim               =  points->dim;                         // number of dimension
  int nThread =  num;                                           // number of threads == number of data points
        
  //=========================================
  // ALLOCATE HOST MEMORY + DATA PREPARATION
  //=========================================
  work_mem_h = (float*) malloc(stride * (nThread + 1) * sizeof(float) );
  // Only on the first iteration
  if(iter == 0)
  {
    allocHostMem(num, dim);
  }
        
  // build center-index table
  int count = 0;
  for( int i=0; i<num; i++)
  {
    if( is_center[i] )
    {
      center_table[i] = count++;
    }
  }

  // Extract 'coord'
  // Only if first iteration OR coord has changed
  if(isCoordChanged || iter == 0)
  {
    for(int i=0; i<dim; i++)
    {
      for(int j=0; j<num; j++)
      {
        coord_h[ (num*i)+j ] = points->p[j].coord[i];
      }
    }
  }
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *serial_t += (double) tmp_t;
        
  hipEventRecord(start,0);
#endif

  //=======================================
  // ALLOCATE GPU MEMORY
  //=======================================
  work_mem_d = (float *) malloc ( stride * (nThread + 1) * sizeof(float));
  // Only on the first iteration
  if( iter == 0 )
  {
    allocDevMem(num, dim);
  }
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *alloc_t += (double) tmp_t;
        
  hipEventRecord(start,0);
#endif

  //=======================================
  // CPU-TO-GPU MEMORY COPY
  //=======================================
  // Only if first iteration OR coord has changed
  if(isCoordChanged || iter == 0)
  {
  memcpy(coord_d,  coord_h,        num * dim * sizeof(float)) ;
  }
  memcpy(center_table_d,  center_table,  num * sizeof(int)) ;
  memcpy(p,  points->p,                                num * sizeof(Point)) ;
        
  memset((void*) switch_membership_d, 0,                    num * sizeof(bool)) ;
  memset((void*) work_mem_d,                0, stride * (nThread + 1) * sizeof(float));
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *cpu_to_gpu_t += (double) tmp_t;
        
  hipEventRecord(start,0);
#endif
        
  //=======================================
  // KERNEL: CALCULATE COST
  //=======================================
  // Determine the number of thread blocks in the x- and y-dimension
  int num_blocks         = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
  int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  / (float) MAXBLOCKS);
  int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) / (float) num_blocks_y);      
  dim3 grid_size(num_blocks_x, num_blocks_y, 1);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(p), reinterpret_cast<uintptr_t>(coord_d), reinterpret_cast<uintptr_t>(work_mem_d), 48, 3, 1);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(center_table_d), reinterpret_cast<uintptr_t>(switch_membership_d), 0, 12, 2, 1);
  hipLaunchKernelGGL(kernel_compute_cost, dim3(grid_size), dim3(THREADS_PER_BLOCK), 0, 0,         
                                                        num,                                    // in:  # of data
                                                        dim,                                    // in:  dimension of point coordinates
                                                        x,                                              // in:  point to open a center at
                                                        p,                                              // in:  data point array
                                                        K,                                              // in:  number of centers
                                                        stride,                                 // in:  size of each work_mem segment
                                                        coord_d,                                // in:  array of point coordinates
                                                        work_mem_d,                             // out: cost and lower field array
                                                        center_table_d,                 // in:  center index table
                                                        switch_membership_d             // out:  changes in membership
                                                                );
  hipDeviceSynchronize();
        
  // error check
  error = hipGetLastError();
  if (error != hipSuccess)
  {
    printf("kernel error: %s\n", hipGetErrorString(error));
    exit(EXIT_FAILURE);
  }
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *kernel_t += (double) tmp_t;
        
  hipEventRecord(start,0);
#endif
        
  //=======================================
  // GPU-TO-CPU MEMORY COPY
  //=======================================
  memcpy(work_mem_h,                  work_mem_d,   stride * (nThread + 1) * sizeof(float));
  memcpy(switch_membership, switch_membership_d,     num * sizeof(bool));
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *gpu_to_cpu_t += (double) tmp_t;
        
  hipEventRecord(start,0);
#endif
        
  //=======================================
  // CPU (SERIAL) WORK
  //=======================================
  int number_of_centers_to_close = 0;
  float gl_cost_of_opening_x = z;
  float *gl_lower = &work_mem_h[stride * nThread];
  // compute the number of centers to close if we are to open i
  for(int i=0; i < num; i++)
  {
    if( is_center[i] )
    {
      float low = z;
      for( int j = 0; j < num; j++ )
      {
        low += work_mem_h[ j*stride + center_table[i] ];
      }
                        
      gl_lower[center_table[i]] = low;
                                
      if ( low > 0 )
      {
        ++number_of_centers_to_close;
        work_mem_h[i*stride+K] -= low;
      }
    }
    gl_cost_of_opening_x += work_mem_h[i*stride+K];
  }

  //if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
  if ( gl_cost_of_opening_x < 0 )
  {
    for(int i = 0; i < num; i++)
    {
      bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
      if ( switch_membership[i] || close_center )
      {
        points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
        points->p[i].assign = x;
      }
    }
                
    for(int i = 0; i < num; i++)
    {
      if( is_center[i] && gl_lower[center_table[i]] > 0 )
      {
        is_center[i] = false;
      }
    }
                
    if( x >= 0 && x < num)
    {
      is_center[x] = true;
    }
    *numcenters = *numcenters + 1 - number_of_centers_to_close;
  }
  else
  {
    gl_cost_of_opening_x = 0;
  }
        
  //=======================================
  // DEALLOCATE HOST MEMORY
  //=======================================
  free(work_mem_h);
        
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *serial_t += (double) tmp_t;
        
  hipEventRecord(start,0);
#endif

  //=======================================
  // DEALLOCATE GPU MEMORY
  //=======================================
  free(work_mem_d);
        
        
#ifdef HIPTIME
  hipEventRecord(stop,0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&tmp_t, start, stop);
  *free_t += (double) tmp_t;
#endif
  iter++;
  return -gl_cost_of_opening_x;
}
