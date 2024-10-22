#include "hip/hip_runtime.h"
#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include <gem5/m5ops.h>

// includes, kernels
#include "needle_kernel.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int blosum62[24][24] = {
                        { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
                        {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
                        {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
                        {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
                        { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
                        {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
                        { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
                        {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
                        {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
                        { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
                        { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
                        {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
                        {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
                        { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d \n", BLOCK_SIZE);

  runTest( argc, argv);

  return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
  exit(1);
}

void runTest( int argc, char** argv) 
{
  int max_rows, max_cols, penalty;
  int *input_itemsets, *output_itemsets, *referrence;
  int **matrix_hip,  **referrence_hip;
  int size;
    
  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  if (argc > 2)
  {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
  }
  else{
    usage(argc, argv);
  }

  if(atoi(argv[1])%16!=0){
    fprintf(stderr,"The dimension values must be a multiple of 16\n");
    exit(1);
  }

  size_t numStreams = atoi(argv[3]);

  hipStream_t hip_stream[numStreams];

  int numGpus;
  hipGetDeviceCount(&numGpus);

  for (int i = 0; i < numStreams; i++) {
    hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1);
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
  input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
  output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );

  if (!input_itemsets) {
    fprintf(stderr, "error: can not allocate memory");
  }

  srand ( 7 );

  for (int i = 0 ; i < max_cols; i++){
    for (int j = 0 ; j < max_rows; j++){
      input_itemsets[i*max_cols+j] = 0;
    }
  }

  printf("Start Needleman-Wunsch\n");

  for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
    input_itemsets[i*max_cols] = rand() % 10 + 1;
  }
  for( int j=1; j< max_cols ; j++){    //please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1 ; i < max_cols; i++){
    for (int j = 1 ; j < max_rows; j++){
      referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
    }
  }

  for( int i = 1; i< max_rows ; i++) {
    input_itemsets[i*max_cols] = -i * penalty;
  }
  for( int j = 1; j< max_cols ; j++) {
    input_itemsets[j] = -j * penalty;
  }

  size = max_cols * max_rows;
  
  hipHostMalloc((void**)& referrence_hip, sizeof(int*)*numStreams);
  for (int i = 0; i < numStreams; ++i) {
    hipHostMalloc((void**)& referrence_hip[i], sizeof(int)*size);
    hipMemcpy(referrence_hip[i], referrence, sizeof(int) * size, hipMemcpyHostToDevice);
  }
  
  hipHostMalloc((void**)& matrix_hip, sizeof(int*)*numStreams);
  for (int i = 0; i < numStreams; ++i) {
    hipHostMalloc((void**)& matrix_hip[i], sizeof(int)*size);
    hipMemcpy(matrix_hip[i], input_itemsets, sizeof(int) * size, hipMemcpyHostToDevice);
  }

  
  dim3 dimGrid;
  dim3 dimBlock(BLOCK_SIZE, 1);
  int block_width = ( max_cols - 1 )/BLOCK_SIZE;

  printf("Processing top-left matrix\n");
  //process top-left matrix
  int count = 1;
  for (int j = 0; j < numStreams; ++j) {
    for( int i = 1 ; i <= block_width ; i++){
      dimGrid.x = i;
      dimGrid.y = 1;
      m5_getKernelArg(reinterpret_cast<uintptr_t>(referrence_hip[j]), reinterpret_cast<uintptr_t>(matrix_hip[j]), 0, 12, 2, count++);
      hipLaunchKernelGGL_lk(needle_hip_shared_1, dim3(dimGrid), dim3(dimBlock), 0, hip_stream[j], 0, referrence_hip[j], matrix_hip[j]
                                                  ,max_cols, penalty, i, block_width); 
    }
  }

    for(int sm = 0; sm < numStreams; sm++) {
      hipHccModuleRingDoorbell(hip_stream[sm]);
  }

  for(int sm = 0; sm < numStreams; sm++) {
      hipStreamSynchronize(hip_stream[sm]);
  }
      m5_dump_reset_stats(0, 0);
  
  printf("Processing bottom-right matrix\n");
  //process bottom-right matrix
  for (int j = 0; j < numStreams; ++j) {
    for( int i = block_width - 1  ; i >= 1 ; i--){
      dimGrid.x = i;
      dimGrid.y = 1;
      m5_getKernelArg(reinterpret_cast<uintptr_t>(referrence_hip[j]), reinterpret_cast<uintptr_t>(matrix_hip[j]), 0, 12, 2, count++);
      hipLaunchKernelGGL_lk(needle_hip_shared_2, dim3(dimGrid), dim3(dimBlock), 0, hip_stream[j], 0, referrence_hip[j], matrix_hip[j]
                                                  ,max_cols, penalty, i, block_width); 
    }
  }
    for(int sm = 0; sm < numStreams; sm++) {
      hipHccModuleRingDoorbell(hip_stream[sm]);
  }

  for(int sm = 0; sm < numStreams; sm++) {
      hipStreamSynchronize(hip_stream[sm]);
  }
      m5_dump_reset_stats(0, 0);
  
  hipMemcpy(output_itemsets, matrix_hip[0], sizeof(int) * size, hipMemcpyDeviceToHost);

  //#define TRACEBACK
#ifdef TRACEBACK

  FILE *fpo = fopen("result.txt","w");
  fprintf(fpo, "print traceback value GPU:\n");
    
  for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
    int nw, n, w, traceback;
    if ( i == max_rows - 2 && j == max_rows - 2 )
      fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
    if ( i == 0 && j == 0 )
      break;
    if ( i > 0 && j > 0 ){
      nw = output_itemsets[(i - 1) * max_cols + j - 1];
      w  = output_itemsets[ i * max_cols + j - 1 ];
      n  = output_itemsets[(i - 1) * max_cols + j];
    }
    else if ( i == 0 ){
      nw = n = LIMIT;
      w  = output_itemsets[ i * max_cols + j - 1 ];
    }
    else if ( j == 0 ){
      nw = w = LIMIT;
      n  = output_itemsets[(i - 1) * max_cols + j];
    }
    else{
    }

    //traceback = maximum(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + referrence[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if(traceback == new_nw)
      traceback = nw;
    if(traceback == new_w)
      traceback = w;
    if(traceback == new_n)
      traceback = n;

    fprintf(fpo, "%d ", traceback);

    if(traceback == nw )
      {i--; j--; continue;}
    else if(traceback == w )
      {j--; continue;}
    else if(traceback == n )
      {i--; continue;}
    else
      ;
  }

  fclose(fpo);

#endif
  
  for (int i = 0; i < numStreams; ++i) {
    hipFree(referrence_hip[i]);
    hipFree(matrix_hip[i]);
  }
  free(referrence);
  free(input_itemsets);
  free(output_itemsets);
}
