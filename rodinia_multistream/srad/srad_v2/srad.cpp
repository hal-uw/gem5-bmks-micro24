#include "hip/hip_runtime.h"
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"
#include <gem5/m5ops.h>

// includes, kernels
#include "srad_kernel.h"
size_t numStreams;
bool individualGpus;
hipStream_t *hip_stream; 

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
  fprintf(stderr, "\t<rows>   - number of rows\n");
  fprintf(stderr, "\t<cols>    - number of cols\n");
  fprintf(stderr, "\t<y1>        - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>   - number of iterations\n");

  exit(1);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  runTest( argc, argv);

  return EXIT_SUCCESS;
}

void
runTest( int argc, char** argv) 
{
  int rows, cols, size_I, size_R, niter = 10, iter;
  float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

#ifdef CPU
  float Jc, G2, L, num, den, qsqr;
  int *iN,*iS,*jE,*jW, k;
  float *dN,*dS,*dW,*dE;
  float cN,cS,cW,cE,D;
#endif

#ifdef GPU

  float **J_hip;
  float **C_hip;
  float **E_C, **W_C, **N_C, **S_C;

#endif

  unsigned int r1, r2, c1, c2;
  float *c;
    
  if (argc == 11)
  {
    rows = atoi(argv[1]);  //number of rows in the domain
    cols = atoi(argv[2]);  //number of cols in the domain
    if ((rows%16!=0) || (cols%16!=0)){
      fprintf(stderr, "rows and cols must be multiples of 16\n");
      exit(1);
    }
    r1   = atoi(argv[3]);  //y1 position of the speckle
    r2   = atoi(argv[4]);  //y2 position of the speckle
    c1   = atoi(argv[5]);  //x1 position of the speckle
    c2   = atoi(argv[6]);  //x2 position of the speckle
    lambda = atof(argv[7]); //Lambda value
    niter = atoi(argv[8]); //number of iterations
    numStreams = atoi(argv[9]);
    individualGpus = atoi(argv[10]);
  }
  else{
    usage(argc, argv);
  }
   
   hip_stream = new hipStream_t [numStreams];
   
   int numGpus;
   hipGetDeviceCount(&numGpus);

   for (int i = 0; i < numStreams; i++) {
    if (individualGpus) {
      hipSetDevice(i % numGpus);
    }
    hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1);
  }

  size_I = cols * rows;
  size_R = (r2-r1+1)*(c2-c1+1);   

  I = (float *)malloc( size_I * sizeof(float) );
  J = (float *)malloc( size_I * sizeof(float) );
  c  = (float *)malloc(sizeof(float)* size_I) ;

#ifdef CPU

  iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
  iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
  jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
  jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


  dN = (float *)malloc(sizeof(float)* size_I) ;
  dS = (float *)malloc(sizeof(float)* size_I) ;
  dW = (float *)malloc(sizeof(float)* size_I) ;
  dE = (float *)malloc(sizeof(float)* size_I) ;    
    

  for (int i=0; i< rows; i++) {
    iN[i] = i-1;
    iS[i] = i+1;
  }    
  for (int j=0; j< cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
  }
  iN[0]    = 0;
  iS[rows-1] = rows-1;
  jW[0]    = 0;
  jE[cols-1] = cols-1;

#endif

#ifdef GPU
  //Allocate device memory
  hipHostMalloc((void**)& J_hip, sizeof(float*)* numStreams);
  hipHostMalloc((void**)& C_hip, sizeof(float*)* numStreams);
  hipHostMalloc((void**)& E_C, sizeof(float*)* numStreams);
  hipHostMalloc((void**)& W_C, sizeof(float*)* numStreams);
  hipHostMalloc((void**)& S_C, sizeof(float*)* numStreams);
  hipHostMalloc((void**)& N_C, sizeof(float*)* numStreams);

  for (int i = 0; i < numStreams; i++) { 

    hipHostMalloc((void**)& J_hip[i], sizeof(float)* size_I);
    hipHostMalloc((void**)& C_hip[i], sizeof(float)* size_I);
    hipHostMalloc((void**)& E_C[i], sizeof(float)* size_I);
    hipHostMalloc((void**)& W_C[i], sizeof(float)* size_I);
    hipHostMalloc((void**)& S_C[i], sizeof(float)* size_I);
    hipHostMalloc((void**)& N_C[i], sizeof(float)* size_I);
  }

#endif 

  printf("Randomizing the input matrix\n");
  //Generate a random matrix
  random_matrix(I, rows, cols);

  for (int k = 0;  k < size_I; k++ ) {
    J[k] = (float)exp(I[k]) ;
  }
  printf("Start the SRAD main loop\n");
  for (iter=0; iter< niter; iter++){     
    sum=0; sum2=0;
    for (int i=r1; i<=r2; i++) {
      for (int j=c1; j<=c2; j++) {
        tmp   = J[i * cols + j];
        sum  += tmp ;
        sum2 += tmp*tmp;
      }
    }
    meanROI = sum / size_R;
    varROI  = (sum2 / size_R) - meanROI*meanROI;
    q0sqr   = varROI / (meanROI*meanROI);

#ifdef CPU
    for (int i = 0 ; i < rows ; i++) {
      for (int j = 0; j < cols; j++) { 
        k = i * cols + j;
        Jc = J[k];
 
        // directional derivates
        dN[k] = J[iN[i] * cols + j] - Jc;
        dS[k] = J[iS[i] * cols + j] - Jc;
        dW[k] = J[i * cols + jW[j]] - Jc;
        dE[k] = J[i * cols + jE[j]] - Jc;

        G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
              + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

        num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
        den  = 1 + (.25*L);
        qsqr = num/(den*den);
 
        // diffusion coefficent (equ 33)
        den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
        c[k] = 1.0 / (1.0+den) ;
                
        // saturate diffusion coefficent
        if (c[k] < 0) {c[k] = 0;}
        else if (c[k] > 1) {c[k] = 1;}
      }
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {        
        // current index
        k = i * cols + j;
                
        // diffusion coefficent
        cN = c[k];
        cS = c[iS[i] * cols + j];
        cW = c[k];
        cE = c[i * cols + jE[j]];

        // divergence (equ 58)
        D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
        // image update (equ 61)
        J[k] = J[k] + 0.25*lambda*D;
      }
    }
#endif // CPU

#ifdef GPU
    //Currently the input size must be divided by 16 - the block size
    int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_x , block_y);
    
printf("malloc done\n");
    //Copy data from main memory to device memory
  for (int i = 0; i < numStreams; i++) { 
    hipMemcpy(J_hip[i], J, sizeof(float) * size_I, hipMemcpyHostToDevice);
  }

printf("copy done numStreams=%d\n", numStreams);
  int count = 257;
for (int i = 0; i < numStreams; i++) { 
    m5_getKernelArg(reinterpret_cast<uintptr_t>(E_C[i]), reinterpret_cast<uintptr_t>(W_C[i]), reinterpret_cast<uintptr_t>(N_C[i]), 63, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(S_C[i]), reinterpret_cast<uintptr_t>(J_hip[i]), reinterpret_cast<uintptr_t>(C_hip[i]), 51, 3, count++);
    //Run kernels
    hipLaunchKernelGGL_lk(srad_hip_1, dim3(dimGrid), dim3(dimBlock), 0, hip_stream[i], 0, E_C[i], W_C[i], N_C[i], S_C[i], J_hip[i], C_hip[i], cols, rows, q0sqr);
}  
  for(int sm = 0; sm < numStreams; sm++) {
      hipHccModuleRingDoorbell(hip_stream[sm]);
  }

  for(int sm = 0; sm < numStreams; sm++) {
      hipStreamSynchronize(hip_stream[sm]);
  }
      m5_dump_reset_stats(0, 0);

printf("kernel1 done\n");
  
for (int i = 0; i < numStreams; ++i) { 
    m5_getKernelArg(reinterpret_cast<uintptr_t>(E_C[i]), reinterpret_cast<uintptr_t>(W_C[i]), reinterpret_cast<uintptr_t>(N_C[i]), 0, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(S_C[i]), reinterpret_cast<uintptr_t>(J_hip[i]), reinterpret_cast<uintptr_t>(C_hip[i]), 12, 3, count++);
    hipLaunchKernelGGL_lk(srad_hip_2, dim3(dimGrid), dim3(dimBlock), 0, hip_stream[i], 0, E_C[i], W_C[i], N_C[i], S_C[i], J_hip[i], C_hip[i], cols, rows, lambda, q0sqr); 
  }
  
  for(int sm = 0; sm < numStreams; sm++) {
      hipHccModuleRingDoorbell(hip_stream[sm]);
  }

  for(int sm = 0; sm < numStreams; sm++) {
      hipStreamSynchronize(hip_stream[sm]);
  }
      m5_dump_reset_stats(0, 0);
    
  //Copy data from device memory to main memory
    hipMemcpy(J, J_hip[0], sizeof(float) * size_I, hipMemcpyDeviceToHost);

#endif   
  }
  hipDeviceSynchronize();

#ifdef OUTPUT
  //Printing output
  printf("Printing Output:\n"); 
  for( int i = 0 ; i < rows ; i++){
    for ( int j = 0 ; j < cols ; j++){
      printf("%.5f ", J[i * cols + j]); 
    }
    printf("\n"); 
  }
#endif 

  printf("Computation Done\n");

  free(I);
  free(J);
#ifdef CPU
  free(iN); free(iS); free(jW); free(jE);
  free(dN); free(dS); free(dW); free(dE);
#endif
printf("kenrels done\n");
#ifdef GPU
  
  for (int i = 0; i < numStreams; ++i) { 
  hipFree(C_hip[i]);
  hipFree(J_hip[i]);
  hipFree(E_C[i]);
  hipFree(W_C[i]);
  hipFree(N_C[i]);
  hipFree(S_C[i]);
  }
#endif 
  free(c);
}

void random_matrix(float *I, int rows, int cols){
  srand(7);

  for( int i = 0 ; i < rows ; i++){
    for ( int j = 0 ; j < cols ; j++){
      I[i * cols + j] = rand()/(float)RAND_MAX ;
    }
  }
}
