//====================================================================================================100
//              UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//              2009.12 Lukasz G. Szafaryn
//              -- reading from image, command line inputs
//              2010.01 Lukasz G. Szafaryn
//              --comments

//====================================================================================================100
//      DEFINE / INCLUDE
//====================================================================================================100

#include "hip/hip_runtime.h"
#include <gem5/m5ops.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "define.h"
#include "extract_kernel.h"
#include "prepare_kernel.h"
#include "reduce_kernel.h"
#include "srad_kernel.h"
#include "srad2_kernel.h"
#include "compress_kernel.h"
#include "graphics.h"
#include "resize.h"
#include "timer.h"

#include "device.h"                             // (in library path specified to compiler)      needed by for device functions
using namespace std;
//====================================================================================================100
//      MAIN FUNCTION
//====================================================================================================100
int main(int argc, char *argv []){
  //================================================================================80
  //    VARIABLES
  //================================================================================80

  // time
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;
  long long time8;
  long long time9;
  long long time10;
  long long time11;
  long long time12;

  time0 = get_time();

  // inputs image, input paramenters
  fp* image_ori;                                                                                                                                // originalinput image
  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;

  // inputs image, input paramenters
  fp* image;                                                                                                                    // input image
  int Nr,Nc;                                                                                                    // IMAGE nbr of rows/cols/elements
  long Ne;

  // algorithm parameters
  int niter;                                                                                                                            // nbr of iterations
  fp lambda;                                                                                                                    // update step size

  // size of IMAGE
  int r1,r2,c1,c2;                                                                                              // row/col coordinates of uniform ROI
  long NeROI;                                                                                                           // ROI nbr of elements

  // surrounding pixel indicies
  int *iN,*iS,*jE,*jW;    

  // counters
  int iter;   // primary loop
  long i,j;    // image row/col

  // memory sizes
  int mem_size_i;
  int mem_size_j;
  int mem_size_single;

  //================================================================================80
  //    GPU VARIABLES
  //================================================================================80

  // CUDA kernel execution parameters
  dim3 threads;
  int blocks_x;
  dim3 blocks;
  dim3 blocks2;
  dim3 blocks3;

  // memory sizes
  int mem_size;                                                                                                                 // matrix memory size

  // HOST
  int no;
  int mul;
  fp total;
  fp total2;
  fp meanROI;
  fp meanROI2;
  fp varROI;
  fp q0sqr;

    //================================================================================80
  //    GET INPUT PARAMETERS
  //================================================================================80

  if(argc != 5){
    printf("ERROR: wrong number of arguments\n");
    return 0;
  }
  else{
    niter = atoi(argv[1]);
    lambda = atof(argv[2]);
    Nr = atoi(argv[3]);                                         // it is 502 in the original image
    Nc = atoi(argv[4]);                                         // it is 458 in the original image
    
  }
  size_t numStreams = 1;
  hipStream_t hip_stream[numStreams];

  // DEVICE
  fp** d_sums;                                                                                                                   // partial sum
  fp** d_sums2;
  int** d_iN;
  int** d_iS;
  int** d_jE;
  int** d_jW;
  fp** d_dN; 
  fp** d_dS; 
  fp** d_dW; 
  fp** d_dE;
  fp** d_I;                                                                                                                              // input IMAGE on DEVICE
  fp** d_c;

  time1 = get_time();


  time2 = get_time();

  //================================================================================80
  //    READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
  //================================================================================80

  // read image
  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;

  image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

  read_graphics(        "multigpu_benchmarks/rodinia/srad/srad_v1/image.pgm",
                        image_ori,
                        image_ori_rows,
                        image_ori_cols,
                        1);

  time3 = get_time();

  //================================================================================80
  //    RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
  //================================================================================80

  Ne = Nr*Nc;

  image = (fp*)malloc(sizeof(fp) * Ne);

  resize(       image_ori,
                image_ori_rows,
                image_ori_cols,
                image,
                Nr,
                Nc,
                1);

  time4 = get_time();

  //================================================================================80
  //    SETUP
  //================================================================================80

  r1     = 0;                                                                                   // top row index of ROI
  r2     = Nr - 1;                                                                      // bottom row index of ROI
  c1     = 0;                                                                                   // left column index of ROI
  c2     = Nc - 1;                                                                      // right column index of ROI

  // ROI image size
  NeROI = (r2-r1+1)*(c2-c1+1);                                                                                  // number of elements in ROI, ROI size

  // allocate variables for surrounding pixels
  mem_size_i = sizeof(int) * Nr;                                                                                        //
  iN = (int *)malloc(mem_size_i) ;                                                                              // north surrounding element
  iS = (int *)malloc(mem_size_i) ;                                                                              // south surrounding element
  mem_size_j = sizeof(int) * Nc;                                                                                        //
  jW = (int *)malloc(mem_size_j) ;                                                                              // west surrounding element
  jE = (int *)malloc(mem_size_j) ;                                                                              // east surrounding element

  // N/S/W/E indices of surrounding pixels (every element of IMAGE)
  for (i=0; i<Nr; i++) {
    iN[i] = i-1;                                                                                                                // holds index of IMAGE row above
    iS[i] = i+1;                                                                                                                // holds index of IMAGE row below
  }
  for (j=0; j<Nc; j++) {
    jW[j] = j-1;                                                                                                                // holds index of IMAGE column on the left
    jE[j] = j+1;                                                                                                                // holds index of IMAGE column on the right
  }

  // N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
  iN[0]    = 0;                                                                                                                 // changes IMAGE top row index from -1 to 0
  iS[Nr-1] = Nr-1;                                                                                                              // changes IMAGE bottom row index from Nr to Nr-1 
  jW[0]    = 0;                                                                                                                 // changes IMAGE leftmost column index from -1 to 0
  jE[Nc-1] = Nc-1;                                                                                                              // changes IMAGE rightmost column index from Nc to Nc-1

  //================================================================================80
  //    GPU SETUP
  //================================================================================80

  // allocate memory for entire IMAGE on DEVICE
  mem_size = sizeof(fp) * Ne;                                                                                                                                           // get the size of float representation of input IMAGE
  
  printf("starting_device\n");
  hipHostMalloc((void **)&d_I, sizeof(fp*)*numStreams);   
  hipHostMalloc((void **)&d_iN, sizeof(int*)*numStreams);   
  hipHostMalloc((void **)&d_iS, sizeof(int*)*numStreams);                                                                                                       // 
  hipHostMalloc((void **)&d_jE, sizeof(int*)*numStreams);                                                                                                       //
  hipHostMalloc((void **)&d_jW, sizeof(int*)*numStreams);                                                                                                       // 

  // allocate memory for partial sums on DEVICE
  hipHostMalloc((void **)&d_sums, sizeof(fp*)*numStreams);                                                                                                       //
  hipHostMalloc((void **)&d_sums2, sizeof(fp*)*numStreams);                                                                                              //

  // allocate memory for derivatives
  hipHostMalloc((void **)&d_dN, sizeof(fp*)*numStreams);                                                                                                         // 
  hipHostMalloc((void **)&d_dS, sizeof(fp*)*numStreams);                                                                                                         // 
  hipHostMalloc((void **)&d_dW, sizeof(fp*)*numStreams);                                                                                                 // 
  hipHostMalloc((void **)&d_dE, sizeof(fp*)*numStreams);                                                                                                         // 

  // allocate memory for coefficient on DEVICE
  hipHostMalloc((void **)&d_c, sizeof(fp*)*numStreams);                                                                                                          // 


  for (int i = 0; i < numStreams; i++) { 
      
    hipHostMalloc((void **)&d_I[i], mem_size);   
    hipHostMalloc((void **)&d_iN[i], mem_size_i);   
  hipHostMalloc((void **)&d_iS[i], mem_size_i);                                                                                                       // 
  hipHostMalloc((void **)&d_jE[i], mem_size_j);                                                                                                       //
  hipHostMalloc((void **)&d_jW[i], mem_size_j);                                                                                                       // 

  // allocate memory for partial sums on DEVICE
  hipHostMalloc((void **)&d_sums[i], mem_size);                                                                                                       //
  hipHostMalloc((void **)&d_sums2[i], mem_size);                                                                                              //

  // allocate memory for derivatives
  hipHostMalloc((void **)&d_dN[i], mem_size);                                                                                                         // 
  hipHostMalloc((void **)&d_dS[i], mem_size);                                                                                                         // 
  hipHostMalloc((void **)&d_dW[i], mem_size);                                                                                                 // 
  hipHostMalloc((void **)&d_dE[i], mem_size);                                                                                                         // 

  // allocate memory for coefficient on DEVICE
  hipHostMalloc((void **)&d_c[i], mem_size);                                                                                                          // 


    //d_I[i] = (fp*)malloc(mem_size);
    //d_iN[i] = (int*)malloc(mem_size_i);
    //d_iS[i] = (int*)malloc(mem_size_i);
    //d_jE[i] = (int*)malloc(mem_size_j);
    //d_jW[i] = (int*)malloc(mem_size_j);
    //d_sums[i] = (fp*)malloc(mem_size);
    //d_sums2[i] = (fp*)malloc(mem_size);
    //d_dN[i] = (fp*)malloc(mem_size);
    //d_dS[i] = (fp*)malloc(mem_size);
    //d_dW[i] = (fp*)malloc(mem_size);
    //d_dE[i] = (fp*)malloc(mem_size);
    //d_c[i] = (fp*)malloc(mem_size);
    //hipMalloc((void **)&d_I[i], mem_size);                                                                                                          //
  }
  printf("init done\n");
  for (int i = 0; i < numStreams; i++) {
     hipMemcpy(d_iN[i], iN, mem_size_i, hipMemcpyHostToDevice);    
  printf("1\n");
     hipMemcpy(d_iS[i], iS, mem_size_i, hipMemcpyHostToDevice);    
  printf("2\n");
     hipMemcpy(d_jE[i], jE, mem_size_j, hipMemcpyHostToDevice);    
  printf("3\n");
     hipMemcpy(d_jW[i], jW, mem_size_j, hipMemcpyHostToDevice);    
  printf("4\n");
  }
  // allocate memory for coordinates on DEVICE
  //hipMemcpy(d_iN, iN, mem_size_i, hipMemcpyHostToDevice);                             //
  //hipMemcpy(d_iS, iS, mem_size_i, hipMemcpyHostToDevice);                             //
  //hipMemcpy(d_jE, jE, mem_size_j, hipMemcpyHostToDevice);                             //
  //hipMemcpy(d_jW, jW, mem_size_j, hipMemcpyHostToDevice);                     //

  // allocate memory for partial sums on DEVICE
  // allocate memory for derivatives

  // allocate memory for coefficient on DEVICE

  checkCUDAError("setup");

  //================================================================================80
  //    KERNEL EXECUTION PARAMETERS
  //================================================================================80

  // all kernels operating on entire matrix
  threads.x = NUMBER_THREADS;                                                                                           // define the number of threads in the block
  threads.y = 1;
  blocks_x = Ne/threads.x;
  if (Ne % threads.x != 0){                                                                                             // compensate for division remainder above by adding one grid
    blocks_x = blocks_x + 1;                                                                                                                                    
  }
  blocks.x = blocks_x;                                                                                                  // define the number of blocks in the grid
  blocks.y = 1;

  time5 = get_time();

  //================================================================================80
  //    COPY INPUT TO CPU
  //================================================================================80
for (int i = 0; i < numStreams; i++) {
  hipMemcpy(d_I[i], image, mem_size, hipMemcpyHostToDevice);
}

  time6 = get_time();
int count = 1;
  //================================================================================80
  //    SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
  //================================================================================80
for (int i = 0; i < numStreams; i++) {
  m5_getKernelArg(reinterpret_cast<uintptr_t>(d_I[i]), 0, 0, 3, 1, count++);
  hipLaunchKernelGGL(extract, dim3(blocks), dim3(threads), 0, hip_stream[i],  Ne,
                                d_I[i]);
}

for (int i = 0; i < numStreams; ++i) {
  hipHccModuleRingDoorbell(hip_stream[i]);
         hipStreamSynchronize(hip_stream[i]);
         m5_dump_reset_stats(0, 0);
   }
  
  checkCUDAError("extract");

  time7 = get_time();

  //================================================================================80
  //    COMPUTATION
  //================================================================================80

  // printf("iterations: ");

  // execute main loop
  for (iter=0; iter<niter; iter++){                                                                             // do for the number of iterations input parameter

    // printf("%d ", iter);
    // fflush(NULL);

    // execute square kernel
    for (int i = 0; i < numStreams; ++i) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_I[i]), reinterpret_cast<uintptr_t>(d_sums[i]), reinterpret_cast<uintptr_t>(d_sums2[i]), 60, 3, count++);
    hipLaunchKernelGGL(prepare, dim3(blocks), dim3(threads), 0, hip_stream[i],        Ne,
                                        d_I[i],
                                        d_sums[i],
                                        d_sums2[i]);
    }

for (int i = 0; i < numStreams; ++i) {
  hipHccModuleRingDoorbell(hip_stream[i]);
         hipStreamSynchronize(hip_stream[i]);
         m5_dump_reset_stats(0, 0);
   }
 
    checkCUDAError("prepare");

    // performs subsequent reductions of sums
    blocks2.x = blocks.x;                                                                                               // original number of blocks
    blocks2.y = blocks.y;                                                                                               
    no = Ne;                                                                                                            // original number of sum elements
    mul = 1;                                                                                                            // original multiplier

    while(blocks2.x != 0){

      checkCUDAError("before reduce");

      // run kernel
for (int i = 0; i < numStreams; ++i) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_sums[i]), reinterpret_cast<uintptr_t>(d_sums2[i]), 0, 15, 2, count++);
      hipLaunchKernelGGL(reduce, dim3(blocks2), dim3(threads), 0, hip_stream[i],      Ne,
                                        no,
                                        mul,
                                        d_sums[i], 
                                        d_sums2[i]);
}


for (int i = 0; i < numStreams; ++i) {
  hipHccModuleRingDoorbell(hip_stream[i]);
         hipStreamSynchronize(hip_stream[i]);
         m5_dump_reset_stats(0, 0);
}

      checkCUDAError("reduce");

      // update execution parameters
      no = blocks2.x;                                                                                           // get current number of elements
      if(blocks2.x == 1){
        blocks2.x = 0;
      }
      else{
        mul = mul * NUMBER_THREADS;                                                                     // update the increment
        blocks_x = blocks2.x/threads.x;                                                         // number of blocks
        if (blocks2.x % threads.x != 0){                                                        // compensate for division remainder above by adding one grid
          blocks_x = blocks_x + 1;
        }
        blocks2.x = blocks_x;
        blocks2.y = 1;
      }

      checkCUDAError("after reduce");

    }

    checkCUDAError("before copy sum");

    // copy total sums to device
    mem_size_single = sizeof(fp) * 1;
    hipMemcpy(&total, d_sums[0], mem_size_single, hipMemcpyDeviceToHost);
    hipMemcpy(&total2, d_sums2[0], mem_size_single, hipMemcpyDeviceToHost);

    checkCUDAError("copy sum");

    // calculate statistics
    meanROI     = total / fp(NeROI);                                                                            // gets mean (average) value of element in ROI
    meanROI2 = meanROI * meanROI;                                                                               //
    varROI = (total2 / fp(NeROI)) - meanROI2;                                           // gets variance of ROI                                                         
    q0sqr = varROI / meanROI2;                                                                                  // gets standard deviation of ROI

    for (int i = 0; i < numStreams; ++i) {
    // execute srad kernel
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_iN[i]), reinterpret_cast<uintptr_t>(d_iS[i]), reinterpret_cast<uintptr_t>(d_jE[i]), 0, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_jW[i]), reinterpret_cast<uintptr_t>(d_dN[i]), reinterpret_cast<uintptr_t>(d_dS[i]), 60, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_dW[i]), reinterpret_cast<uintptr_t>(d_dE[i]), reinterpret_cast<uintptr_t>(d_c[i]), 63, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_I[i]), 0, 0, 0, 1, count++);


    hipLaunchKernelGGL(srad, dim3(blocks), dim3(threads), 0, hip_stream[i],   lambda,                                                                 // SRAD coefficient 
                                Nr,                                                                             // # of rows in input image
                                Nc,                                                                             // # of columns in input image
                                Ne,                                                                             // # of elements in input image
                                d_iN[i],                                                                   // indices of North surrounding pixels
                                d_iS[i],                                                                   // indices of South surrounding pixels
                                d_jE[i],                                                                   // indices of East surrounding pixels
                                d_jW[i],                                                                   // indices of West surrounding pixels
                                d_dN[i],                                                                   // North derivative
                                d_dS[i],                                                                   // South derivative
                                d_dW[i],                                                                   // West derivative
                                d_dE[i],                                                                   // East derivative
                                q0sqr,                                                                  // standard deviation of ROI 
                                d_c[i],                                                                    // diffusion coefficient
                                d_I[i]);                                                                   // output image
    }

    for (int i = 0; i < numStreams; ++i) {
      hipHccModuleRingDoorbell(hip_stream[i]);
       hipStreamSynchronize(hip_stream[i]); 
       m5_dump_reset_stats(0, 0);
    }
    checkCUDAError("srad");

    // execute srad2 kernel
    for (int i = 0; i < numStreams; ++i) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_iN[i]), reinterpret_cast<uintptr_t>(d_iS[i]), reinterpret_cast<uintptr_t>(d_jE[i]), 0, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_jW[i]), reinterpret_cast<uintptr_t>(d_dN[i]), reinterpret_cast<uintptr_t>(d_dS[i]), 60, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_dW[i]), reinterpret_cast<uintptr_t>(d_dE[i]), reinterpret_cast<uintptr_t>(d_c[i]), 63, 3, count);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(d_I[i]), 0, 0, 0, 1, count++);
    hipLaunchKernelGGL(srad2, dim3(blocks), dim3(threads), 0, hip_stream[i],  lambda,                                                                 // SRAD coefficient 
                                Nr,                                                                             // # of rows in input image
                                Nc,                                                                             // # of columns in input image
                                Ne,                                                                             // # of elements in input image
                                d_iN[i],                                                                   // indices of North surrounding pixels
                                d_iS[i],                                                                   // indices of South surrounding pixels
                                d_jE[i],                                                                   // indices of East surrounding pixels
                                d_jW[i],                                                                   // indices of West surrounding pixels
                                d_dN[i],                                                                   // North derivative
                                d_dS[i],                                                                   // South derivative
                                d_dW[i],                                                                   // West derivative
                                d_dE[i],                                                                   // East derivative
                                d_c[i],                                                                    // diffusion coefficient
                                d_I[i]);                                                                   // output image
    } 
    
   
    for (int i = 0; i < numStreams; ++i) {
      hipHccModuleRingDoorbell(hip_stream[i]);
       hipStreamSynchronize(hip_stream[i]); 
       m5_dump_reset_stats(0, 0);
    }

    checkCUDAError("srad2");
  }

  // printf("\n");

  time8 = get_time();

  //================================================================================80
  //    SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
  //================================================================================80
    for (int i = 0; i < numStreams; ++i) {
      m5_getKernelArg(reinterpret_cast<uintptr_t>(d_I[i]), 0, 0, 3, 1, count++);
  hipLaunchKernelGGL(compress, dim3(blocks), dim3(threads), 0, hip_stream[i],         Ne,
                                        d_I[i]);
    }
    for (int i = 0; i < numStreams; ++i) {
      hipHccModuleRingDoorbell(hip_stream[i]);
        hipStreamSynchronize(hip_stream[i]);
        m5_dump_reset_stats(0, 0);
    }

  checkCUDAError("compress");

  time9 = get_time();

  //================================================================================80
  //    COPY RESULTS BACK TO CPU
  //================================================================================80
    hipMemcpy(image, d_I[0], mem_size, hipMemcpyDeviceToHost);

  checkCUDAError("copy back");

  time10 = get_time();

  //================================================================================80
  //    WRITE IMAGE AFTER PROCESSING
  //================================================================================80
  // write_graphics(       "image_out.pgm",
  //                       image,
  //                       Nr,
  //                       Nc,
  //                       1,
  //                       255);

  time11 = get_time();

  //================================================================================80
  //    DEALLOCATE
  //================================================================================80
  free(image_ori);
  free(image);
  free(iN); 
  free(iS); 
  free(jW); 
  free(jE);

    for (int i = 0; i < numStreams; ++i) {
  hipFree(d_I[i]);
  hipFree(d_c[i]);
  hipFree(d_iN[i]);
  hipFree(d_iS[i]);
  hipFree(d_jE[i]);
  hipFree(d_jW[i]);
  hipFree(d_dN[i]);
  hipFree(d_dS[i]);
  hipFree(d_dE[i]);
  hipFree(d_dW[i]);
  hipFree(d_sums[i]);
  hipFree(d_sums2[i]);
    }

  time12 = get_time();

  //================================================================================80
  //    DISPLAY TIMING
  //================================================================================80
  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f percent : SETUP VARIABLES\n",
         (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : READ COMMAND LINE PARAMETERS\n",
         (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : READ IMAGE FROM FILE\n",
         (float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : RESIZE IMAGE\n",
         (float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
         (float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : COPY DATA TO CPU->GPU\n",
         (float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : EXTRACT IMAGE\n",
         (float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : COMPUTE\n",
         (float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : COMPRESS IMAGE\n",
         (float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : COPY DATA TO GPU->CPU\n",
         (float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : SAVE IMAGE INTO FILE\n",
         (float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
  printf("%15.12f s, %15.12f percent : FREE MEMORY\n",
         (float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", (float) (time12-time0) / 1000000);
}

//====================================================================================================100
//      END OF FILE
//====================================================================================================100
