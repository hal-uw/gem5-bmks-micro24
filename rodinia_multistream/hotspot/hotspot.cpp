#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
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

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)     */
#define MAX_PD  (3.0e6)
/* required precision in degrees        */
#define PRECISION       0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor   */
#define FACTOR_CHIP     0.5

/* chip parameters      */
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all      */
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void 
fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file){
  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 ) {
    printf( "The file was not opened\n" );
  }


  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
    {
      sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
    }
                
  fclose(fp);   
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file){
  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 ) {
    printf( "The file was not opened\n" );
  }

  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
    {
      fgets(str, STR_SIZE, fp);
      if (feof(fp)) {
        char const * str = "not enough lines in file";
        fatal((char *)str);
      }
      //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1)) {
        const char * str = "invalid file format";
        fatal((char *)str);
      }
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);   
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(int iteration,   //number of iteration
                               float *power,    //power input
                               float *temp_src, //temperature input/output
                               float *temp_dst, //temperature input/output
                               int grid_cols,   //Col of grid
                               int grid_rows,   //Row of grid
                               int border_cols, // border offset 
                               int border_rows, // border offset
                               float Cap,       //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step, 
                               float time_elapsed){
  __shared__ float temp_on_hip[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float power_on_hip[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

  float amb_temp = 80.0;
  float step_div_Cap;
  float Rx_1,Ry_1,Rz_1;

  int bx = hipBlockIdx_x;
  int by = hipBlockIdx_y;

  int tx=hipThreadIdx_x;
  int ty=hipThreadIdx_y;

  step_div_Cap=step/Cap;

  Rx_1=1/Rx;
  Ry_1=1/Ry;
  Rz_1=1/Rz;

  // each block finally computes result for a small block
  // after N iterations. 
  // it is the non-overlapping small blocks that cover 
  // all the input data

  // calculate the small block size
  int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
  int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

  // calculate the boundary for the block according to 
  // the boundary of its small block
  int blkY = small_block_rows*by-border_rows;
  int blkX = small_block_cols*bx-border_cols;
  int blkYmax = blkY+BLOCK_SIZE-1;
  int blkXmax = blkX+BLOCK_SIZE-1;

  // calculate the global thread coordination
  int yidx = blkY+ty;
  int xidx = blkX+tx;

  // load data if it is within the valid input range
  int loadYidx=yidx, loadXidx=xidx;
  int index = grid_cols*loadYidx+loadXidx;
       
  if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
    temp_on_hip[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
    power_on_hip[ty][tx] = power[index];// Load the power data from global memory to shared memory
  }
  __syncthreads();

  // effective range within this block that falls within 
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validYmin = (blkY < 0) ? -blkY : 0;
  int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

  int N = ty-1;
  int S = ty+1;
  int W = tx-1;
  int E = tx+1;
        
  N = (N < validYmin) ? validYmin : N;
  S = (S > validYmax) ? validYmax : S;
  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool computed;
  for (int i=0; i<iteration ; i++){ 
    computed = false;
    if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
        IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
        IN_RANGE(tx, validXmin, validXmax) && \
        IN_RANGE(ty, validYmin, validYmax) ) {
      computed = true;
      temp_t[ty][tx] =   temp_on_hip[ty][tx] + step_div_Cap * (power_on_hip[ty][tx] + 
                                                                (temp_on_hip[S][tx] + temp_on_hip[N][tx] - 2.0*temp_on_hip[ty][tx]) * Ry_1 + 
                                                                (temp_on_hip[ty][E] + temp_on_hip[ty][W] - 2.0*temp_on_hip[ty][tx]) * Rx_1 + 
                                                                (amb_temp - temp_on_hip[ty][tx]) * Rz_1);
        
    }
    __syncthreads();
    if(i==iteration-1)
      break;
    if(computed)         //Assign the computation range
      temp_on_hip[ty][tx]= temp_t[ty][tx];
    __syncthreads();
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the 
  // small block perform the calculation and switch on ``computed''
  if (computed){
    temp_dst[index]= temp_t[ty][tx];
  }
}

/*
   compute N time steps
*/

int compute_tran_temp(float **MatrixPower,float **MatrixTemp[2], int col, int row,
                      int total_iterations, int num_iterations, int blockCols,
                      int blockRows, int borderCols, int borderRows, size_t numStreams, hipStream_t* hip_stream) 
{
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(blockCols, blockRows);  

  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  float t;
  float time_elapsed;
  time_elapsed=0.001;

  int src = 1, dst = 0; int count = 257;
  for (int i = 0; i < numStreams; ++i) {
    for (t = 0; t < total_iterations; t+=num_iterations) {
      int temp = src;
      src = dst;
      dst = temp;
      m5_getKernelArg(reinterpret_cast<uintptr_t>(MatrixPower[i]), reinterpret_cast<uintptr_t>(MatrixTemp[src][i]), reinterpret_cast<uintptr_t>(MatrixTemp[dst][i]), 48, 3, count++);
      hipLaunchKernelGGL_lk(calculate_temp, dim3(dimGrid), dim3(dimBlock), 0, hip_stream[i] , 0, MIN(num_iterations, total_iterations-t), MatrixPower[i],MatrixTemp[src][i],MatrixTemp[dst][i],\
                         col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
    }
  }

    for(int sm = 0; sm < numStreams; sm++) {
      hipHccModuleRingDoorbell(hip_stream[sm]);
  }

  for(int sm = 0; sm < numStreams; sm++) {
      hipStreamSynchronize(hip_stream[sm]);
  }
      m5_dump_reset_stats(0, 0);

  return dst;
}

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char** argv)
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut; 
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

    if (argc != 9) {
      usage(argc, argv);
    }
    if((grid_rows = atoi(argv[1]))<=0||
       (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0) {
      usage(argc, argv);
    }

    tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];

    size=grid_rows*grid_cols;

    /* --------------- pyramid parameters --------------- */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp || !MatrixOut) {
        const char * str = "unable to allocate memory";
        fatal((char *)str);
    }

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\n"
           "blockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",
           pyramid_height, grid_cols, grid_rows, borderCols, borderRows,
           blockCols, blockRows, smallBlockCol, smallBlockRow);

    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
    
    size_t numStreams = atoi(argv[7]);
    bool individualGpus = atoi(argv[8]);

    hipStream_t hip_stream[numStreams];

    int numGpus;
    hipGetDeviceCount(&numGpus);

    for (int i = 0; i < numStreams; i++) {
      if (individualGpus) {
        hipSetDevice(i % numGpus);
      }
      hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1);
    }

    float **MatrixTemp[2], **MatrixPower;
    
    hipHostMalloc((void **) &MatrixTemp[0], sizeof(float*)*numStreams);
    for (int i = 0; i < numStreams; ++i) {
    	hipHostMalloc((void **) &MatrixTemp[0][i], sizeof(float)*size);
        hipMemcpy(MatrixTemp[0][i], FilesavingTemp, sizeof(float)*size, hipMemcpyHostToDevice);
    }
    
    
    hipHostMalloc((void **) &MatrixTemp[1], sizeof(float*)*numStreams);
    for (int i = 0; i < numStreams; ++i) {
    	hipHostMalloc((void **) &MatrixTemp[1][i], sizeof(float)*size);
    }
    

    hipHostMalloc((void **) &MatrixPower, sizeof(float*)*numStreams);
    for (int i = 0; i < numStreams; ++i) {
    	hipHostMalloc((void **) &MatrixPower[i], sizeof(float)*size);
        hipMemcpy( MatrixPower[i],  FilesavingPower, sizeof(float)*size, hipMemcpyHostToDevice); 
    }
    
    /*
    hipMalloc((void**)&MatrixPower, sizeof(float)*size);
    hipMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, hipMemcpyHostToDevice);
    */
    
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
                                total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows, numStreams, hip_stream);
    printf("Ending simulation\n");
    
    hipMemcpy(MatrixOut, MatrixTemp[ret][0], sizeof(float)*size, hipMemcpyDeviceToHost);

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    hipFree(MatrixPower);
    hipFree(MatrixTemp[0]);
    hipFree(MatrixTemp[1]);
    free(MatrixOut);
}
