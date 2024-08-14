#include <stdio.h>
#include <hip/hip_runtime.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <string>
#include <gem5/m5ops.h>

#include <gem5/m5ops.h>

float feature_vect[] = {2.240018,    2.2570236,    0.11304555,   -0.21307051,
                        0.8988138,   0.039065503,  0.023874786,  0.13153112,
                        0.15324382,  0.16986738,   -0.020297153, -0.26773554,
                        0.40202165,  0.35923952,   0.060746543,  0.35402644,
                        0.086052455, -0.10499257,  0.04395058,   0.026407119,
                        -0.48301497, 0.120889395,  0.67980754,   -0.19875681,
                        -0.5443737,  -0.039534688, 0.20888293,   0.054865785,
                        -0.4846478,  0.1,          0.1,          0.1};

float *means_vect;
float *precs_vect;
float *weight_vect;
float *factor_vect;
float *score_vect;

__device__ __constant__ float logZero = -3.4028235E38;
__device__ __constant__ float maxLogValue = 7097004.5;
__device__ __constant__ float minLogValue = -7443538.0;
__device__ __constant__ float naturalLogBase = (float)1.00011595E-4;
__device__ __constant__ float inverseNaturalLogBase = 9998.841;
// fixed for a given accoustic model
__device__ __constant__ int comp_size = 32;
__device__ __constant__ int feat_size = 29;
__device__ __constant__ int senone_size = 5120;

extern "C"

    __global__ void
    computeScore(const float *feature_vect, float *means_vect,
                 float *precs_vect, float *weight_vect, float *factor_vect,
                 float *score_vect) {
  int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (i < senone_size) {
    float local_score_vect = logZero;

#pragma unroll 32
    for (int j = 0; j < comp_size; j++) {
      // getScore
      float logDval = 0.0f;
#pragma unroll 29
      for (int k = 0; k < feat_size; k++) {
        int idx = i + senone_size * j + k * comp_size * senone_size;
        float logDiff = feature_vect[k] - means_vect[idx];
        logDval += logDiff * logDiff * precs_vect[idx];
      }

      // Convert to the appropriate base.
      if (logDval != logZero) {
        logDval = logDval * inverseNaturalLogBase;
      }

      int idx2 = i + j * senone_size;

      // Add the precomputed factor, with the appropriate sign.
      logDval -= factor_vect[idx2];

      if (logDval < logZero) {
        logDval = logZero;
      }
      // end of getScore

      float logVal2 = logDval + weight_vect[idx2];

      float logHighestValue = local_score_vect;
      float logDifference = local_score_vect - logVal2;

      // difference is always a positive number
      if (logDifference < 0) {
        logHighestValue = logVal2;
        logDifference = -logDifference;
      }

      float logValue = -logDifference;
      float logInnerSummation;
      if (logValue < minLogValue) {
        logInnerSummation = 0.0;
      } else if (logValue > maxLogValue) {
        logInnerSummation = FLT_MAX;
      } else {
        if (logValue == logZero) {
          logValue = logZero;
        } else {
          logValue = logValue * naturalLogBase;
        }
        logInnerSummation = __expf(logValue);
      }

      logInnerSummation += 1.0;

      float returnLogValue;
      if (logInnerSummation <= 0.0) {
        returnLogValue = logZero;
      } else {
        returnLogValue = __logf(logInnerSummation) * inverseNaturalLogBase;
        if (returnLogValue > FLT_MAX) {
          returnLogValue = FLT_MAX;
        } else if (returnLogValue < -FLT_MAX) {
          returnLogValue = -FLT_MAX;
        }
      }
      // sum log
      local_score_vect = logHighestValue + returnLogValue;
    }
    score_vect[i] = local_score_vect;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    fprintf(stderr, "[ERROR] Invalid arguments provided.\n\n");
    fprintf(stderr, "Usage: %s [INPUT FILE] [STREAMS] [THREADSPERBLOCK] [INDIVIDUAL GPUS]\n\n", argv[0]);
    exit(0);
  }

  size_t numStreams = atoi(argv[2]);
  unsigned threadsPerBlock = atoi(argv[3]);
  bool individualGpus = atoi(argv[4]);

  hipStream_t hip_stream[numStreams];

  int numGpus;
  hipGetDeviceCount(&numGpus);

  for (int i = 0; i < numStreams; i++)
  {
    if (individualGpus) {
      hipSetDevice(i % numGpus);
    }
    hipStreamCreateWithFlags(&hip_stream[i]);
  }

  float *dev_feat_vect;

  int comp_size = 32;
  int senone_size = 5120;

  int means_array_size = senone_size * comp_size * comp_size;
  int comp_array_size = senone_size * comp_size;

  means_vect = (float *)malloc(means_array_size * sizeof(float));
  precs_vect = (float *)malloc(means_array_size * sizeof(float));
  weight_vect = (float *)malloc(comp_array_size * sizeof(float));
  factor_vect = (float *)malloc(comp_array_size * sizeof(float));

  float *means_vect2 = (float *)malloc(means_array_size * sizeof(float));
  float *precs_vect2 = (float *)malloc(means_array_size * sizeof(float));
  float *weight_vect2 = (float *)malloc(comp_array_size * sizeof(float));
  float *factor_vect2 = (float *)malloc(comp_array_size * sizeof(float));

  float *dev_means_vect;
  float *dev_precs_vect;
  float *dev_weight_vect;
  float *dev_factor_vect;

  score_vect = (float *)malloc(senone_size * sizeof(float));

  float *dev_score_vect;

  int blockSizeX = 256;
  int gridSizeX = (int)ceil(senone_size / blockSizeX);

  int div_grid = ((int)(gridSizeX / 32));
  gridSizeX = (div_grid + 1) * 32;

  // load model from file
  FILE *fp = fopen(argv[1], "r");
  if (fp == NULL) {  // checks for the file
    printf("\n Can’t open file");
    exit(-1);
  }

  int idx = 0;
  for (int i = 0; i < senone_size; i++) {
    for (int j = 0; j < comp_size; j++) {
      for (int k = 0; k < comp_size; k++) {
        float elem;
        fscanf(fp, "%f", &elem);
        means_vect[idx] = elem;
        idx = idx + 1;
      }
    }
    printf("i1 = %d\n", i);
  }

  idx = 0;
  for (int i = 0; i < senone_size; i++) {
    for (int j = 0; j < comp_size; j++) {
      for (int k = 0; k < comp_size; k++) {
        float elem;
        fscanf(fp, "%f", &elem);
        precs_vect[idx] = elem;
        idx = idx + 1;
      }
    }
    printf("i2 = %d\n", i);
  }

  idx = 0;
  for (int i = 0; i < senone_size; i++) {
    for (int j = 0; j < comp_size; j++) {
      float elem;
      fscanf(fp, "%f", &elem);
      weight_vect[idx] = elem;
      idx = idx + 1;
    }
  }

  idx = 0;
  for (int i = 0; i < senone_size; i++) {
    for (int j = 0; j < comp_size; j++) {
      float elem;
      fscanf(fp, "%f", &elem);
      factor_vect[idx] = elem;
      idx = idx + 1;
    }
  }

  fclose(fp);

  int idx3 = 0;
  for (int j = 0; j < comp_size; j++) {
    for (int i = 0; i < senone_size; i++) {
      int ij = j + i * comp_size;
      weight_vect2[idx3] = weight_vect[ij];
      factor_vect2[idx3] = factor_vect[ij];
      idx3 += 1;
    }
  }

  int idx4 = 0;
  for (int k = 0; k < comp_size; k++) {
    for (int j = 0; j < comp_size; j++) {
      for (int i = 0; i < senone_size; i++) {
        int ijk = k + comp_size * j + i * comp_size * comp_size;
        means_vect2[idx4] = means_vect[ijk];
        precs_vect2[idx4] = precs_vect[ijk];
        idx4 += 1;
      }
    }
    printf("i3 = %d\n", k);
  }

  for (int i = 0; i < senone_size; i++) {
    for (int j = 0; j < comp_size; j++) {
      for (int k = 0; k < 29; k++) {
        int ijk = k + comp_size * j + i * comp_size * comp_size;
        int kji = i + senone_size * j + k * comp_size * senone_size;
        if (means_vect2[kji] != means_vect[ijk]) {
          printf("%f != %f\n", means_vect2[kji], means_vect[ijk]);
        }
      }
    }
    printf("i4 = %d\n", i);
  }

  dim3 block(threadsPerBlock);
  dim3 grid;
  grid.x = (senone_size + block.x - 1) / block.x;

  if (grid.x < 32) grid.x = 32;

  m5_dump_reset_stats(0, 0);
  for (int i = 0; i < numStreams; i++) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(feature_vect), reinterpret_cast<uintptr_t>(means_vect2), reinterpret_cast<uintptr_t>(precs_vect2), 0, 3, 1);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(weight_vect2), reinterpret_cast<uintptr_t>(factor_vect2), reinterpret_cast<uintptr_t>(score_vect), 48, 3, 1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(computeScore), dim3(grid), dim3(block),
                       0, hip_stream[i], 0, feature_vect,
                       means_vect2, precs_vect2, weight_vect2,
                       factor_vect2, score_vect);
  }
  // for (int i = 0; i < numStreams; i++) {
  //   hipHccModuleRingDoorbell(hip_stream[i]);
  // }

  for (int i = 0; i < numStreams; i++) {
    hipStreamSynchronize(hip_stream[i]);
  }
  hipDeviceSynchronize();
  m5_dump_reset_stats(0, 0);
#if TESTING
  FILE *f = fopen("../input/gmm_scoring.gpu", "w");

  for (int i = 0; i < senone_size; ++i) fprintf(f, "%.0f\n", score_vect[i]);

  fclose(f);
#endif
  free(means_vect);
  free(precs_vect);

  free(weight_vect);
  free(factor_vect);

  free(score_vect);
}
