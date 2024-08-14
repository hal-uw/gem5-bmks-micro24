// includes, system
#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gem5/m5ops.h>

// includes, kernels
#include "backprop_kernel.h"
#include "backprop.h"
#include "facetrain.h"
#include "backprop_hip.h"

////////////////////////////////////////////////////////////////////////////////
double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  setup(argc, argv);
}

void bpnn_train_hip(BPNN *net, float *eo, float *eh, size_t numStreams, bool individualGPUs)
{
  int in = 0, hid = 0, out = 0;
  float out_err = 0.0f, hid_err = 0.0f;
  int numGpus = 0;

  // initialize streams
  hipStream_t hip_stream[numStreams];
  hipGetDeviceCount(&numGpus);

  for (int i = 0; i < numStreams; i++) {
    if (individualGPUs) {
      hipSetDevice(i % numGpus);
    }
    hipStreamCreate(&hip_stream[i]);
  }

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  int m = 0;
  float * partial_sum;
  float sum = 0.0f;
  float * input_weights_one_dim;
  float * input_weights_prev_one_dim;
  num_blocks = (in / 16);
  dim3  grid(1, num_blocks);
  dim3  threads(16, 16);
  
  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  float * input_hip = (float *) malloc(sizeof(float) * numStreams * (in + 1));
  float * output_hidden_hip = (float *) malloc(sizeof(float) * numStreams * (hid + 1));
  float * input_hidden_hip = (float *) malloc(sizeof(float) * numStreams * (in + 1) * (hid + 1));
  float * hidden_partial_sum = (float *) malloc(sizeof(float) * numStreams * num_blocks * WIDTH);
#endif

#ifdef CPU
  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
#endif

#ifdef GPU
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

  for (int i = 0; i < numStreams; ++i) {
    // 1D array holding 2D array indexed by [numStreams][in+1]
    memcpy(&input_hip[i * (in+1)], net->input_units, (in + 1) * sizeof(float));

    // 1D array holding 2D array indexed by [numStreams][(in+1)*(hid+1)]
    memcpy(&input_hidden_hip[i * (in+1) * (hid+1)], input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float));
  }

  for (int i = 0; i < numStreams; ++i) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(&input_hip[i * (in+1)]), reinterpret_cast<uintptr_t>(&output_hidden_hip[i *(hid+1)]), reinterpret_cast<uintptr_t>(&input_hidden_hip[i * (in+1)*(hid+1)]), 48, 3, 1);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(&hidden_partial_sum[i * (num_blocks * WIDTH)]), 0, 0, 3, 1, 1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(bpnn_layerforward_HIP), dim3(grid), dim3(threads), 0, hip_stream[i], /*0,*/
			  // these are 2D arrays being represented as 1D arrays, so need to offset into them
			  &input_hip[i * (in+1)],
			  &output_hidden_hip[i *(hid+1)],
			  &input_hidden_hip[i * (in+1)*(hid+1)],
			  &hidden_partial_sum[i * (num_blocks * WIDTH)],
			  in,
			  hid);
  }

  // for(int i = 0; i < numStreams; i++) {
  //     hipHccModuleRingDoorbell(hip_stream[i]);
  // }

  for (int i = 0; i < numStreams; i++) {
      hipStreamSynchronize(hip_stream[i]);
  }

  hipDeviceSynchronize();

  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    printf("bpnn kernel error: %s\n", hipGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  // just copy back from stream 0
  memcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float));

  for (int j = 1; j <= hid; j++) {
    sum = 0.0f;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j-1] ;
    }
    sum += net->input_weights[0][j];
    net-> hidden_units[j] = (float)((1.0f) / (1.0 + exp(-sum)));
  }
#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
#endif  

#ifdef GPU
  float * hidden_delta_hip = (float *) malloc (sizeof(float) * numStreams * (hid+1));
  float * input_prev_weights_hip = (float *) malloc (sizeof(float) * numStreams * (in+1)*(hid+1));

  for (int i = 0; i < numStreams; ++i) {
    memcpy(&hidden_delta_hip[i * (hid+1)], net->hidden_delta, (hid + 1) * sizeof(float));
    memcpy(&input_prev_weights_hip[i * (in+1)*(hid+1)], input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float));
    memcpy(&input_hidden_hip[i * (in+1)*(hid+1)], input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float));
  }

  for (int i = 0; i < numStreams; ++i) {
    m5_getKernelArg(reinterpret_cast<uintptr_t>(&hidden_delta_hip[i * (hid+1)]), reinterpret_cast<uintptr_t>(&input_hip[i * (in+1)]), reinterpret_cast<uintptr_t>(&input_hidden_hip[i * (in+1)*(hid+1)]), 48, 3, 2);
    m5_getKernelArg(reinterpret_cast<uintptr_t>(&input_prev_weights_hip[i * (in+1)*(hid+1)]), 0, 0, 3, 1, 2);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(bpnn_adjust_weights_hip), dim3(grid), dim3(threads), 0, hip_stream[i], /* 0, */
			  // these are 2D arrays being represented as 1D arrays, so need to offset into them
			  &hidden_delta_hip[i * (hid+1)],
			  hid,
			  &input_hip[i * (in+1)],
			  in,
			  &input_hidden_hip[i * (in+1)*(hid+1)],
			  &input_prev_weights_hip[i * (in+1)*(hid+1)]
			  );
  }

  // for(int i = 0; i < numStreams; i++) {
  //     hipHccModuleRingDoorbell(hip_stream[i]);
  // }

  for (int i = 0; i < numStreams; i++) {
      hipStreamSynchronize(hip_stream[i]);
  }

  memcpy(net->input_units, input_hip, (in + 1) * sizeof(float));
  memcpy(input_weights_one_dim, input_hidden_hip, (in + 1) * (hid + 1) * sizeof(float));

  free(input_hip);
  free(output_hidden_hip);
  free(input_hidden_hip);
  free(hidden_partial_sum);
  free(input_prev_weights_hip);
  free(hidden_delta_hip);
  
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
#endif
}
