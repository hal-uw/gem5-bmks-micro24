#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#ifdef OPEN
#include "omp.h"
#endif
#include "facetrain.h"
#include "backprop_hip.h"
#include "imagenet.h"

int layer_size = 0;

void backprop_face(size_t numStreams, bool individualGPUs)
{
  BPNN *net = NULL;
  int i = 0;
  float out_err = 0.0f, hid_err = 0.0f;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d, numStreams: %ld, individualGPUs: %d\n", layer_size, numStreams, individualGPUs);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_hip(net, &out_err, &hid_err, numStreams, individualGPUs);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(int argc, char *argv[])
{
  int seed;

  if (argc!=4){
    fprintf(stderr, "usage: backprop <num of input elements> <numStreams> <individualGPUs>\n");
    exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
    fprintf(stderr, "The number of input points must be divided by 16\n");
    exit(0);
  }
  size_t numStreams = atoi(argv[2]);
  bool individualGPUs = atoi(argv[3]);

  seed = 7;
  bpnn_initialize(seed);
  backprop_face(numStreams, individualGPUs);

  exit(0);
}
