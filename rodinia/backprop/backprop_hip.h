#ifndef BACKPROP_HIP_H_
#define BACKPROP_HIP_H_

int main (int argc, char ** argv);
void bpnn_train_hip(BPNN *net, float *eo, float *eh, size_t numStreams, bool individualGPUs);

#endif
