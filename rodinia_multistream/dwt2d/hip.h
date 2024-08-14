#ifndef HIP_H_
#define HIP_H_

#include "hip/hip_runtime.h"
#include <stdio.h>

void checkHIPError(const char *msg) {
  hipError_t err = hipGetLastError();
  if( hipSuccess != err) {
    // fprintf(stderr, "Hip error: %s: %s.\n", msg, hipGetErrorString( err) );
    printf("HIP error: %s: %s.\n", msg, hipGetErrorString( err) );
    fflush(NULL);
    exit(EXIT_FAILURE);
  }
}

#endif
