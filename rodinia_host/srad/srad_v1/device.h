#ifndef DEVICE_H_
#define DEVICE_H_

#include "hip/hip_runtime.h"

//======================================================================================================================================================150
//      FUNCTIONS
//======================================================================================================================================================150

//====================================================================================================100
//      SET DEVICE
//====================================================================================================100
void setdevice(void){
  // variables
  int num_devices;
  int device;

  // work
  hipGetDeviceCount(&num_devices);
  if (num_devices > 1) {
    // variables
    int max_multiprocessors; 
    int max_device;
    hipDeviceProp_t properties;

    // initialize variables
    max_multiprocessors = 0;
    max_device = 0;
                
    for (device = 0; device < num_devices; device++) {
      hipGetDeviceProperties(&properties, device);
      if (max_multiprocessors < properties.multiProcessorCount) {
        max_multiprocessors = properties.multiProcessorCount;
        max_device = device;
      }
    }
    hipSetDevice(max_device);
  }
}

//====================================================================================================100
//      GET LAST ERROR
//====================================================================================================100
void checkCUDAError(const char *msg)
{
  hipError_t err = hipGetLastError();
  if( hipSuccess != err) {
    // fprintf(stderr, "Cuda error: %s: %s.\n", msg, hipGetErrorString( err) );
    printf("Cuda error: %s: %s.\n", msg, hipGetErrorString( err) );
    fflush(NULL);
    exit(EXIT_FAILURE);
  }
}

//===============================================================================================================================================================================================================200
//      END SET_DEVICE CODE
//===============================================================================================================================================================================================================200

#endif
