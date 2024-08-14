#ifndef __HIP_HELPERS__
#define __HIP_HELPERS__
/************************************************************************/
/* Init HIP                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitHIP(void){return true;}

#else
bool InitHIP(void)
{
  int count = 0;
  int i = 0;

  hipGetDeviceCount(&count);
  if(count == 0) {
    fprintf(stderr, "There is no device.\n");
    return false;
  }

  for(i = 0; i < count; i++) {
    hipDeviceProp_t prop;
    if(hipGetDeviceProperties(&prop, i) == hipSuccess) {
      if(prop.major >= 1) {
        break;
      }
    }
  }
  if(i == count) {
    fprintf(stderr, "There is no device supporting HIP.\n");
    return false;
  }
  hipSetDevice(i);

  printf("HIP initialized.\n");
  return true;
}
#endif
#endif
