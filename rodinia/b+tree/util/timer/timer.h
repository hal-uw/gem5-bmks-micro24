#ifndef TIMER_H_
#define TIMER_H_

//===============================================================================================================================================================================================================200
  //    TIMER HEADER
  //===============================================================================================================================================================================================================200

  //======================================================================================================================================================150
  //    INCLUDE/DEFINE
  //======================================================================================================================================================150
  #include <stdlib.h>
  #include <sys/time.h>
  
  //======================================================================================================================================================150
  //    FUNCTION PROTOTYPES
  //======================================================================================================================================================150

  long long 
  get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
  }

  //===============================================================================================================================================================================================================200
  //    END TIMER HEADER
  //===============================================================================================================================================================================================================200

#endif
