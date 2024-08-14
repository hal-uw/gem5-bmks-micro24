//	(c) 2019 Colin Vanden Heuvel. All rights reserved.
//
//	For the exclusive use of the University of Wisconsin - Madison.
//
//
//	STUDENTS SHOULD NOT ALTER THIS FILE UNLESS INSTRUCTED TO DO SO
//


// Guard for multiple includes
#ifndef ME759_RANDOMS
#define ME759_RANDOMS

// Define size_t
#include <stddef.h>

// Force C-style function linkage for C++ applications
#ifdef __cplusplus
extern "C" {
#endif

// Behavioral notes:
// 	- All randoms_* below generate numbers in the range [amin, amax] inclusive
//	- A seed of UINT_MAX will use the current system time as the seed 

void random_floats(float* a, float amin, float amax, size_t n, unsigned int seed);
void random_doubles(double* a, double amin, double amax, size_t n, unsigned int seed);
void random_ints(int* a, int amin, int amax, size_t n, unsigned int seed);

// END C-style function linkage
#ifdef __cplusplus
}
#endif

// END guard for multiple includes
#endif
