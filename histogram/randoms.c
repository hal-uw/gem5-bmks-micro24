//	(c) 2019 Colin Vanden Heuvel. All rights reserved.
//	
//	For the exclusive use of the University of Wisconsin - Madison.
//	
//	
//	STUDENTS SHOULD NOT ALTER THIS FILE UNLESS INSTRUCTED TO DO SO
//	


#include <stddef.h>
#include <stdlib.h>
#include <time.h>

// Force C-style function linkage for C++ applications
#ifdef __cplusplus
extern "C" {
#endif

//unsigned int _globally_seeded = 0;

void random_floats(
	float* a, float amin, float amax, size_t n, unsigned int seed
) {
	
	// If seed is -1, use the system time instead
	if ((signed)seed == -1) {
		seed = time(NULL);
	}
	
//	if (_globally_seeded == 0) {
		srand(seed);
//		_globally_seeded = 1;
//	}
	
	float range = amax - amin;
	
	for (size_t i = 0; i < n; i++) {
		a[i] = amin + (rand() / (RAND_MAX / range));
	}
}

void random_doubles(
	double* a, double amin, double amax, size_t n, unsigned int seed
) {
	// If seed is -1, use the system time instead
	if ((signed)seed == -1) {
		seed = time(NULL);
	}
	
//	if (_globally_seeded == 0) {
		srand(seed);
//		_globally_seeded = 1;
//	}
	
	double range = amax - amin;
	
	for (size_t i = 0; i < n; i++) {
		a[i] = amin + (rand() / (RAND_MAX / range));
	}
}

void random_ints(
	int* a, int amin, int amax, size_t n, unsigned int seed
) {
	
	// If seed is -1, use the system time instead
	if ((signed)seed == -1) {
		seed = time(NULL);
	}

//	if (_globally_seeded == 0) {
		srand(seed);
//		_globally_seeded = 1;
//	}

	// For integer randoms, the likelihood of generating exactly amax
	// is astronomically low, so we're going to help it along
	amax++;
	
	int range = amax - amin;
	
	for (size_t i = 0; i < n; i++) {
		a[i] = amin + (rand() / (RAND_MAX / range));
		if (a[i] == amax) {
			a[i] -= 1;
		}
	}
}

// END C-style function linkage
#ifdef __cplusplus
}
#endif
