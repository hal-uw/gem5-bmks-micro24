/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _COMMON_H
#define _COMMON_H

size_t numStreams;
bool individualGpus;

hipStream_t *hip_stream;

//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

////hip timing macros
//#define CTIMERINIT  hipEvent_t cstart, cstop; \
//                    hipEventCreate(&cstart); \
//                    hipEventCreate(&cstop); \
//                    float elapsedTime
//#define CTIMERSTART(cstart) hipEventRecord(cstart,0)
//#define CTIMERSTOP(cstop) hipEventRecord(cstop,0); \
//                          hipEventSynchronize(cstop); \
//                          hipEventElapsedTime(&elapsedTime, cstart, cstop)

//divide and round up macro
#define DIVANDRND(a, b) ((((a) % (b)) != 0) ? ((a) / (b) + 1) : ((a) / (b)))

#  define hipCheckError( msg ) {                                            \
    hipError_t err = hipGetLastError();                                     \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "%s: %i: %s: %s.\n",                                \
                __FILE__, __LINE__, msg, hipGetErrorString( err) );         \
        exit(-1);                                                           \
    } }

#  define hipCheckAsyncError( msg ) {                                       \
      hipDeviceSynchronize();                                               \
      hipCheckError( msg );                                                 \
    }


#endif
