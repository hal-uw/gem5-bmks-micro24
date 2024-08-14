/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * Copyright (c) 2021 Gaurav Jain and Matthew D. Sinclair                           *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR"�) (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <sys/time.h>
#include "../graph_parser/parse.h"
#include "../graph_parser/util.h"
#include "kernel.h"
#include <unistd.h>
#include <sys/mman.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef GEM5_FUSION
#include <stdint.h>
#include <gem5/m5ops.h>
#endif

// Iteration count
#define ITER 20

void print_vectorf(float *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar = NULL;

    int num_nodes;
    int num_edges;
    int file_format = -1;
    bool directed = 0;
    bool mode_set = false;
    bool create_mmap = false;
    bool use_mmap = false;
    
    int opt;
    hipError_t err = hipSuccess;

    while ((opt = getopt(argc, argv, "f:hm:t:")) != -1) {
        switch (opt) {
            case 'f': // Input file name
                tmpchar = optarg;
                break;
            case 'h': // Help
                fprintf(stderr, "SWITCHES\n\t-f [file name]\n\t\tinput file name\n");
                fprintf(stderr, "\t-m [mode]\n\t\toperation mode: default (run without mmap), generate, usemmap\n");
                fprintf(stderr, "\t-t [file type] \n\t\tfile type (not required when running in usemmap mode): dimacs9 (0), metis (1), matrixmarket (2)\n");
                exit(0);
            case 'm':  // Mode
                if (strcmp(optarg, "default") == 0 || optarg[0] == '0') {
                    mode_set = true;
                } else if (strcmp(optarg, "generate") == 0 || optarg[0] == '1') {
                    create_mmap = true;
                } else if (strcmp(optarg, "usemmap") == 0 || optarg[0] == '2') {
                    use_mmap = true;
                } else {
                    fprintf(stderr, "Unrecognized mode: %s\n", optarg);
                    exit(1);
                }
                break;
            case 't':  // Input file type
                if (strcmp(optarg, "dimacs9") == 0 || optarg[0] == '0') {
                    file_format = 0;
                } else if (strcmp(optarg, "metis") == 0 || optarg[0] == '1') {
                    file_format = 1;
                } else if (strcmp(optarg, "matrixmarket") == 0 || optarg[0] == '2') {
                    file_format = 2;
                } else {
                    fprintf(stderr, "Unrecognized file type: %s\n", optarg);
                    exit(1);
                }
                break;
            default:
                fprintf(stderr, "Unrecognized switch: -%c\n", opt);
                exit(1);
        }
    }

    if (!(mode_set || create_mmap || use_mmap)) {
        fprintf(stderr, "Execution mode not specified! Use -h for help\n");
        exit(1);
    } else if (use_mmap && (tmpchar != NULL || file_format != -1)) {
        fprintf(stdout, "Ignoring input file specifiers\n");
    } else if ((mode_set || create_mmap) && tmpchar == NULL) {
        fprintf(stderr, "Input file not specified! Use -h for help\n");
        exit(1);
    } else if ((mode_set || create_mmap) && file_format == -1) {
        fprintf(stderr, "Input file type not specified! Use -h for help\n");
        exit(1);
    }
    
    csr_array *csr;

    if (use_mmap) {
        printf("Using an mmap!\n");

        // get num_nodes
        int fd = open("row_mmap-pr.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file!\n");
            exit(1);
        }

        int offset = 0;
        num_nodes = *((int *)mmap(NULL, 1 * sizeof(int), PROT_READ, MAP_PRIVATE, fd, offset));

        // read row_array in
        int *row_array_map = (int *)mmap(NULL, (num_nodes + 2) * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (row_array_map == MAP_FAILED) {
            fprintf(stderr, "mmap failed!\n");
            exit(1);
        }

        // Copy row_array
        csr = (csr_array *)malloc(sizeof(csr_array));
        if (csr == NULL) {
            printf("csr_array malloc failed!\n");
            exit(1);
        }

        int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
        memcpy(row_array, &row_array_map[1], (num_nodes + 1) * sizeof(int));

        munmap(row_array_map, (num_nodes + 2) * sizeof(int));
        close(fd);

        // get num_edges
        fd = open("col_mmap-pr.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file!\n");
            exit(1);
        }

        offset = 0;
        num_edges = *((int *)mmap(NULL, 1 * sizeof(int), PROT_READ, MAP_PRIVATE, fd, offset));

        // read row_array in
        int *col_array_map = (int *)mmap(NULL, (num_edges + 1) * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (col_array_map == MAP_FAILED) {
            fprintf(stderr, "mmap failed!\n");
            exit(1);
        }

        // Copy row_array
        int *col_array = (int *)malloc(num_edges * sizeof(int));
        memcpy(col_array, &col_array_map[1], num_edges * sizeof(int));

        munmap(col_array_map, (num_edges + 1) * sizeof(int));
        close(fd);

        memset(csr, 0, sizeof(csr_array));
        csr->row_array = row_array;
        csr->col_array = col_array;
    } else {
        // Parse graph files into csr structure
        if (file_format == 1) {
            // Metis
            csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
        } else if (file_format == 0) {
            // Dimacs9
            csr = parseCOO(tmpchar, &num_nodes, &num_edges, 1);
        } else if (file_format == 2) {
            // Matrix market
            csr = parseMM(tmpchar, &num_nodes, &num_edges, directed, 0);
        } else {
            printf("reserve for future");
            exit(1);
        }
    
        if (create_mmap) {
            printf("creating an mmap\n");

            // prints csr to file
            std::ofstream row_out("row_mmap-pr.bin", std::ios::binary);

            row_out.write((char *)&num_nodes, sizeof(int));
            row_out.write((char *)csr->row_array, (num_nodes + 1) * sizeof(int));

            row_out.close();

            // num_edges * sizeof(int)
            std::ofstream col_out("col_mmap-pr.bin", std::ios::binary);

            col_out.write((char *)&num_edges, sizeof(int));
            col_out.write((char *)csr->col_array, num_edges * sizeof(int));

            col_out.close();

            csr->freeArrays();
            free(csr);
            printf("mmaps created!\n");
            return 0;
        }
    }

    // Allocate rank_array
    float *rank_array = (float *)malloc(num_nodes * sizeof(float));
    if (!rank_array) {
        fprintf(stderr, "rank array not allocated successfully\n");
        return -1;
    }

    int *row_d = NULL;
    int *col_d = NULL;
    //int *data_d = NULL;

    float *pagerank1_d = NULL;
    float *pagerank2_d = NULL;

    // Create device-side buffers for the graph
    int *row_d = (int *) malloc (num_nodes * sizeof(int));
    int *col_d = (int *) malloc(num_edges * sizeof(int));
    int *data_d = (int *) malloc(num_edges * sizeof(int));
    float *pagerank1_d = (float *) malloc(num_nodes * sizeof(float));
    float *pagerank2_d = (float *) malloc(num_nodes * sizeof(float));

//    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy the data to the device-side buffers
    memcpy(row_d, csr->row_array, num_nodes * sizeof(int));
    memcpy(col_d, csr->col_array, num_edges * sizeof(int));

    // Set up work dimensions
    int block_size  = 256;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    dim3 threads(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    hipStream_t hip_stream;
#ifdef GEM5_FUSION
    hipStreamCreateWithFlags(&hip_stream, 0x01, -1);
#else
    hipStreamCreate(&hip_stream);
#endif
    int count = 1;
    //    double timer3 = gettime();

#ifdef GEM5_FUSION
    m5_getKernelArg(reinterpret_cast<uintptr_t>(row_d), reinterpret_cast<uintptr_t>(pagerank1_d), reinterpret_cast<uintptr_t>(pagerank2_d), 0b111100 , 3, count++);
#endif
    // Launch the initialization kernel
    hipLaunchKernelGGL(HIP_KERNEL_NAME(inibuffer), dim3(grid), dim3(threads), 0, hip_stream, row_d, pagerank1_d, pagerank2_d, num_nodes,
                       num_edges);
#ifdef GEM5_FUSION
    hipHccModuleRingDoorbell(hip_stream);
#endif
    hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
    m5_dump_reset_stats(0, 0);
#endif
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n", hipGetErrorString(err));
      return -1;
    }

    // Run PageRank for some iter. TO: convergence determination
    for (int i = 0; i < ITER; i++) {
      // Launch pagerank kernel 1
#ifdef GEM5_FUSION
      m5_getKernelArg(reinterpret_cast<uintptr_t>(row_d), reinterpret_cast<uintptr_t>(pagerank1_d), reinterpret_cast<uintptr_t>(pagerank2_d), 0b110000 , 3, count);
      m5_getKernelArg(reinterpret_cast<uintptr_t>(col_d), 0, 0, 0b0000 , 1, count++);
#endif
      hipLaunchKernelGGL(HIP_KERNEL_NAME(pagerank1), dim3(grid), dim3(threads), 0, hip_stream, row_d, col_d, pagerank1_d,
                         pagerank2_d, num_nodes, num_edges);
#ifdef GEM5_FUSION
      hipHccModuleRingDoorbell(hip_stream);
#endif
      hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
      m5_dump_reset_stats(0, 0);
      m5_getKernelArg(reinterpret_cast<uintptr_t>(pagerank1_d), reinterpret_cast<uintptr_t>(pagerank2_d), 0, 0b111100, 3, count);
#endif
      // Launch pagerank kernel 2
      hipLaunchKernelGGL(HIP_KERNEL_NAME(pagerank2), dim3(grid), dim3(threads), 0, hip_stream, pagerank1_d,
                         pagerank2_d, num_nodes);
#ifdef GEM5_FUSION
      hipHccModuleRingDoorbell(hip_stream);
#endif
      hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
      m5_dump_reset_stats(0, 0);
#endif
    }
    hipDeviceSynchronize();

//    double timer4 = gettime();

    // Copy the rank buffer back
    err = hipMemcpy(rank_array, pagerank1_d, num_nodes * sizeof(float), hipMemcpyDeviceToHost);

    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy() failed (%s)\n", hipGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

//    double timer2 = gettime();

    // Report timing characteristics
//    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
//    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

#if 1
    // Print rank array
    print_vectorf(rank_array, num_nodes);
#endif

    // Free the host-side arrays
    free(rank_array);
    csr->freeArrays();
    free(csr);

    // Free the device buffers
    free(row_d);
    free(col_d);
    free(data_d);

    free(pagerank1_d);
    free(pagerank2_d);

    return 0;

}

void print_vectorf(float *vector, int num)
{
    FILE * fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++) {
        fprintf(fp, "%f\n", vector[i]);
    }

    fclose(fp);
}

