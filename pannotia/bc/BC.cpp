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
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
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
#include <algorithm>
#include "BC.h"
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

#ifdef GEM5_FUSION
#define MAX_ITERS 150
#else
#include <stdint.h>
#define MAX_ITERS INT32_MAX
#endif

void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar = NULL;
    bool mode_set = false;
    bool create_mmap = false;
    bool use_mmap = false;
    bool directed = 1;

    int num_nodes;
    int num_edges;

    int opt;
    hipError_t err;

    // Input arguments
    while ((opt = getopt(argc, argv, "f:hm:")) != -1) {
        switch (opt) {
        case 'f': // Input file name
            tmpchar = optarg;
            break;
        case 'h': // Help
            fprintf(stderr, "SWITCHES\n");
            fprintf(stderr, "\t-f [file name]\n");
            fprintf(stderr, "\t\tinput file name\n");
            fprintf(stderr, "\t-m [mode]\n");
            fprintf(stderr, "\t\toperation mode: default (run without mmap), generate, usemmap\n");
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
        default:
            fprintf(stderr, "Unrecognized switch: -%c\n", opt);
            exit(1);
        }
    }

    if (!(mode_set || create_mmap || use_mmap)) {
        fprintf(stderr, "Execution mode not specified! Use -h for help\n");
        exit(1);
    } else if (use_mmap && tmpchar != NULL) {
        fprintf(stdout, "Ignoring input file specifiers\n");
    } else if ((mode_set || create_mmap) && tmpchar == NULL) {
        fprintf(stderr, "Input file not specified! Use -h for help\n");
        exit(1);
    }

    csr_array *csr;

    if (use_mmap) {
        printf("Using an mmap!\n");

        // get num_nodes
        int fd = open("row_mmap-bc.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! row_mmap-bc.bin is missing!\n");
            exit(1);
        }

        int offset = 0;
        num_nodes = *((int *)mmap(NULL, 1 * sizeof(int), PROT_READ, MAP_PRIVATE, fd, offset));

        // read row_array in
        int *row_array_map = (int *)mmap(NULL, (num_nodes + 2) * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (row_array_map == MAP_FAILED) {
            fprintf(stderr, "row mmap failed!\n");
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
        fd = open("col_mmap-bc.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! col_mmap-bc.bin is missing!\n");
            exit(1);
        }

        offset = 0;
        num_edges = *((int *)mmap(NULL, 1 * sizeof(int), PROT_READ, MAP_PRIVATE, fd, offset));

        // read row_array in
        int *col_array_map = (int *)mmap(NULL, (num_edges + 1) * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (col_array_map == MAP_FAILED) {
            fprintf(stderr, "col mmap failed!\n");
            exit(1);
        }

        // Copy col_array
        int *col_array = (int *)malloc(num_edges * sizeof(int));
        memcpy(col_array, &col_array_map[1], num_edges * sizeof(int));

        munmap(col_array_map, (num_edges + 1) * sizeof(int));
        close(fd);

        fd = open("row_t_mmap-bc.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! row_t_mmap-bc.bin is missing!\n");
            exit(1);
        }

        offset = 0;

        // read row_t_array in
        int *row_array_t_map = (int *)mmap(NULL, (num_nodes + 1) * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (row_array_t_map == MAP_FAILED) {
            fprintf(stderr, "row_t mmap failed!\n");
            exit(1);
        }

        // Copy row_t_array        
        int *row_array_t = (int *)malloc((num_nodes + 1) * sizeof(int));
        memcpy(row_array_t, row_array_t_map, (num_nodes + 1) * sizeof(int));

        munmap(row_array_t_map, (num_nodes + 1) * sizeof(int));
        close(fd);

        fd = open("col_t_mmap-bc.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! col_t_mmap-bc.bin is missing!\n");
            exit(1);
        }

        offset = 0;

        // read col_t_array in
        int *col_array_t_map = (int *)mmap(NULL, num_edges * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (col_array_t_map == MAP_FAILED) {
            fprintf(stderr, "col_t mmap failed!\n");
            exit(1);
        }

        // Copy col_t_array
        int *col_array_t = (int *)malloc(num_edges * sizeof(int));
        memcpy(col_array_t, col_array_t_map, num_edges * sizeof(int));

        munmap(col_array_t_map, num_edges * sizeof(int));
        close(fd);
        
        memset(csr, 0, sizeof(csr_array));
        csr->row_array = row_array;
        csr->col_array = col_array;
        csr->row_array_t = row_array_t;
        csr->col_array_t = col_array_t;
        
        close(fd);
    } else {
        // Parse graph and store it in a CSR format
        csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);

        if (create_mmap) {
            printf("creating an mmap\n");

            // prints csr to file
            std::ofstream row_out("row_mmap-bc.bin", std::ios::binary);

            row_out.write((char *)&num_nodes, sizeof(int));
            row_out.write((char *)csr->row_array, (num_nodes + 1) * sizeof(int));

            row_out.close();

            std::ofstream col_out("col_mmap-bc.bin", std::ios::binary);

            col_out.write((char *)&num_edges, sizeof(int));
            col_out.write((char *)csr->col_array, num_edges * sizeof(int));

            col_out.close();

            std::ofstream row_t_out("row_t_mmap-bc.bin", std::ios::binary);

            row_t_out.write((char *)csr->row_array_t, (num_nodes + 1) * sizeof(int));

            row_t_out.close();

            std::ofstream col_t_out("col_t_mmap-bc.bin", std::ios::binary);

            col_t_out.write((char *)csr->col_array_t, num_edges * sizeof(int));

            col_t_out.close();
            
            free(csr->row_array);
            free(csr->col_array);
            free(csr->data_array);
            free(csr->row_array_t);
            free(csr->col_array_t);
            free(csr->data_array_t);
            free(csr);
            printf("mmaps created!\n");
            return 0;
        }
    }

    // Allocate the bc host array
    float *bc_h = (float *)malloc(num_nodes * sizeof(float));
    printf("num_nodes: %d", num_nodes );
    if (!bc_h) fprintf(stderr, "malloc failed bc_h\n");

    // Create device-side buffers
    float *bc_d, *sigma_d, *rho_d;
    int *dist_d, *stop_d;
    int *row_d, *col_d, *row_trans_d, *col_trans_d;

    // Create betweenness centrality buffers
    err = hipMalloc(&bc_d, num_nodes * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc bc_d %s\n", hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&dist_d, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc dist_d %s\n", hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&sigma_d, num_nodes * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc sigma_d %s\n", hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&rho_d, num_nodes * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc rho_d %s\n", hipGetErrorString(err));
        return -1;
    }

    // Create termination variable buffer
    err = hipMalloc(&stop_d, sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc stop_d %s\n", hipGetErrorString(err));
        return -1;
    }

    // Create graph buffers
    err = hipMalloc(&row_d, (num_nodes + 1) * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc row_d %s\n", hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&col_d, num_edges * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc col_d %s\n", hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&row_trans_d, (num_nodes + 1) * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc row_trans_d %s\n", hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&col_trans_d, num_edges * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc col_trans_d %s\n", hipGetErrorString(err));
        return -1;
    }

    //double timer1, timer2;
    //double timer3, timer4;

    //timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy data to device-side buffers
    err = hipMemcpy(row_d, csr->row_array, (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy row_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(col_d, csr->col_array, num_edges * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy col_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    // Copy data to device-side buffers
    err = hipMemcpy(row_trans_d, csr->row_array_t, (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy row_trans_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(col_trans_d, csr->col_array_t, num_edges * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy col_trans_d (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }

    //timer3 = gettime();

    // Set up kernel dimensions
    int local_worksize = 128;
    dim3 threads(local_worksize, 1, 1);
    int num_blocks = (num_nodes + local_worksize - 1) / local_worksize;
    dim3 grid(num_blocks, 1, 1);
    hipStream_t hip_stream;
#ifdef GEM5_FUSION
    hipStreamCreateWithFlags(&hip_stream, 0x01, -1);
#else
    hipStreamCreate(&hip_stream);
#endif
    int count = 1;
    // Initialization
    hipLaunchKernelGGL(HIP_KERNEL_NAME(clean_bc), dim3(grid), dim3(threads ), 0, hip_stream, bc_d, num_nodes);
#ifdef GEM5_FUSION
    hipHccModuleRingDoorbell(hip_stream);
#endif
    hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
    m5_dump_reset_stats(0, 0);
#endif

    // Main computation loop
    for (int i = 0; i < num_nodes && i < MAX_ITERS; i++) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(clean_1d_array), dim3(grid), dim3(threads ), 0, hip_stream, i, dist_d, sigma_d, rho_d,
                               num_nodes);
#ifdef GEM5_FUSION
        hipHccModuleRingDoorbell(hip_stream);
#endif
        hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
        m5_dump_reset_stats(0, 0);
#endif

        // Depth of the traversal
        int dist = 0;
        // Termination variable
        int stop = 1;

        // Traverse the graph from the source node i
        do {
            stop = 0;

            // Copy the termination variable to the device
            hipMemcpy(stop_d, &stop, sizeof(int), hipMemcpyHostToDevice);

            hipLaunchKernelGGL(HIP_KERNEL_NAME(bfs_kernel), dim3(grid), dim3(threads ), 0, hip_stream, row_d, col_d, dist_d, rho_d, stop_d,
                                   num_nodes, num_edges, dist);
#ifdef GEM5_FUSION
            hipHccModuleRingDoorbell(hip_stream);
#endif
            hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
            m5_dump_reset_stats(0, 0);
#endif
            // Copy back the termination variable from the device
            hipMemcpy(&stop, stop_d, sizeof(int), hipMemcpyDeviceToHost);

            // Another level
            dist++;

        } while (stop);

        hipDeviceSynchronize();

        // Traverse back from the deepest part of the tree
        while (dist) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(backtrack_kernel), dim3(grid), dim3(threads ), 0, hip_stream, row_trans_d, col_trans_d,
                                   dist_d, rho_d, sigma_d,
                                   num_nodes, num_edges, dist, i,
                                   bc_d);
#ifdef GEM5_FUSION
            hipHccModuleRingDoorbell(hip_stream);
#endif
            hipStreamSynchronize(hip_stream);
#ifdef GEM5_FUSION
            m5_dump_reset_stats(0, 0);
#endif
            // Back one level
            dist--;
        }
        hipDeviceSynchronize();
        fprintf(stdout, "Completed iteration %d\n", i);
    }
    hipDeviceSynchronize();
    //timer4 = gettime();

    // Copy back the results for the bc array
    err = hipMemcpy(bc_h, bc_d, num_nodes * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: read buffer bc_d (%s)\n", hipGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    //timer2 = gettime();

    //printf("kernel + memcopy time = %lf ms\n", (timer4 - timer3) * 1000);
    //printf("kernel execution time = %lf ms\n", (timer2 - timer1) * 1000);

#if 1
    //dump the results to the file
    print_vectorf(bc_h, num_nodes);
#endif

    // Clean up the host-side buffers
    free(bc_h);
    free(csr->row_array);
    free(csr->col_array);
    free(csr->data_array);
    free(csr->row_array_t);
    free(csr->col_array_t);
    free(csr->data_array_t);
    free(csr);

    // Clean up the device-side buffers
    hipFree(bc_d);
    hipFree(dist_d);
    hipFree(sigma_d);
    hipFree(rho_d);
    hipFree(stop_d);
    hipFree(row_d);
    hipFree(col_d);
    hipFree(row_trans_d);
    hipFree(col_trans_d);

    return 0;
}

void print_vector(int *vector, int num)
{
    for (int i = 0; i < num; i++)
        printf("%d: %d \n", i + 1, vector[i]);
    printf("\n");
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