/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * Copyright (c) 2021 Gaurav Jain and Matthew D. Sinclair                           *
 * Copyright (c) 2024 James Braun and Matthew D. Sinclair                           *
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
#include <sys/time.h>
#include <algorithm>
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
#include <gem5/m5ops.h>
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define BIGNUM 99999999

void print_vector(int *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar = NULL;
    bool mode_set = false;
    bool create_mmap = false;
    bool use_mmap = false;
    bool directed = 0;

    int num_nodes;
    int num_edges;
    int file_format = 1;

    int opt;
    hipError_t err = hipSuccess;

    // Input arguments
    while ((opt = getopt(argc, argv, "df:hm:t:")) != -1) {
        switch (opt) {
        case 'd': // Directed graph
            directed = 1;
            break;
        case 'f': // Input file name
            tmpchar = optarg;
            break;
        case 'h': // Help
            fprintf(stderr, "SWITCHES\n");
            fprintf(stderr, "\t-d\n");
            fprintf(stderr, "\t\tdirected graph (default is not directed)\n");
            fprintf(stderr, "\t-f [file name]\n");
            fprintf(stderr, "\t\tinput file name\n");
            fprintf(stderr, "\t-m [mode]\n");
            fprintf(stderr, "\t\toperation mode: default (run without mmap), generate, usemmap\n");
            fprintf(stderr, "\t-t [file type] \n");
            fprintf(stderr, "\t\tfile type (not required when running in usemmap mode): dimacs9 (0), metis (1), matrixmarket (2)\n");
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

    // Allocate the csr structure
    csr_array *csr;

    if (use_mmap) {
        printf("Using an mmap!\n");

        // get num_nodes
        int fd = open("row_mmap-sssp_ell.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! row_mmap-sssp_ell.bin is missing!\n");
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

        csr = (csr_array *)malloc(sizeof(csr_array));
        if (csr == NULL) {
            printf("csr_array malloc failed!\n");
            exit(1);
        }

        // Copy row_array
        int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
        memcpy(row_array, &row_array_map[1], (num_nodes + 1) * sizeof(int));

        munmap(row_array_map, (num_nodes + 2) * sizeof(int));
        close(fd);

        // get num_edges
        fd = open("col_mmap-sssp_ell.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! col_mmap-sssp_ell.bin is missing!\n");
            exit(1);
        }

        offset = 0;
        num_edges = *((int *)mmap(NULL, 1 * sizeof(int), PROT_READ, MAP_PRIVATE, fd, offset));

        // read col_array in
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

        fd = open("data_mmap-sssp_ell.bin", std::ios::binary | std::fstream::in);
        if (fd == -1) {
            fprintf(stderr, "error: %s\n", strerror(errno));
            fprintf(stderr, "You need to create an mmapped input file! data_mmap-sssp_ell.bin is missing!\n");
            exit(1);
        }

        // read data_array in
        int *data_array_map = (int *)mmap(NULL, num_edges * sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, offset);

        // Check that maping was sucessful
        if (data_array_map == MAP_FAILED) {
            fprintf(stderr, "data mmap failed!\n");
            exit(1);
        }

        // Copy data_array
        int *data_array = (int *)malloc(num_edges * sizeof(int));
        memcpy(data_array, data_array_map, num_edges * sizeof(int));

        munmap(data_array_map, num_edges * sizeof(int));
        close(fd);

        memset(csr, 0, sizeof(csr_array));
        csr->row_array = row_array;
        csr->col_array = col_array;
        csr->data_array = data_array;

    } else {
        // Parse graph file and store into a CSR format
        if (file_format == 1)
            csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
        else if (file_format == 0)
            csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);
        else {
            printf("reserve for future");
            exit(1);
        }

        if (create_mmap) {
            printf("creating an mmap\n");

            // prints csr to file
            std::ofstream row_out("row_mmap-sssp_ell.bin", std::ios::binary);

            row_out.write((char *)&num_nodes, sizeof(int));
            row_out.write((char *)csr->row_array, (num_nodes + 1) * sizeof(int));

            row_out.close();

            // num_edges * sizeof(int)
            std::ofstream col_out("col_mmap-sssp_ell.bin", std::ios::binary);

            col_out.write((char *)&num_edges, sizeof(int));
            col_out.write((char *)csr->col_array, num_edges * sizeof(int));

            col_out.close();

            std::ofstream data_out("data_mmap-sssp_ell.bin", std::ios::binary);

            data_out.write((char *)csr->data_array, num_edges * sizeof(int));

            data_out.close();

            csr->freeArrays();
            free(csr);
            printf("mmaps created!\n");
            return 0;
        }
    }

    // Allocate ell and transform from csr
    ell_array *ell = csr2ell(csr, num_nodes, num_edges, BIGNUM);
    int height = ell->max_height;

    // Allocate the cost array
    int *cost_array = (int *)malloc(num_nodes * sizeof(int));
    if (!cost_array) fprintf(stderr, "malloc failed cost_array\n");

    // Set the cost array to zero
    for (int i = 0; i < num_nodes; i++) {
        cost_array[i] = 0;
    }

    // Create device-side buffers
	int **ell_col_d;
	int **ell_data_d;
	int **vector_d1;
	int **vector_d2;
	int **stop_d;

	//int numStreams = 1;
	size_t numStreams = 4; //atoi(argv[2]);

	hipMalloc( (void**) &ell_col_d, sizeof(int *)*numStreams) ;
	hipMalloc( (void**) &ell_data_d, sizeof(int *)*numStreams) ;
	hipMalloc( (void**) &vector_d1, sizeof(int *)*numStreams) ;
	hipMalloc( (void**) &vector_d2, sizeof(int *)*numStreams) ;
	hipMalloc( (void**) &stop_d, sizeof(int *)*numStreams) ;

    // Create the device-side graph structure
	/*err = hipMalloc(&ell_col_d, height * num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc ell_col_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&ell_data_d, height * num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc ell_data_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }

    // Termination variable
    err = hipMalloc(&stop_d, sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc stop_d (size:%d) => %s\n", 1, hipGetErrorString(err));
        return -1;
    }

    // Create the device-side buffers for sssp
    err = hipMalloc(&vector_d1, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc vector_d1 (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
    }
    err = hipMalloc(&vector_d2, num_nodes * sizeof(int));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc vector_d2 (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
        return -1;
	}*/

	for (int i = 0; i < numStreams; ++i) {
		err = hipMalloc((void **)&vector_d1[i], sizeof(int) * num_nodes);
		if (err != hipSuccess) {
			fprintf(stderr, "ERROR: hipMalloc vector_d1 (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
			return -1;
		}
	}

	for (int i = 0; i   < numStreams; ++i) {
		err = hipMalloc((void **)&vector_d2[i], sizeof(int) * num_nodes);
		if (err != hipSuccess) {
			fprintf(stderr, "ERROR: hipMalloc vector_d2 (size:%d) => %s\n", num_nodes, hipGetErrorString(err));
			return -1;
		}
    }

    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy data to device side buffers
	/*err = hipMemcpy(ell_col_d, ell->col_array, height * num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy ell_col_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
    }

    err = hipMemcpy(ell_data_d, ell->data_array, height * num_nodes * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy ell_data_d (size:%d) => %s\n", height * num_nodes, hipGetErrorString(err));
        return -1;
	  }*/

	for (int i = 0; i < numStreams; ++i) {
		hipMalloc((void **)&ell_col_d[i], sizeof(int) * height * num_nodes);
		hipMemcpy( ell_col_d[i], ell->col_array, sizeof(int) * height * num_nodes, hipMemcpyHostToDevice) ;
	}

	for (int i = 0; i < numStreams; ++i) {
		hipMalloc((void **)&ell_data_d[i], sizeof(int) * height * num_nodes);
		hipMemcpy( ell_data_d[i], ell->data_array, sizeof(int) * height * num_nodes, hipMemcpyHostToDevice) ;
	}
	hipStream_t hip_stream[numStreams];
	bool stop_perStream[numStreams];

    // Work dimensions
    int block_size = 64;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    dim3 threads(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    // Source vertex 0
    int sourceVertex = 0;

	int stop = 1;
	int cnt = 0;

    int count = 1;
	//double timer3 = gettime();

	for(int i = 0; i < numStreams; i++) {

		//timer3 = gettime();

		hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1);	

		m5_getKernelArg(reinterpret_cast<uintptr_t>(vector_d1[i]), reinterpret_cast<uintptr_t>(vector_d2[i]), 0, 0b1111 , 2, count++);
    // Launch the initialization kernel
		hipLaunchKernelGGL_lk(vector_init, dim3(grid), dim3(threads), 0, hip_stream[i], 0, vector_d1[i], vector_d2[i], sourceVertex, num_nodes);
		printf("Launching the Initialization kernel");
	}

	for(int i = 0; i < numStreams; i++) {

		hipHccModuleRingDoorbell(hip_stream[i]);
	}
	for(int i = 0; i < numStreams; i++) {

		hipStreamSynchronize(hip_stream[i]);
	}
    m5_dump_reset_stats(0, 0);
	printf("Dumped stats\n");

    hipDeviceSynchronize();

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: vector_init failed (%s)\n", hipGetErrorString(err));
        return -1;
    }


    // Main computation loop
	for (int n = 1; n < num_nodes; n++) {
        // Reset the termination variable
        stop = 0;

		printf("Main computation loop");


        // Copy the termination variable to the device
		/* err = hipMemcpy(stop_d, &stop, sizeof(int), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: write stop_d (%s)\n", hipGetErrorString(err));
            return -1;
		   }*/
		for(int i = 0; i < numStreams; i++) {

			printf("Main computation stream:%d\n", i);

			stop_perStream[i] = false;

			hipMalloc((void **)&stop_d[i], sizeof(int));
			hipMemcpy( stop_d[i], &stop_perStream, sizeof(int), hipMemcpyHostToDevice) ;
			printf("Copied stop | stream: %d\n", i);

			m5_getKernelArg(reinterpret_cast<uintptr_t>(vector_d1[i]), reinterpret_cast<uintptr_t>(vector_d2[i]), 0, 0b0011 , 2, count++);
        // Launch the assignment kernel
			hipLaunchKernelGGL_lk(vector_assign, dim3(grid), dim3(threads), 0, hip_stream[i], 0, vector_d1[i], vector_d2[i], num_nodes);
		}
		for(int i = 0; i < numStreams; i++) {

			hipHccModuleRingDoorbell(hip_stream[i]);
		}
		for(int i = 0; i < numStreams; i++) {

			hipStreamSynchronize(hip_stream[i]);
		}
        m5_dump_reset_stats(0, 0);

		for(int i = 0; i < numStreams; i++) {

			m5_getKernelArg(reinterpret_cast<uintptr_t>(ell_col_d[i]), reinterpret_cast<uintptr_t>(ell_data_d[i]), reinterpret_cast<uintptr_t>(vector_d1[i]), 0b000000 , 3, count);
			m5_getKernelArg(reinterpret_cast<uintptr_t>(vector_d2[i]), 0, 0, 0b11 , 1, count++);
        // Launch the min.+ kernel
			hipLaunchKernelGGL_lk(ell_min_dot_plus_kernel, dim3(grid), dim3(threads), 0, hip_stream[i], 0, num_nodes, height,
					ell_col_d[i], ell_data_d[i],
					vector_d1[i], vector_d2[i]);
		}
		for(int i = 0; i < numStreams; i++) {
			hipHccModuleRingDoorbell(hip_stream[i]);
		}
		for(int i = 0; i < numStreams; i++) {

			hipStreamSynchronize(hip_stream[i]);
		}
        m5_dump_reset_stats(0, 0);

		for(int i = 0; i < numStreams; i++) {

        // Launch the check kernel
			m5_getKernelArg(reinterpret_cast<uintptr_t>(vector_d1[i]), reinterpret_cast<uintptr_t>(vector_d2[i]), reinterpret_cast<uintptr_t>(stop_d[i]), 0b110000 , 3, count++);
			hipLaunchKernelGGL_lk(vector_diff, dim3(grid), dim3(threads), 0, hip_stream[i], 0, vector_d1[i], vector_d2[i],
					stop_d[i], num_nodes);
		}
		for(int i = 0; i < numStreams; i++) {

			hipHccModuleRingDoorbell(hip_stream[i]);
		}
		for(int i = 0; i < numStreams; i++) {

			hipStreamSynchronize(hip_stream[i]);
		}
        m5_dump_reset_stats(0, 0);

		/*err = hipMemcpy(&stop, stop_d, sizeof(int), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: read stop_d (%s)\n", hipGetErrorString(err));
		  return -1;
		  }*/
		err = hipMemcpy(&stop_perStream, stop_d, sizeof(int), hipMemcpyDeviceToHost);
		if (err != hipSuccess) {
			fprintf(stderr, "ERROR: hipMemcpy stop_perStream - %s\n", hipGetErrorString(err));
            return -1;
        }
		// combine all stop signals per stream -- init with first one to avoid false being immediately decided because of init value of stop

		stop = stop_perStream[0];
		for (int i = 1; i < numStreams; ++i) {
			stop &= stop_perStream[i];
		}
        // Exit the loop
        if (stop == 0) {
            break;
        }
        cnt++;
    }
    hipDeviceSynchronize();
	//double timer4 = gettime();

    // Read the cost_array back
	/*err = hipMemcpy(cost_array, vector_d1, num_nodes * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: read vector_d1 (%s)\n", hipGetErrorString(err));
        return -1;
	  }*/
	//Copy only from stream 0
	hipMemcpy(cost_array, vector_d1[0], num_nodes * sizeof(int), hipMemcpyDeviceToHost);

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    double timer2 = gettime();

    // Print the timing statistics
	//printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);
	//printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("number iterations = %d\n", cnt);

#if 1
    // Print cost_array
    print_vector(cost_array, num_nodes);
#endif

    // Clean up the host arrays
    free(cost_array);
    csr->freeArrays();
    free(csr);

    free(ell->col_array);
    free(ell->data_array);
    free(ell);

    // Clean up the device-side buffers
	/*hipFree(ell_col_d);
    hipFree(ell_data_d);
    hipFree(stop_d);
    hipFree(vector_d1);
	  hipFree(vector_d2);*/

    return 0;
}

void print_vector(int *vector, int num)
{

    FILE * fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++)
        fprintf(fp, "%d: %d\n", i + 1, vector[i]);

    fclose(fp);
}
