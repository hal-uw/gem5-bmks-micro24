/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include "hip/hip_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gem5/m5ops.h>

#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
  int starting;
  int no_of_edges;
};

#include "kernel.h"
#include "kernel2.h"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  no_of_nodes=0;
  edge_list_size=0;
  BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){
  fprintf(stderr,"Usage: %s <input_file> <numStreams <individualGPUs>>\n", argv[0]);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
    char *input_f;
    if(argc!=4){
      Usage(argc, argv);
      exit(0);
    }

    input_f = argv[1];
    printf("Reading File\n");
    //Read in Graph from a file
    fp = fopen(input_f,"r");
    if(!fp)
    {
      printf("Error Reading graph file\n");
      return;
    }
    size_t numStreams = atoi(argv[2]);
    bool individualGpus = atoi(argv[3]);

    hipStream_t hip_stream[numStreams];

    int numGpus;
    hipGetDeviceCount(&numGpus);

    for (int i = 0; i < numStreams; i++) {
      if (individualGpus) {
	hipSetDevice(i % numGpus);
      }
      hipStreamCreateWithFlags(&hip_stream[i], 0x01, -1);
    }

    int source = 0;

    fscanf(fp,"%d",&no_of_nodes);

    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    //Make execution Parameters according to the number of nodes
    //Distribute threads across multiple Blocks if necessary
    if(no_of_nodes>MAX_THREADS_PER_BLOCK)
    {
      num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
      num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    // allocate host memory
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

    int start, edgeno;   
    // initalize the memory
    for( unsigned int i = 0; i < no_of_nodes; i++) 
    {
      fscanf(fp,"%d %d",&start,&edgeno);
      h_graph_nodes[i].starting = start;
      h_graph_nodes[i].no_of_edges = edgeno;
      h_graph_mask[i]=false;
      h_updating_graph_mask[i]=false;
      h_graph_visited[i]=false;
    }

    //read the source node from the file
    fscanf(fp,"%d",&source);
    source=0;

    //set the source node as true in the mask
    h_graph_mask[source]=true;
    h_graph_visited[source]=true;

    fscanf(fp,"%d",&edge_list_size);

    int id,cost;
    int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    for(int i=0; i < edge_list_size ; i++)
    {
      fscanf(fp,"%d",&id);
      fscanf(fp,"%d",&cost);
      h_graph_edges[i] = id;
    }

    if(fp) {
      fclose(fp);
    }

    printf("Read File\n");

    //Copy the Node list to device memory -- separate per stream
    Node ** d_graph_nodes;
    hipMalloc( (void**) &d_graph_nodes, sizeof(Node *)*numStreams) ;
    for (int i = 0; i < numStreams; ++i) {
      hipMalloc((void **)&d_graph_nodes[i], sizeof(Node) * no_of_nodes);
      hipMemcpy( &d_graph_nodes[i][0], h_graph_nodes, sizeof(Node)*no_of_nodes, hipMemcpyHostToDevice) ;
    }

    //Copy the Edge List to device Memory
    int ** d_graph_edges;
    hipMalloc( (void**) &d_graph_edges, sizeof(int *)*numStreams) ;
    for (int i = 0; i < numStreams; ++i) {
      hipMalloc( (void**) &d_graph_edges[i], sizeof(int)*edge_list_size) ;
      hipMemcpy( &d_graph_edges[i][0], h_graph_edges, sizeof(int)*edge_list_size, hipMemcpyHostToDevice) ;
    }

    //Copy the Mask to device memory
    bool ** d_graph_mask;
    hipMalloc( (void**) &d_graph_mask, sizeof(bool *)*numStreams) ;
    for (int i = 0; i < numStreams; ++i) {
      hipMalloc( (void**) &d_graph_mask[i], sizeof(bool)*no_of_nodes) ;
      hipMemcpy( &d_graph_mask[i][0], h_graph_mask, sizeof(bool)*no_of_nodes, hipMemcpyHostToDevice) ;
    }

    bool ** d_updating_graph_mask;
    hipMalloc( (void**) &d_updating_graph_mask, sizeof(bool *)*numStreams) ;
    for (int i = 0; i < numStreams; ++i) {
      hipMalloc( (void**) &d_updating_graph_mask[i], sizeof(bool)*no_of_nodes) ;
      hipMemcpy( &d_updating_graph_mask[i][0], h_updating_graph_mask, sizeof(bool)*no_of_nodes, hipMemcpyHostToDevice) ;
    }

    //Copy the Visited nodes array to device memory
    bool ** d_graph_visited;
    hipMalloc( (void**) &d_graph_visited, sizeof(bool *)*numStreams) ;
    for (int i = 0; i < numStreams; ++i) {
      hipMalloc( (void**) &d_graph_visited[i], sizeof(bool)*no_of_nodes) ;
      hipMemcpy( &d_graph_visited[i][0], h_graph_visited, sizeof(bool)*no_of_nodes, hipMemcpyHostToDevice) ;
    }

    // allocate mem for the result on host side
    int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
    for(int i=0;i<no_of_nodes;i++) {
      h_cost[i]=-1;
    }
    h_cost[source]=0;
        
    // allocate device memory for result
    int ** d_cost;
    hipMalloc( (void**) &d_cost, sizeof(int *)*numStreams) ;
    for (int i = 0; i < numStreams; ++i) {
      hipMalloc( (void**) &d_cost[i], sizeof(int)*no_of_nodes);
      hipMemcpy( &d_cost[i][0], h_cost, sizeof(int)*no_of_nodes, hipMemcpyHostToDevice) ;
    }

    //make a bool to check if the execution is over
    bool * d_over;
    hipMalloc( (void**) &d_over, sizeof(bool)*numStreams) ;

    printf("Copied Everything to GPU memory\n");

    // setup execution parameters
    dim3  grid( num_of_blocks, 1, 1);
    dim3  threads( num_of_threads_per_block, 1, 1);

    int k=0;
    printf("Start traversing the tree\n");
    bool stop;
    bool stop_perStream[numStreams];
    int count = 1;
    //Call the Kernel untill all the elements of Frontier are not false
    do
    {
      //if no thread changes this value then the loop stops
      stop=false;

      for (int i = 0; i < numStreams; ++i) {
	stop_perStream[i] = false;

        hipMemcpy( &d_over[i], &stop_perStream[i], sizeof(bool), hipMemcpyHostToDevice) ;
        m5_getKernelArg(reinterpret_cast<uintptr_t>(&d_graph_nodes[i][0]), reinterpret_cast<uintptr_t>(&d_graph_edges[i][0]), reinterpret_cast<uintptr_t>(&d_graph_mask[i][0]), 48 , 3, count);
        m5_getKernelArg(reinterpret_cast<uintptr_t>(&d_updating_graph_mask[i][0]), reinterpret_cast<uintptr_t>(&d_graph_visited[i][0]), reinterpret_cast<uintptr_t>(&d_cost[i][0]), 51, 3, count++);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(Kernel), dim3(grid), dim3(threads), 0 , hip_stream[i], /*0,*/
                              &d_graph_nodes[i][0], &d_graph_edges[i][0], &d_graph_mask[i][0], &d_updating_graph_mask[i][0], &d_graph_visited[i][0], &d_cost[i][0], no_of_nodes);
        // check if kernel execution generated and error

      }

      for(int i = 0; i < numStreams; i++) {
          hipHccModuleRingDoorbell(hip_stream[i]);
          hipStreamSynchronize(hip_stream[i]);
          m5_dump_reset_stats(0, 0);
      }

      for (int i = 0; i < numStreams; ++i) {
  m5_getKernelArg(reinterpret_cast<uintptr_t>(&d_graph_mask[i][0]), reinterpret_cast<uintptr_t>(&d_updating_graph_mask[i][0]), reinterpret_cast<uintptr_t>(&d_graph_visited[i][0]), 63 , 3, count);
  m5_getKernelArg(reinterpret_cast<uintptr_t>(&d_over[i]), 0, 0, 3 , 1, count++);       
	hipLaunchKernelGGL(HIP_KERNEL_NAME(Kernel2), dim3(grid), dim3(threads), 0 , hip_stream[i],/*0,*/
			      &d_graph_mask[i][0], &d_updating_graph_mask[i][0], &d_graph_visited[i][0], &d_over[i], no_of_nodes);
	// check if kernel execution generated and error
      }
                
      for (int i = 0; i < numStreams; i++) {
	hipHccModuleRingDoorbell(hip_stream[i]);
  hipStreamSynchronize(hip_stream[i]);
  m5_dump_reset_stats(0, 0);
      }

      hipMemcpy( &stop_perStream, d_over, sizeof(bool) * numStreams, hipMemcpyDeviceToHost) ;
      
      // combine all stop signals per stream -- init with first one to avoid false being immediately decided because of init value of stop
      stop = stop_perStream[0];
      for (int i = 1; i < numStreams; ++i) {
	stop &= stop_perStream[i];
      }
      
      k++;
    }
    while(stop);


    printf("Kernel Executed %d times\n",k);

    // copy result from device to host -- just copy from stream 0
    hipMemcpy( h_cost, &d_cost[0][0], sizeof(int)*no_of_nodes, hipMemcpyDeviceToHost) ;

    //Store the result into a file
    FILE *fpo = fopen("result.txt","w");
    for(int i=0;i<no_of_nodes;i++) {
      fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
    }
    fclose(fpo);
    printf("Result stored in result.txt\n");

    // cleanup memory
    free( h_graph_nodes);
    free( h_graph_edges);
    free( h_graph_mask);
    free( h_updating_graph_mask);
    free( h_graph_visited);
    free( h_cost);
    /*
    for (int i = 0; i  < numStreams; ++i) {
      hipFree(&d_graph_nodes[i]);
      hipFree(&d_graph_edges[i]);
      hipFree(&d_graph_mask[i]);
      hipFree(&d_updating_graph_mask[i]);
      hipFree(&d_graph_visited[i]);
      hipFree(&d_cost[i]);
    }
    hipFree(d_graph_nodes);
    hipFree(d_graph_edges);
    hipFree(d_graph_mask);
    hipFree(d_updating_graph_mask);
    hipFree(d_graph_visited);
    hipFree(d_cost);
    */
}
