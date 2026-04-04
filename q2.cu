#include<cuda_runtime.h>
#include<iostream>

#define Blocks 256
#define threads_per_block 8192

__global__ void static_shared_memory_kernel(int* input,int* output)
{
    volatile __shared__ int shmem[threads_per_block];

    int wi = threadIdx.x + blockIdx.x*blockDim.x;

    shmem[threadIdx.x] = input[wi];

    __syncthreads();

    if(threadIdx.x==0) 
    {
        int sum=0;
        for(int i=0;i<threads_per_block;i++)
        {
            sum += shmem[i];
        }
        output[blockIdx.x] = sum;
    }
}

__global__ void dynamic_shared_memory_kernel(int* input,int* output)
{
    volatile extern __shared__ int shmem[];

    int wi = threadIdx.x + blockIdx.x*blockDim.x;

    shmem[threadIdx.x] = input[wi];

    __syncthreads();

    if(threadIdx.x==0)
    {
        int sum=0;
        for(int i=0;i<threads_per_block;i++)
        {
            sum += shmem[i];
        }
        output[blockIdx.x] = sum;
    }
}

void static_shmem(int* input,int* output,int m)
{
    int* d_input=NULL;
    int* d_output=NULL;

    cudaMalloc(&d_input,sizeof(int)*m);
    cudaMalloc(&d_output,sizeof(int)*Blocks);
    cudaMemcpy(d_input,input,sizeof(int)*m,cudaMemcpyDefault);
    cudaMemcpy(d_output,output,sizeof(int)*Blocks,cudaMemcpyDefault);


    static_shared_memory_kernel<<<Blocks,threads_per_block>>>(d_input,d_output);
    cudaDeviceSynchronize();

    float avg_time=0;

    for(int i=0;i<1000;i++)
    {
        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        static_shared_memory_kernel<<<Blocks,threads_per_block>>>(d_input,d_output);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        avg_time+=elapsed_time;
    }

    std::cout<<"static shared memory krenel elapsed time: "<<avg_time/1000<<"ms\n";
    cudaMemcpy(output,d_output,sizeof(int)*Blocks,cudaMemcpyDefault);

    cudaFree(d_input);
    cudaFree(d_output);
}

void dynamic_shmem(int* input,int* output,int m)
{
    int* d_input=NULL;
    int* d_output=NULL;

    cudaMalloc(&d_input,sizeof(int)*m);
    cudaMalloc(&d_output,sizeof(int)*Blocks);
    cudaMemcpy(d_input,input,sizeof(int)*m,cudaMemcpyDefault);
    cudaMemcpy(d_output,output,sizeof(int)*Blocks,cudaMemcpyDefault);

    int dynamic_shared_mem = threads_per_block*sizeof(int);
    float avg_time=0;

    dynamic_shared_memory_kernel<<<Blocks,threads_per_block,dynamic_shared_mem>>>(d_input,d_output);
    cudaDeviceSynchronize();

    for(int i=0;i<1000;i++)
    {
        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        dynamic_shared_memory_kernel<<<Blocks,threads_per_block,dynamic_shared_mem>>>(d_input,d_output);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        avg_time+=elapsed_time;
    }

    std::cout<<"dynamic shared memory kernel elapsed time: "<<avg_time/1000<<"ms\n";

    cudaMemcpy(output,d_output,sizeof(int)*Blocks,cudaMemcpyDefault);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    int m = Blocks*threads_per_block;
    int* input=NULL;
    int* output=NULL;

    std::cout<<"shared memory size: "<<threads_per_block*sizeof(int)<<" bytes\n";

    cudaMallocHost(&input,sizeof(int)*m);
    cudaMallocHost(&output,sizeof(int)*Blocks);

    for(int i=0;i<m;i++) input[i]=i;
    for(int i=0;i<Blocks;i++) output[i]=0;

    static_shmem(input,output,m);
    dynamic_shmem(input,output,m);

    // for(int i=0;i<Blocks;i++) std::cout<<output[i]<<' ';
    // std::cout<<'\n';

    cudaFreeHost(input);
    cudaFreeHost(output);
}