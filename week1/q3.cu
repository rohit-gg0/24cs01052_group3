#include<iostream>
#include<cuda_runtime.h>

#define threads_per_block 32
#define blocks 1024

__global__ void strided(int* a,int stride)
{
    volatile __shared__ int shmem[1024]; // volatile so the compiler forces to read every iteration

    shmem[threadIdx.x*stride] = threadIdx.x;
    __syncthreads();

    int sum = 0;

    for(int i=0;i<1000;i++)
    {
        sum += shmem[threadIdx.x*stride];
    }

    a[blockDim.x*blockIdx.x + threadIdx.x] = sum;
}

int main()
{
    int* dev_a=nullptr;

    int strides[] = {1,2,4,8,16,32};
    cudaMalloc(&dev_a,blocks*threads_per_block*sizeof(int));

    for(int i=0;i<6;i++)
    {
        int stride = strides[i];

        strided<<<blocks,threads_per_block>>>(dev_a,stride);
        cudaDeviceSynchronize();

        float avg_elapsed = 0;

        for(int i=0;i<1000;i++)
        {
            cudaEvent_t start;
            cudaEvent_t stop;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            strided<<<blocks,threads_per_block>>>(dev_a,stride);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_time;
            cudaEventElapsedTime(&elapsed_time, start, stop);

            avg_elapsed +=elapsed_time;
        }

        std::cout<<"stride "<<stride<<": "<<avg_elapsed/1000<<" ms\n";
    }

    cudaFree(dev_a);
}   