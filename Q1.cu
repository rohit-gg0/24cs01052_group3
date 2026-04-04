#include<cuda_runtime.h>
#include<iostream>

__global__ void static_memory_kernel(int* input_data, int* output_data, int* block_shared_space) 
{
    const int shared_space = 32; //gradually increasing until failure

    *block_shared_space = shared_space;
    __shared__ int shared_data[shared_space/sizeof(int)];

    shared_data[threadIdx.x] = input_data[threadIdx.x];

    __syncthreads();

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}

int main()
{
    int threads = 256;
    int blocks = 2;

    int* device_input=NULL;
    int* device_output=NULL;
    int* block_shared_space;

    cudaMalloc(&device_input,sizeof(int)*threads);
    cudaMalloc(&device_output,sizeof(int)*blocks);
    cudaMalloc(&block_shared_space,sizeof(int));

    static_memory_kernel<<<blocks,threads>>>(device_input,device_output,block_shared_space);
    cudaDeviceSynchronize();

    int shared_space;
    cudaMemcpy(&shared_space,block_shared_space,sizeof(int),cudaMemcpyDefault);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(block_shared_space);

    std::cout<<"Block Shared Space: "<<shared_space<<"\n";

    cudaDeviceProp prop;
    cudaError_t k = cudaGetDeviceProperties(&prop,0);
    std::cout<<"sharedMemPerBlock: "<<prop.sharedMemPerBlock<<" bytes\n";
    std::cout<<"sharedMemPerMultiprocessor: "<<prop.sharedMemPerMultiprocessor<<" bytes\n";
    return 0;
}