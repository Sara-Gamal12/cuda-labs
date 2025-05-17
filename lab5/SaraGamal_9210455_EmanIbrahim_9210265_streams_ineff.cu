
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> 


#define block_dim 256
#define streams_num 8


// Kernel: Compute per-block prefix sum
__global__ void block_prefix_sum(int *d_in, int *d_out, int *block_sums, int *block_finished, int n, int blocks)
{
    __shared__ int sh_in[block_dim];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int local_i = threadIdx.x;

    // Load input to shared memory
    if (i < n)
    {
        sh_in[local_i] = d_in[i];
    }
    else
    {
        sh_in[local_i] = 0;
    }
    __syncthreads();

    // Perform prefix sum within block
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int temp = 0;
        if (local_i >= stride)
        {
            temp = sh_in[local_i - stride];
        }
        __syncthreads();
        sh_in[local_i] += temp;
        __syncthreads();
    }

    if (i < n)
    {
        d_out[i] = sh_in[local_i];
        __threadfence();
    }

    // Wait for previous block to complete
    while (blockIdx.x > 0 && atomicAdd(&block_finished[blockIdx.x - 1], 0) == 0)
    {
    }
    __syncthreads();

    // Add previous block's sum
    if (i < n && blockIdx.x > 0)
    {
        d_out[i] += d_out[(blockIdx.x - 1) * blockDim.x + blockDim.x - 1];
    }

    // Signal block completion
    if (threadIdx.x == 0)
    {
        __threadfence();
        atomicAdd(&block_finished[blockIdx.x], 1);
    }

    // Store the block sum for the last thread in the last block
    if (local_i == blockDim.x - 1 && blockIdx.x == blocks - 1)
    {
        block_sums[0] = d_out[i];
    }
}

// Kernel: Add previous streams' sums
__global__ void add_stream_sums(int *d_out, int stream_sum, int n, int offset, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size && (i + offset) < n)
    {
        d_out[i + offset] += stream_sum;
    }
}

// Host function to perform prefix sum with streams
void prefix_sum_streams(int *h_in, int *h_out, int n)
{
    int *d_in, *d_out;
    int **d_block_sums; // Per-stream block sums
    int *h_block_sums;  // Host storage for block sums
    int *block_finished;
    cudaStream_t streams[streams_num];
    cudaEvent_t events[streams_num];
    int chunk_size = (n + streams_num - 1) / streams_num;

    // Allocate device memory
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_block_sums, streams_num * sizeof(int *));
    h_block_sums = (int *)malloc(streams_num * sizeof(int)); // Host memory
    int total_blocks = (n + block_dim - 1) / block_dim;
    cudaMalloc(&block_finished, total_blocks * sizeof(int));
    cudaMemset(block_finished, 0, total_blocks * sizeof(int));

    // Allocate host memory for block sums pointers
    int *h_block_sums_ptrs[streams_num];
    for (int i = 0; i < streams_num; i++)
    {
        int size = min(chunk_size, n - i * chunk_size);
        int local_blocks = (size + block_dim - 1) / block_dim;
        cudaMalloc(&h_block_sums_ptrs[i], local_blocks * sizeof(int));
    }
    cudaMemcpy(d_block_sums, h_block_sums_ptrs, streams_num * sizeof(int *), cudaMemcpyHostToDevice);

    // Create streams and events
    for (int i = 0; i < streams_num; i++)
    {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // Process each stream
    int block_offset = 0;
    for (int i = 0; i < streams_num; i++)
    {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);
        if (size > 0)
        {
            int local_blocks = (size + block_dim - 1) / block_dim;

            // Copy input data
            cudaMemcpyAsync(d_in + offset, h_in + offset, size * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

            // Launch block prefix sum kernel
            block_prefix_sum<<<local_blocks, block_dim, 0, streams[i]>>>(d_in + offset, d_out + offset, h_block_sums_ptrs[i], block_finished + block_offset, size, local_blocks);

            // Copy last block sum to host
            if (local_blocks > 0)
            {
                cudaMemcpyAsync(&h_block_sums[i], d_out + offset + size - 1, sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            }

            cudaEventRecord(events[i], streams[i]);
            block_offset += local_blocks;
        }
    }

    // Add previous streams' sums
    for (int i = 1; i < streams_num; i++)
    {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);
        if (size > 0)
        {
            int local_blocks = (size + block_dim - 1) / block_dim;

            // Wait for previous stream to complete
            cudaStreamWaitEvent(streams[i], events[i - 1], 0);

            // Compute cumulative sum of previous streams
            int cumulative_sum = 0;
            for (int j = 0; j < i; j++)
            {
                cudaStreamSynchronize(streams[j]);
                cumulative_sum += h_block_sums[j];
            }

            // Launch kernel to add cumulative sum
            add_stream_sums<<<local_blocks, block_dim, 0, streams[i]>>>(d_out, cumulative_sum, n, offset, size);
        }
    }

    // Copy output to host
    for (int i = 0; i < streams_num; i++)
    {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);
        if (size > 0)
        {
            cudaMemcpyAsync(h_out + offset, d_out + offset, size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        }
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < streams_num; i++)
    {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
        cudaFree(h_block_sums_ptrs[i]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    free(h_block_sums);
    cudaFree(block_finished);
}

void check(int *h_in, int *h_out, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += h_in[i];
        if (h_out[i] != sum)
        {
            printf("failed\n");
            return;
        }
    }
    printf("passed\n");
}

int main() {
    int n;
    int *h_in, *h_out;

    FILE *file = fopen("input2.txt", "r");
    if (!file) {
        perror("Can't open the input file");
        return -1;
    }
    fscanf(file, "%d", &n);
    h_in = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        fscanf(file, "%d", &h_in[i]);
    }
    fclose(file);

    h_out = (int *)malloc(n * sizeof(int));

    // Timing starts here
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    prefix_sum_streams(h_in, h_out, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total execution time including all streams: %.6f ms\n", milliseconds);

    // Clean up the timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    check(h_in, h_out, n);

    FILE *out_file = fopen("output.txt", "w");
    if (!out_file) {
        perror("Can't open the output file");
        return -1;
    }
    for (int i = 0; i < n; i++) {
        fprintf(out_file, "%d ", h_out[i]);
    }
    fclose(out_file);

    free(h_in);
    free(h_out);
    return 0;
}