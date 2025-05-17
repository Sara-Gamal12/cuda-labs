#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> 


#define block_dim 256
#define streams_num 8


// Kernel: Efficient prefix sum 
__global__ void efficient_prefix_sum(int *in, int *out, int *block_finished, int n) {
    __shared__ int sh_in[2*256];
    int i = threadIdx.x;
    int start = 2 * blockIdx.x * blockDim.x;

    // load 1st element
    if (start + i < n)
        sh_in[i] = in[start + i];
    else
        sh_in[i] = 0;

    // load 2nd element
    if (start+ i +blockDim.x  < n)
        sh_in[i+blockDim.x] = in[start + i+ blockDim.x ];
    else
        sh_in[blockDim.x + i] = 0;

    // reduction
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int indx = (i + 1) * 2 * stride - 1;
        if (indx < 2 * blockDim.x) {
            sh_in[indx] += sh_in[indx - stride];
        }
    }

    if (i == 0) {
        sh_in[2 * blockDim.x - 1] = 0;
    }

    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        __syncthreads();
        int indx = (i + 1) * 2 * stride - 1;
        if (indx < 2 * blockDim.x) {
            int temp = sh_in[indx - stride];
            sh_in[indx - stride] = sh_in[indx];
            sh_in[indx] += temp;
        }
    }

    __syncthreads();
    if (start + i < n) {
        out[start + i] = sh_in[i] + in[start + i]; 
        __threadfence();
    }
    if (start+ i +blockDim.x  < n) {
        out[start+ i +blockDim.x ] = sh_in[blockDim.x + i] + in[start + i +blockDim.x];
        __threadfence();
    }

    while (blockIdx.x > 0 && atomicAdd(&block_finished[blockIdx.x-1],0) == 0) {}
    __syncthreads();

    if (start + i < n && blockIdx.x > 0) {
        out[start + i] += out[2 * blockIdx.x * blockDim.x - 1]; 
    }
    if (start +i+ blockDim.x  < n && blockIdx.x > 0) {
        out[start +i+ blockDim.x] += out[2 * blockIdx.x * blockDim.x - 1]; 
    }

    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(&block_finished[blockIdx.x],1);
    }
}

// Kernel: Add previous streams' sums
__global__ void add_stream_sums(int *d_out, int stream_sum, int n, int offset, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size && (i + offset) < n) {
        d_out[i + offset] += stream_sum;
    }
}

// Host function to perform prefix sum with streams
void prefix_sum_streams(int *h_in, int *h_out, int n) {
    int *d_in, *d_out;
    int **d_block_finished; // Per-stream block_finished arrays
    int *h_block_sums;      // Host storage for last prefix sum per stream
    cudaStream_t streams[streams_num];
    cudaEvent_t events[streams_num];
    int chunk_size = (n + streams_num - 1) / streams_num;

    // Allocate device memory
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    h_block_sums = (int *)malloc(streams_num * sizeof(int));
    d_block_finished = (int **)malloc(streams_num * sizeof(int *));

    // Allocate per-stream block_finished arrays
    int *h_block_finished_ptrs[streams_num];
    for (int i = 0; i < streams_num; i++) {
        int size = min(chunk_size, n - i * chunk_size);
        int local_blocks = (size + 2 * block_dim - 1) / (2 * block_dim);
        cudaMalloc(&h_block_finished_ptrs[i], local_blocks * sizeof(int));
        cudaMemset(h_block_finished_ptrs[i], 0, local_blocks * sizeof(int));
    }

    // Create streams and events
    for (int i = 0; i < streams_num; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // Process each stream
    for (int i = 0; i < streams_num; i++) {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);
        if (size > 0) {
            int local_blocks = (size + 2 * block_dim - 1) / (2 * block_dim);

            // Copy input data
            cudaMemcpyAsync(d_in + offset, h_in + offset, size * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

            // Launch efficient prefix sum kernel
            efficient_prefix_sum<<<local_blocks, block_dim, 2 * block_dim * sizeof(int), streams[i]>>>(d_in + offset, d_out + offset, h_block_finished_ptrs[i], size);
            cudaGetLastError();

            // Copy last prefix sum to host
            if (local_blocks > 0) {
                cudaMemcpyAsync(&h_block_sums[i], d_out + offset + size - 1, sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            }

            cudaEventRecord(events[i], streams[i]);
        }
    }

    // Add previous streams' sums
    for (int i = 1; i < streams_num; i++) {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);
        if (size > 0) {
            int local_blocks = (size + 2 * block_dim - 1) / (2 * block_dim);

            // Wait for previous stream
            cudaStreamWaitEvent(streams[i], events[i - 1], 0);

            // Compute cumulative sum of previous streams
            int cumulative_sum = 0;
            for (int j = 0; j < i; j++) {
                cudaStreamSynchronize(streams[j]);
                cumulative_sum += h_block_sums[j];
            }

            // Launch kernel to add cumulative sum
            add_stream_sums<<<local_blocks, block_dim * 2, 0, streams[i]>>>(d_out, cumulative_sum, n, offset, size);
            cudaGetLastError();
        }
    }

    // Copy output to host
    for (int i = 0; i < streams_num; i++) {
        int offset = i * chunk_size;
        int size = min(chunk_size, n - offset);
        if (size > 0) {
            cudaMemcpyAsync(h_out + offset, d_out + offset, size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        }
    }

    // Synchronize and clean up
    cudaDeviceSynchronize();
    for (int i = 0; i < streams_num; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
        cudaFree(h_block_finished_ptrs[i]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_block_sums);
    free(d_block_finished);
}

void check(int *h_in, int *h_out, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += h_in[i];
        if (h_out[i] != sum) {
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