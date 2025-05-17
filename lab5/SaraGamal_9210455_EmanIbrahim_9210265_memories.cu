#include <stdio.h>
#include <stdlib.h>

__global__ void inefficient_prefix_sum(int *in, int *out, int *block_finished, int n)
{
    __shared__ int sh_in[256];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        sh_in[threadIdx.x] = in[i];
    }
    else
    {
        sh_in[threadIdx.x] = 0;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        int temp = 0;
        if (stride <= threadIdx.x)
        {
            temp = sh_in[threadIdx.x - stride];
        }
        __syncthreads();
        sh_in[threadIdx.x] += temp;
    }
    if (i < n)
    {
        out[i] = sh_in[threadIdx.x];
        __threadfence();
    }


    while (blockIdx.x > 0 &&  atomicAdd(&block_finished[blockIdx.x-1],0) == 0){}
    __syncthreads();

    if ( blockIdx.x > 0 && i < n )
    {
        out[i] += out[(blockIdx.x - 1) * blockDim.x + blockDim.x - 1];
    }

    if (threadIdx.x == 0)
    {
        __threadfence();
        atomicAdd(&block_finished[blockIdx.x],1);
    }
}


__global__ void efficient_prefix_sum(int *in, int *out,int *block_finished, int n) {
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
        out[start+ i +blockDim.x ] = sh_in[blockDim.x + i] + in[start+ i +blockDim.x];
         __threadfence();
    }

    while (blockIdx.x > 0 && atomicAdd (&block_finished[blockIdx.x-1],0) == 0 ){}
    __syncthreads();
  

    if (start + i < n && blockIdx.x > 0) {
        out[start + i] += out[2 * blockIdx.x * blockDim.x - 1]; 
    }
    if (start +i+ blockDim.x  < n && blockIdx.x > 0) {
        out[start +i+ blockDim.x] += out[2 * blockIdx.x * blockDim.x - 1]; 
    }

    if (threadIdx.x == 0)
    {
        __threadfence();
        atomicAdd(&block_finished[blockIdx.x],1);
    }
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


int main() //pageable
{
    int n;
    int *h_in, *h_out;
    int *d_in, *d_out;
    int *d_block_finished;

    FILE *file = fopen("input2.txt", "r");
    if (!file)
    {
        perror("Can't open the input file");
        return -1;
    }
    fscanf(file, "%d", &n);
    h_in = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
      fscanf(file, "%d", &h_in[i]);

    fclose(file);

    h_out = (int *)malloc(n * sizeof(int));
    
    
    cudaMalloc((void **)&d_in, n * sizeof(int));
    cudaMalloc((void **)&d_out, n * sizeof(int));
    
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    

    int block_dim=256;
    int grid_dim = (n + block_dim - 1) / block_dim; //for ineff
    //int grid_dim = (n + 2*block_dim - 1) / (2*block_dim); //for eff


    cudaMalloc((void **)&d_block_finished, grid_dim * sizeof(int));
    cudaMemset(d_block_finished, 0, grid_dim * sizeof(int));

    inefficient_prefix_sum<<<grid_dim, block_dim>>>( d_in,d_out, d_block_finished, n);
    //efficient_prefix_sum<<<grid_dim, block_dim>>>(d_in,d_out, d_block_finished, n);

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    

    check(h_in, h_out, n);

    FILE*out_file = fopen("output.txt", "w");
    if (!out_file)
    {
        perror("Can't open the output file");
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        fprintf(out_file, "%d ", h_out[i]);
    }
    fclose(out_file);


  
      free(h_in);
      free(h_out);
      cudaFree(d_in);
      cudaFree(d_out);
   
    
    return 0;
}

// int main() //pinned
// {
//     int n;
//     int *h_in, *h_out;
//     int *d_in, *d_out;
//     int *d_block_finished;

//     FILE *file = fopen("input2.txt", "r");
//     if (!file)
//     {
//         perror("Can't open the input file");
//         return -1;
//     }
//     fscanf(file, "%d", &n);
//     cudaMallocManaged((int **)&h_in, n * sizeof(int));;

//     for (int i = 0; i < n; i++)
//       fscanf(file, "%d", &h_in[i]);

//     fclose(file);

//     h_out = (int *)malloc(n * sizeof(int));
    
    
//     cudaMalloc((void **)&d_in, n * sizeof(int));
//     cudaMalloc((void **)&d_out, n * sizeof(int));
    
//     cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    

//     int block_dim=256;
//     //int grid_dim = (n + block_dim - 1) / block_dim; //for ineff
//     int grid_dim = (n + 2*block_dim - 1) / (2*block_dim); //for eff


//     cudaMalloc((void **)&d_block_finished, grid_dim * sizeof(int));
//     cudaMemset(d_block_finished, 0, grid_dim * sizeof(int));

//     //inefficient_prefix_sum<<<grid_dim, block_dim>>>( d_in,d_out, d_block_finished, n);
//     efficient_prefix_sum<<<grid_dim, block_dim>>>(d_in,d_out, d_block_finished, n);

//     cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    

//     check(h_in, h_out, n);

//     FILE*out_file = fopen("output.txt", "w");
//     if (!out_file)
//     {
//         perror("Can't open the output file");
//         return -1;
//     }

//     for (int i = 0; i < n; i++)
//     {
//         fprintf(out_file, "%d ", h_out[i]);
//     }
//     fclose(out_file);


  
//       cudaFree(h_in);
//       free(h_out);
//       cudaFree(d_in);
//       cudaFree(d_out);
   
    
//     return 0;
// }

// int main() //memory mapped
// {
//     int n;
//     int *h_in, *h_out;
//     int *d_in, *d_out;
//     int *d_block_finished;

//     FILE *file = fopen("input2.txt", "r");
//     if (!file)
//     {
//         perror("Can't open the input file");
//         return -1;
//     }
//     fscanf(file, "%d", &n);
//     cudaHostAlloc((int **)&h_in, n * sizeof(int),cudaHostAllocMapped);
//     cudaHostGetDevicePointer((int **)&d_in, (int*)h_in, 0);

//     cudaHostAlloc((int **)&h_out, n * sizeof(int),cudaHostAllocMapped);
//     cudaHostGetDevicePointer((int **)&d_out, (int*)h_out, 0);

//     for (int i = 0; i < n; i++)
//       fscanf(file, "%d", &h_in[i]);

//     fclose(file);
    

//     int block_dim=256;
//     //int grid_dim = (n + block_dim - 1) / block_dim; //for ineff
//     int grid_dim = (n + 2*block_dim - 1) / (2*block_dim); //for eff


//     cudaMalloc((void **)&d_block_finished, grid_dim * sizeof(int));
//     cudaMemset(d_block_finished, 0, grid_dim * sizeof(int));

//     //inefficient_prefix_sum<<<grid_dim, block_dim>>>( d_in,d_out, d_block_finished, n);
//     efficient_prefix_sum<<<grid_dim, block_dim>>>(d_in,d_out, d_block_finished, n);
//     cudaDeviceSynchronize();

//     check(h_in, h_out, n);

//     FILE*out_file = fopen("output.txt", "w");
//     if (!out_file)
//     {
//         perror("Can't open the output file");
//         return -1;
//     }

//     for (int i = 0; i < n; i++)
//     {
//         fprintf(out_file, "%d ", h_out[i]);
//     }
//     fclose(out_file);


  
//       cudaFreeHost(h_in);
//       cudaFreeHost(h_out);
     
    
//     return 0;
// }




// int main() //unified
// {
//     int n;
//     int *h_in, *h_out;
//     int *d_block_finished;

//     FILE *file = fopen("input2.txt", "r");
//     if (!file)
//     {
//         perror("Can't open the input file");
//         return -1;
//     }
//     fscanf(file, "%d", &n);

//     cudaMallocManaged((void **)&h_in, n * sizeof(int));
//     cudaMallocManaged((void **)&h_out, n * sizeof(int));


//     for (int i = 0; i < n; i++)
//       fscanf(file, "%d", &h_in[i]);

//     fclose(file);
    

//     int block_dim=256;
//     //int grid_dim = (n + block_dim - 1) / block_dim; //for ineff
//     int grid_dim = (n + 2*block_dim - 1) / (2*block_dim); //for eff


//     cudaMalloc((void **)&d_block_finished, grid_dim * sizeof(int));
//     cudaMemset(d_block_finished, 0, grid_dim * sizeof(int));

//     //inefficient_prefix_sum<<<grid_dim, block_dim, block_dim>>>( h_in,h_out, d_block_finished, n);
//     efficient_prefix_sum<<<grid_dim, block_dim, 2*block_dim>>>(h_in,h_out, d_block_finished, n);
//     cudaDeviceSynchronize();

//     check(h_in, h_out, n);


    
//     FILE*out_file = fopen("output.txt", "w");
//     if (!out_file)
//     {
//         perror("Can't open the output file");
//         return -1;
//     }

//     for (int i = 0; i < n; i++)
//     {
//         fprintf(out_file, "%d ", h_out[i]);
//     }
//     fclose(out_file);

//       cudaFree(h_in);
//       cudaFree(h_out);
   
    
//     return 0;
// }
