#include <stdio.h>
#include <stdlib.h>
#include <math.h>
__global__ void MatAdd(float *a, float *b,float *out,int rows,int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<rows){
      for(int i=0;i<cols;i++){
        int index=row*cols+i;
        out[index]=a[index]+b[index];
      }
    }
}

int main(int argc, char *argv[]){
  if (argc < 3) { 
        printf("Two file paths are needed");
        return 1; 
    }
    char *input = argv[1];
    char *output = argv[2];

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    FILE *file = fopen(input, "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    int testcases,cols, rows;
    fscanf(file, "%d", &testcases);

    for(int t=0;t<testcases;t++){

    fscanf(file, "%d %d", &rows, &cols);

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * rows *cols);
    b   = (float*)malloc(sizeof(float) * rows *cols);
    out = (float*)malloc(sizeof(float) * rows *cols);

    for (int i = 0; i < rows *cols; i++) {
        fscanf(file, "%f", &a[i]); 
    }
    for (int i = 0; i < rows *cols; i++) {
        fscanf(file, "%f", &b[i]); 
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, rows *cols * sizeof(float));
    cudaMalloc((void**)&d_b,  rows *cols * sizeof(float));
    cudaMalloc((void**)&d_out,  rows *cols * sizeof(float));

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a,  rows *cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b,  rows *cols * sizeof(float), cudaMemcpyHostToDevice);

    // Executing kernel 
    int threadsPerBlock=256;
    int numBlocks=(rows + threadsPerBlock - 1) / threadsPerBlock;
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_out,rows,cols);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, rows *cols * sizeof(float), cudaMemcpyDeviceToHost);

    FILE*outfile = fopen(output, "a");
    if (outfile == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(outfile, "%.3f ", out[i * cols + j]); 
        }
        fprintf(outfile, "\n"); 
    }
    fprintf(outfile, "\n"); 
    fclose(outfile);
    
    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

    }
    }
