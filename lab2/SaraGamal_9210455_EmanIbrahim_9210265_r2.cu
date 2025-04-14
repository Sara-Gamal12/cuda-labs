#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void From3DTo2D(float *mat3D, float *mat2D, int rows, int cols ,int depth)

{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<rows && col<cols){
      float sum=0;
      for(int i=0;i<depth;i++){
        int index3D=rows*cols*i+cols*row+col;
        sum+=mat3D[index3D];
    }
    int index2D=cols*row+col;
    mat2D[index2D]=sum;
    }
    
}
__global__ void From2DTo1D(float *mat2D, float *mat1D, int rows, int cols)
                
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row<rows){
      float sum=0;
      for(int i=0;i<cols;i++){
      int index2D=row*cols+i;
      sum+=mat2D[index2D];
    }
      mat1D[row]=sum;
    }

}
__host__ float VectorAdd(float *vector,int rows)

{
    float sum =0;
    for(int i=0;i<rows;i++){
      sum+=vector[i];

    }
    return sum;
}



int main(int argc, char *argv[]){

  if (argc < 3) { 
        printf("Two file paths are needed");
        return 1; 
    }
    char *input = argv[1];
    char *output = argv[2];

    float *mat3D,*vector;
    float sum;
    float *d_mat3D, *d_mat2D, *d_vector;

    FILE *file = fopen(input, "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    int cols, rows, depth;
    fscanf(file, "%d %d %d", &cols, &rows, &depth);

    // Allocate host memory
    mat3D   = (float*)malloc(sizeof(float) * rows *cols *depth);
    vector = (float*)malloc(sizeof(float) * rows);

    for (int i = 0; i < rows *cols *depth; i++) {
        fscanf(file, "%f", &mat3D[i]); 
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_mat3D, rows *cols * depth * sizeof(float));
    cudaMalloc((void**)&d_mat2D,  rows *cols * sizeof(float));
    

    // Transfer data from host to device memory
    cudaMemcpy(d_mat3D, mat3D,  rows *cols * depth * sizeof(float), cudaMemcpyHostToDevice);

    // Executing kernel 
    dim3 threadsPerBlock1(16, 16);
    dim3 numBlocks1((cols + threadsPerBlock1.x - 1) / threadsPerBlock1.x,
                   (rows + threadsPerBlock1.y - 1) / threadsPerBlock1.y);
    From3DTo2D<<<numBlocks1, threadsPerBlock1>>>(d_mat3D,d_mat2D,rows,cols,depth);
   
    printf("Kernal 1 Started...\n");
    cudaDeviceSynchronize();

    // Allocate device memory
    cudaMalloc((void**)&d_vector,  rows * sizeof(float));

    // Executing kernel 
    int threadsPerBlock2=256;
    int numBlocks2=(rows + threadsPerBlock2 - 1) / threadsPerBlock2;
    From2DTo1D<<<numBlocks2, threadsPerBlock2>>>(d_mat2D, d_vector, rows, cols);

    printf("Kernal 2 Started...\n");
    cudaDeviceSynchronize();
    
    // Transfer data back to host memory
    cudaMemcpy(vector, d_vector, rows *sizeof(float), cudaMemcpyDeviceToHost);
      
    sum=VectorAdd(vector,rows);
    printf("Kernal 3 Started...\n");

    printf("Sum = %.3f\n", sum);
    
    FILE*outfile = fopen(output, "w");  // Open file in write mode
    if (outfile == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(outfile, "%.3f", sum);

    // Deallocate device memory
    cudaFree(d_mat3D);
    cudaFree(d_mat2D);
    cudaFree(d_vector);

    // Deallocate host memory
    free(mat3D);
    free(vector);

    
    }
