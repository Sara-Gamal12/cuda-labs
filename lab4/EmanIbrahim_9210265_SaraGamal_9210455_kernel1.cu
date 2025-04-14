#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <float.h>
#include<iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <sys/stat.h>
__constant__ float c_mask[256];


void save_images(const char* output_folder, float* output_data,   int width, int height, int channels,  int batch_size,std::vector<std::string> input_paths,int batch_start) {
    // Create output directory if it doesn't exist
    mkdir(output_folder, 0777);


    // Process each image in the current batch
    for (int i = 0; i < batch_size; i++) {

        // Extract filename from input path
        std::string path = input_paths[ i+batch_start];

        size_t last_slash = path.find_last_of("/\\");
        std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);

        // Create output path (preserve extension)
        std::string output_path = std::string(output_folder) + "/conv_" + filename;

        // Allocate memory for output image (convert from float to uint8)
        unsigned char* image_data = (unsigned char*)malloc(width * height*channels );

      float min_pixel = FLT_MAX;
       float max_pixel = -FLT_MAX;

  for (int k=0;k<channels;k++)
       for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

          int output_idx = ((i*channels+k )* height * width ) +        (y * width ) +   (x ) ;
            if (output_data[output_idx] < min_pixel)
                min_pixel = output_data[output_idx];
            if (output_data[output_idx] > max_pixel)
                max_pixel = output_data[output_idx];
        }}
        // Convert and normalize output data
        for(int k=0;k<channels;k++)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              {
                    // Calculate indices (NHWC layout)
                    int output_idx = ((i*channels+k ) * height * width ) +
                                   (y * width ) +
                                   (x ) ;

                    float pixel_val = output_data[output_idx];

                    pixel_val=static_cast<unsigned char>(255.0f *(pixel_val-min_pixel)/(max_pixel-min_pixel));
                   image_data[((y * width + x)*channels+k) ] = pixel_val;
                }
            }
        }

        // Save image (preserve original format)
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        int success = 0;
        if (ext == "png") {
            success = stbi_write_png(output_path.c_str(), width, height, channels, image_data, width * channels);
        }
        else if (ext == "jpg" || ext == "jpeg") {
            success = stbi_write_jpg(output_path.c_str(), width, height, channels, image_data, 90);  // 90% quality
        }
        else {
            printf("Unsupported output format for %s, defaulting to PNG\n", output_path.c_str());
            success = stbi_write_png(output_path.c_str(), width, height, 1, image_data, width * 1);
        }

        if (!success) {
            printf("Failed to save image %s\n", output_path.c_str());
        }

        free(image_data);
    }
}

__global__ void conv3D_basic(const uint8_t *input, int width, int height, int depth,int batch_size, float *output, float *mask, int maskWidth)
 {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_index= threadIdx.z+blockIdx.z * blockDim.z;


    if (col >= width || row >= height||batch_index>=batch_size ) return;


   for (int channel=0;channel<depth;channel++)
{
   float sum = 0.0f;
    for (int i = 0; i < maskWidth; ++i) {
        for (int j =0; j < maskWidth; ++j) {

            int curr_row = row+i-maskWidth/2;
            int curr_col = col+j-maskWidth/2;
            if(curr_col<width&& curr_row<height&&curr_col>=0&&curr_row>=0)
            {

             {
               sum+=c_mask[i*maskWidth+j]*static_cast<float>(input[batch_index*height*width*depth  +  curr_row*width*depth+curr_col*depth+channel]);

}
            }
        }
    }
      int outIdx = (batch_index*depth+channel)*height*width+row*width+col;
               output[outIdx] = sum;}

}


float* read_mask(const char* file_path, int& maskWidth) {
    FILE* file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open mask file %s\n", file_path);
        return nullptr;
    }

    // Read mask dimension (first line)
    if (fscanf(file, "%d", &maskWidth) != 1) {
        fprintf(stderr, "Error: Could not read mask dimension from %s\n", file_path);
        fclose(file);
        return nullptr;
    }

    float* mask = (float*)malloc(maskWidth * maskWidth * sizeof(float));
    if (!mask) {
        fprintf(stderr, "Error: Memory allocation failed for mask\n");
        fclose(file);
        return nullptr;
    }

    // Read mask values (subsequent lines)
    for (int i = 0; i < maskWidth; i++) {
        for (int j = 0; j < maskWidth; j++) {
            if (fscanf(file, "%f", &mask[i * maskWidth + j]) != 1) {
                fprintf(stderr, "Error: Invalid mask data at row %d, column %d\n", i+1, j+1);
                free(mask);
                fclose(file);
                return nullptr;
            }
        }
    }

    fclose(file);
    return mask;
}


uint8_t* load_images(const char* folder_path, int& width, int& height, int& channels, int batch_size,int & num_images, std::vector<std::string>& image_paths) {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(folder_path)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".jpg") != std::string::npos ||
                filename.find(".jpeg") != std::string::npos ||
                filename.find(".png") != std::string::npos) {
                image_paths.push_back(std::string(folder_path) + "/" + filename);
            }
        }
        closedir(dir);
    } else {
        perror("Could not open directory");
        return nullptr;
    }

    if (image_paths.empty()) {
        printf("No images found in %s\n", folder_path);
        return nullptr;
    }

   num_images=image_paths.size();
   uint8_t* h_input;
    // Load images into batch
    for (int i = 0; i < image_paths.size(); i++) {
        int img_width, img_height, img_channels;
        unsigned char* image_data = stbi_load(image_paths[i].c_str(), &img_width, &img_height, &img_channels, 0);

        if(i==0)
        {
           height=img_height;
        width=img_width;
        channels=img_channels;
           size_t input_size = image_paths.size() * height * width * channels * sizeof(uint8_t);
            h_input = (uint8_t*)malloc(input_size);

        }
        if (!image_data) {
            printf("Failed to load image: %s\n", image_paths[i].c_str());
            continue;
        }


        // Copy image data to batch (NHWC layout)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    int src_idx = (y * width + x) * channels + c;
                    int dst_idx = (i * height * width * channels) +
                                 (y * width * channels) +
                                 (x * channels) + c;
                    h_input[dst_idx] = image_data[src_idx];
                }
            }
        }

        stbi_image_free(image_data);
    }

    return h_input;
}

int main(int argc, char** argv)
{

   if (argc != 5) {
        printf("arguments are incorrect");
        return 1;
    }
    const char* input_folder = argv[1];
    const char* output_folder = argv[2];
    int batch_size = atoi(argv[3]);
    const char* mask_file = argv[4];



    int maskWidth;
    float*h_mask=read_mask(mask_file,maskWidth);
    if(!h_mask)
    {
      return 1;

    }


    int height,width,depth;
    uint8_t* h_input;
    int num_images;
    std::vector<std::string> input_paths;
    h_input=load_images(input_folder,width,height,depth,batch_size, num_images,input_paths);






    uint8_t* d_input;
    float* d_output;
    float* d_mask;

    //Allocate


    cudaMalloc(&d_mask, maskWidth * maskWidth * sizeof(float));
    cudaMemcpy(d_mask, h_mask, maskWidth * maskWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, h_mask, maskWidth * maskWidth * sizeof(float));

    for(int batch_start=0;batch_start<num_images;batch_start+=batch_size)
    {

        int current_batch_size = (batch_start + batch_size > num_images) ? num_images - batch_start : batch_size;
        size_t input_size = current_batch_size * height * width * sizeof(uint8_t)*depth;
        size_t output_size = current_batch_size * height * width * sizeof(float)*depth;;
         float* h_output = (float*)malloc(output_size);

            //copy to gpu
            cudaMalloc(&d_input, input_size);
            cudaMalloc(&d_output, output_size);


            cudaMemcpy(d_input,  &h_input[batch_start * width * height * depth], input_size, cudaMemcpyHostToDevice);

   dim3 block_size(16, 16, 1);
   dim3 grid_size(
       (width + block_size.x - 1) / block_size.x,
       (height + block_size.y - 1) / block_size.y,
       current_batch_size

   );


            conv3D_basic<<<grid_size, block_size>>>(d_input, width, height, depth, current_batch_size,
                d_output, d_mask, maskWidth);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            }

            cudaDeviceSynchronize();  // Required to flush printf output


            cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);


            save_images(output_folder,h_output,width,height,depth,current_batch_size,input_paths,batch_start);
            cudaFree(d_input);
            cudaFree(d_output);
            free(h_output);


    }



  free(h_mask);
    free(h_input);
    cudaFree(d_mask);

    return 0;


}
