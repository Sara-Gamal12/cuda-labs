import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time
import sys
from torch.autograd.profiler import profile, record_function


def load_mask(path):
    with open(path, "r") as f:
        lines = f.read().strip().split('\n')
        n = int(lines[0])
        mask_vals = [list(map(float, line.strip().split())) for line in lines[1:]]
        mask = torch.tensor(mask_vals, dtype=torch.float32)
    return mask.view(1, 1, 1, n, n), n  # Shape: [1, 1, 1, k, k]

def load_input_images(input_folder):
    input_files = sorted(os.listdir(input_folder))
    input_images = []
    for input_file in input_files:
      if input_file.endswith(".png") or input_file.endswith(".jpg"):
        input_path = os.path.join(input_folder, input_file)
        input_images.append(Image.open(input_path).convert("RGB"))
    return input_images

def pytorch_convolution(input_images, kernel, kernel_size, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    transform = transforms.ToTensor()
    outputs = []
    timings = {'total_execution_time_ms': 0, 'batch_times_ms': []}

    kernel = kernel.to(device)

    for i in range(0, len(input_images), batch_size):
        batch_images = input_images[i:i+batch_size]
        batch_tensors = [transform(img).unsqueeze(0) for img in batch_images]
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)  # [B, 3, H, W]

        B, C, H, W = batch_tensor.shape

        # Reshape for 3D: Treat RGB as 'depth'
        batch_tensor = batch_tensor.view(B, C, 1, H, W)  # [B, C, D=1, H, W]
        batch_tensor = batch_tensor.permute(0, 2, 1, 3, 4)  # [B, 1, C, H, W]

        # Pad (H, W) only
        padding = kernel_size // 2
        batch_tensor_padded = F.pad(batch_tensor, (padding, padding, padding, padding), mode='constant', value=0)

        # Convolve

        batch_start_time = time.time()

        if i == 0:
            with profile() as prof:
                  with record_function(f"conv3d_batch_{i}"):
                      output = F.conv3d(batch_tensor_padded, kernel)

            print(prof.key_averages().table(sort_by="cpu_time_total"))
        else:
            with record_function(f"conv3d_batch_{i}"):
                output = F.conv3d(batch_tensor_padded, kernel)

        batch_end_time = time.time()
        timings['batch_times_ms'].append((batch_end_time - batch_start_time) * 1000)



        # Reshape back
        output = output.permute(0, 2, 3, 4, 1).squeeze(-1)  # [B, C, H, W]
        output = output.cpu().numpy()

        for img_array in output:
            outputs.append(img_array)


    timings['total_execution_time_ms'] = sum(timings['batch_times_ms'])
    return outputs, timings

def save_output(output_images, output_path):
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(output_images):
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize to [0,1]
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]
        Image.fromarray(img).save(os.path.join(output_path, f"pytorch_output_{i}.jpg"))

def print_timings(timings):
    print("Total execution time of F.conv3d (ms):", timings['total_execution_time_ms'])
    for i, batch_time in enumerate(timings['batch_times_ms']):
        print(f"\tBatch {i+1} execution time (ms): {batch_time}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python B_1_17_2_14.py <input_folder> <output_folder> <batch_size> <mask_file>")
        return

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    batch_size = int(sys.argv[3])
    mask_file = sys.argv[4]

    kernel, kernel_size = load_mask(mask_file)
    input_images = load_input_images(input_folder)

    output_images, timings = pytorch_convolution(input_images, kernel, kernel_size, batch_size)
    save_output(output_images, output_folder)
    print_timings(timings)

if __name__ == "__main__":
    main()
