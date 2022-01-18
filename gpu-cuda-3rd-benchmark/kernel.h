#pragma once

#include <vector>

#include <cuda_runtime.h>

typedef unsigned char byte_t;
typedef std::vector<byte_t> bytes_t;

__host__ __device__ static double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

__declspec(align(1)) struct rgb_t
{
  byte_t r, g, b;
};

#define transform_pixel(rgb_pixel)\
  byte_t(sigmoid((0.3 * rgb_pixel.r) + (0.59 * rgb_pixel.g) + (0.11 * rgb_pixel.b)) * 0xff)

typedef void (*pfn_transform_image)(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);

void transform_image_cpu(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
void transform_image_gpu_native(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
void transform_image_gpu_thrust(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
void transform_image_gpu_wrapper(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
