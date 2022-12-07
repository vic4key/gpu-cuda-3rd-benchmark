#pragma once

#include <vector>

#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // M_PI

typedef unsigned char byte_t;
typedef std::vector<byte_t> bytes_t;

__declspec(align(1)) struct rgb_t
{
  byte_t r, g, b;
};

__host__ __device__ static byte_t truncate(double v)
{
  if (v < 0.)   return 0x00;
  if (v > 255.) return 0xff;
  return byte_t(v);
}

__host__ __device__ static void shift_hue_pixel(rgb_t& rgb_pixel, const double hue_val, rgb_t& result)
{
  const double u = cos(hue_val * M_PI / 180);
  const double w = sin(hue_val * M_PI / 180);

  double m[3][3] = // rotation matrix
  {
    {u + (1.0f - u) / 3.0f, 1.0f / 3.0f * (1.0f - u) - sqrtf(1.0f / 3.0f) * w, 1.0f / 3.0f * (1.0f - u) + sqrtf(1.0f / 3.0f) * w},
    {1.0f / 3.0f * (1.0f - u) + sqrtf(1.0f / 3.0f) * w, u + 1.0f / 3.0f * (1.0f - u), 1.0f / 3.0f * (1.0f - u) - sqrtf(1.0f / 3.0f) * w},
    {1.0f / 3.0f * (1.0f - u) - sqrtf(1.0f / 3.0f) * w, 1.0f / 3.0f * (1.0f - u) + sqrtf(1.0f / 3.0f) * w, u + 1.0f / 3.0f * (1.0f - u)},
  };

  result.r = truncate(rgb_pixel.r * m[0][0] + rgb_pixel.g * m[0][1] + rgb_pixel.b * m[0][2]);
  result.g = truncate(rgb_pixel.r * m[1][0] + rgb_pixel.g * m[1][1] + rgb_pixel.b * m[1][2]);
  result.b = truncate(rgb_pixel.r * m[2][0] + rgb_pixel.g * m[2][1] + rgb_pixel.b * m[2][2]);
}

__host__ __device__ static void convert_to_gray_pixel(rgb_t& rgb_pixel, byte_t& result)
{
  result = truncate((0.3 * rgb_pixel.r) + (0.59 * rgb_pixel.g) + (0.11 * rgb_pixel.b));
}

__host__ __device__ static byte_t transform_pixel(rgb_t& rgb_pixel) // transform a rgb pixel with heavy calculation (hue + gray)
{
  byte_t result = 0;
  shift_hue_pixel(rgb_pixel, 0.1, rgb_pixel);
  convert_to_gray_pixel(rgb_pixel, result);
  return result;
}

typedef void (*pfn_transform_image)(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);

void transform_image_cpu(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
void transform_image_gpu_native(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
void transform_image_gpu_thrust(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
void transform_image_gpu_wrapper(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels);
