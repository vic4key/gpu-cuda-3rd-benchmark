// kernel.cu
//

#include "kernel.h"

#include <iostream>
using namespace std;

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda/runtime_api.hpp>

#include <vu_cuda>

__global__ void cuda_native_kernel_1d(int width, int height, rgb_t* d_src_pixels, byte_t* d_dst_pixels)
{
  int num_pixels = width * height;
  int i = vu::device::current_element_index_1d();
  if (i < num_pixels)
  {
    d_dst_pixels[i] = transform_pixel(d_src_pixels[i]);
  }
}

__global__ void cuda_native_kernel_2d(int width, int height, rgb_t* d_src_pixels, byte_t* d_dst_pixels)
{
  int num_pixels = width * height;
  int i = vu::device::current_element_index_2d(width, height);
  if (i < num_pixels)
  {
    d_dst_pixels[i] = transform_pixel(d_src_pixels[i]);
  }
}

// #define _KERNEL_1D_

#ifdef _KERNEL_1D_
#define cuda_native_kernel cuda_native_kernel_1d
#define calculate_execution_configuration calculate_execution_configuration_1d
#else
#define cuda_native_kernel cuda_native_kernel_2d
#define calculate_execution_configuration calculate_execution_configuration_2d
#endif _KERNEL_1D_

void transform_image_gpu_native(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels)
{
  try
  {
    int num_pixels = width * height;

    assert(num_pixels == static_cast<int>(dst_pixels.size()));

    rgb_t* d_src_pixels = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_src_pixels), num_pixels * sizeof(rgb_t));
    cudaMemcpy(d_src_pixels, src_pixels, num_pixels * sizeof(rgb_t), cudaMemcpyHostToDevice);

    byte_t* d_dst_pixels = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_dst_pixels), num_pixels);

    const auto ec = vu::host::calculate_execution_configuration(width, height, cuda_native_kernel);
    cuda_native_kernel<<<ec.first, ec.second>>>(width, height, d_src_pixels, d_dst_pixels);

    vu::host::device_synchronize();

    cudaMemcpy(&dst_pixels[0], d_dst_pixels, num_pixels, cudaMemcpyDeviceToHost);
  }
  catch (const std::exception& e)
  {
    cout << e.what() << endl;
  }
}

void transform_image_gpu_thrust(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels)
{
  try
  {
    int num_pixels = width * height;

    assert(num_pixels == static_cast<int>(dst_pixels.size()));

    thrust::device_vector<rgb_t>  d_src_pixels(src_pixels, src_pixels + num_pixels);
    thrust::device_vector<byte_t> d_dst_pixels(num_pixels);

    thrust::transform(
      d_src_pixels.begin(), d_src_pixels.end(),
      d_dst_pixels.begin(),
      [=] __device__(rgb_t& src_pixel) // required `--extended-lambda`
    {
      return transform_pixel(src_pixel);
    });

    vu::host::device_synchronize();

    thrust::copy(d_dst_pixels.begin(), d_dst_pixels.end(), &dst_pixels[0]);
  }
  catch (const std::exception& e)
  {
    cout << e.what() << endl;
  }
}

void transform_image_gpu_wrapper(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels)
{
  try
  {
    int num_pixels = width * height;

    assert(num_pixels == static_cast<int>(dst_pixels.size()));

    auto d_src_pixels = cuda::memory::device::make_unique<rgb_t[]>(num_pixels);
    cuda::memory::copy(d_src_pixels.get(), src_pixels, num_pixels * sizeof(rgb_t));

    auto d_dst_pixels = cuda::memory::device::make_unique<byte_t[]>(num_pixels);

    const auto ec = vu::host::calculate_execution_configuration(width, height, cuda_native_kernel);
    cuda::launch(
      cuda_native_kernel,
      cuda::make_launch_config(ec.first, ec.second),
      width, height, d_src_pixels.get(), d_dst_pixels.get());

    vu::host::device_synchronize();

    cuda::memory::copy(&dst_pixels[0], d_dst_pixels.get(), num_pixels * sizeof(byte_t));
  }
  catch (const std::exception& e)
  {
    cout << e.what() << endl;
  }
}
