// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cassert>

#include <vu>
#include <vu_cuda>

#include <soil2/soil2.h>

#include "kernel.h"

int main()
{
  using namespace std;

  assert(vu::host::device_count() > 0);
  cout << vu::host::device_name(vu::host::device_id()) << endl;

  vu::ScopeStopWatchA watcher("BENCHMARK", " ", vu::ScopeStopWatchA::console);

  int width = 0;
  int height = 0;
  int channel = 0;
  auto pixels = SOIL_load_image("data\\test.bmp", &width, &height, &channel, SOIL_LOAD_AUTO);
  assert(pixels != nullptr);

  watcher.log("LOAD DATA   :");

  auto src_pixels = reinterpret_cast<rgb_t*>(pixels);
  bytes_t dst_pixels(width * height);

  const auto fn_conv_and_save = [&](
    const char* id, const char* file_path, pfn_transform_image fn, int num_loop = 100) -> void
  {
    vu::host::device_reset();
    std::fill(dst_pixels.begin(), dst_pixels.end(), 0);
    watcher.reset();
    for (int i = 0; i < num_loop; i++) fn(width, height, src_pixels, dst_pixels);
    watcher.log(id);
    SOIL_save_image(file_path, SOIL_SAVE_TYPE_BMP, width, height, 1, dst_pixels.data());
    vu::host::device_reset();
  };

  {
    fn_conv_and_save("CPU-CPU     :", "data\\test_cpu.bmp", transform_image_cpu);
    fn_conv_and_save("GPU-NATIVE  :", "data\\test_gpu_native.bmp", transform_image_gpu_native);
    fn_conv_and_save("GPU-THRUST  :", "data\\test_gpu_thrust.bmp", transform_image_gpu_thrust);
    fn_conv_and_save("GPU-WRAPPER :", "data\\test_gpu_wrapper.bmp", transform_image_gpu_wrapper);
  }

  SOIL_free_image_data(pixels);
  pixels = nullptr;

  return 0;
}
