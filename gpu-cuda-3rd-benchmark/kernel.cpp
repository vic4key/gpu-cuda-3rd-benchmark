// kernel.cpp
//

#include "kernel.h"

#include <iostream>
#include <cassert>

void transform_image_cpu(int width, int height, rgb_t* src_pixels, bytes_t& dst_pixels)
{
  int num_pixels = width * height;
  for (int i = 0; i < num_pixels; i++)
  {
    dst_pixels[i] = transform_pixel(src_pixels[i]);
  }
}
