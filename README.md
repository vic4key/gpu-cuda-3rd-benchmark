### CUDA Benchmark Testing

Simple benchmark testing CUDA's cpp wrapper libraries
>1. [nvidia native c](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
>2. [nvidia's thrust](https://github.com/NVIDIA/thrust)
>3. [eyalroz's cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers)
>4. [harrism's hemi](https://github.com/harrism/hemi) (not yet)

Installation
>1. Install [Vutils](https://github.com/vic4key/Vutils.git) @ https://github.com/vic4key/Vutils.git
>2. Run the `Setup.cmd` file to install required [sub-modules](https://github.com/vic4key/gpu-cuda-3rd-benchmark/blob/master/.gitmodules)
>3. Open `gpu-cuda-3rd-benchmark.sln` in VS to start to work

Result (`CPU / Intel(R) Core(TM) i5-11400H` & `GPU / NVIDIA GeForce RTX 3050`)

>![](screenshots/release-mode.png?)

>![](screenshots/debug-mode.png?)
