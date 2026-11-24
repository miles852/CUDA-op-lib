// 打印 0 号 GPU 的关键属性，用来算 FP32 峰值、屋脊点，并填进 ENV.md
//   nvcc -arch=sm_90 devq.cu -o devq && ./devq
#include <cuda_runtime.h>
#include <cstdio>

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int clk = 0, mclk = 0, bus = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);              // kHz
  cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);       // kHz
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);   // bit
  printf("%s cc%d.%d SM=%d L2=%dKB smemPerSM=%zuKB smemPerBlockOptin=%zuKB regsPerSM=%d "
         "clk=%dMHz memclk=%dMHz bus=%d\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize / 1024,
         p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor,
         clk / 1000, mclk / 1000, bus);
  // FP32 峰值 = SM 数 × 每 SM 128 个 FP32 单元 × 2（FMA）× 时钟；Hopper / Ampere 都是每 SM 128 个
  printf("FP32 peak (128 FP32 lanes/SM) = %.1f TFLOPS\n", p.multiProcessorCount * 128.0 * 2 * clk * 1e3 / 1e12);
  return 0;
}
