// 带宽探针：1 GiB -> 1 GiB 的 float4 拷贝。测出的有效带宽作为访存算子的 100% 基准
//   nvcc -O3 -arch=sm_90 copy.cu -o copy && ./copy
#include "common.cuh"

// grid-stride 循环，每个线程每次搬一个 float4（16 B）
__global__ void copy_f4(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x)
    out[i] = in[i];
}

int main(int argc, char** argv) {
  std::string mode = run_mode(argc, argv);
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  size_t n = mode == "check" ? (1ull << 20) : (1ull << 28);  // check 用 4 MiB，bench 用 1 GiB
  std::vector<float> h(n);
  fill_rand(h, -1, 1, 0);
  DevBuf<float> a(h), b(n);
  // grid 固定为 SM 数 × 8 个 block，和数据量无关；剩下的元素由 grid-stride 循环分摊给每个线程
  auto run = [&] { copy_f4<<<sms * 8, 512>>>((const float4*)a.d, (float4*)b.d, n / 4); };

  run();  // 先跑一次，检查拷贝结果是否和输入一致
  printf("copy %zu floats  err %.1e\n", n, rel_err(b.down(), h));
  if (mode == "check") return 0;

  float ms = time_kernel(run);
  printf("copy 1GiB->1GiB  %.3f ms  %.0f GB/s\n", ms, gbps(2.0 * n * 4, ms));  // 读 + 写各 1 GiB
  return 0;
}
