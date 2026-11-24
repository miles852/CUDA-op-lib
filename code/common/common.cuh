// 所有算子共用的主机端工具：错误检查、中位数计时、带宽 / 算力换算、随机数、相对误差、显存缓冲。
// 只放和具体算子无关的东西，每个算子的 kernel 和 CPU 参考都写在各自的 .cu 里。
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

// 检查 CUDA runtime 调用的返回值，出错时打印文件名、行号和错误信息后直接退出
#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// 返回中位数耗时（ms）。中位数比均值更不容易被偶发的频率波动带偏
// run 里可以是一次或多次 kernel 启动（例如两段式归约），计的是整个 run 的耗时
inline float time_kernel(const std::function<void()>& run, int warmup = 20, int iters = 100) {
  // 预热：让 GPU 升到稳定频率，并把 kernel 加载、cuBLAS 句柄初始化等一次性开销排除在计时之外
  for (int i = 0; i < warmup; ++i) run();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  std::vector<float> ms(iters);
  for (int i = 0; i < iters; ++i) {
    // 两个 event 和 kernel 在同一个（默认）stream 上排队，两者之差就是 GPU 上的执行时间
    cudaEventRecord(t0);
    run();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);  // 每次都等 GPU 跑完，单独记录这一次的耗时
    cudaEventElapsedTime(&ms[i], t0, t1);
  }
  CUDA_CHECK(cudaGetLastError());  // kernel 启动错误是异步报告的，在这里统一检查
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  std::sort(ms.begin(), ms.end());
  return ms[iters / 2];
}

// 有效带宽 / 算力：bytes 是算子理论上最少要搬的字节数，flop 是理论运算量；GB / TFLOPS 都按 10 进制
inline double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double tflops(double flop, float ms) { return flop / (ms * 1e-3) / 1e12; }

// [lo, hi) 均匀分布。seed 固定，每次运行的输入完全一样，误差和性能可复现
inline void fill_rand(std::vector<float>& v, float lo, float hi, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  for (auto& x : v) x = d(g);
}

// max|got - ref| / max|ref|：对整体量级归一，避免接近 0 的元素把相对误差放大
inline double rel_err(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    num = std::max(num, (double)std::fabs(got[i] - ref[i]));
    den = std::max(den, (double)std::fabs(ref[i]));
  }
  return num / (den > 0 ? den : 1);
}

// 显存缓冲：析构时自动释放，up / down 在主机 vector 和显存之间拷贝
template <class T>
struct DevBuf {
  T* d = nullptr;
  size_t n;
  explicit DevBuf(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&d, n * sizeof(T))); }
  explicit DevBuf(const std::vector<T>& h) : DevBuf(h.size()) { up(h); }
  ~DevBuf() { cudaFree(d); }
  DevBuf(const DevBuf&) = delete;  // 禁止拷贝，否则两个对象析构时会重复 cudaFree 同一块显存
  DevBuf& operator=(const DevBuf&) = delete;
  void up(const std::vector<T>& h) { CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> down() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// 命令行第一个参数：check 只做正确性检查，bench 测性能（便宜的正确性检查照样会跑），省略时两者都做
inline std::string run_mode(int argc, char** argv) { return argc > 1 ? argv[1] : "all"; }
