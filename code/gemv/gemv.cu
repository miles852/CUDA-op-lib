#include <cublas_v2.h>
#include "../common/common.cuh"

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for(int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__global__ void gemv_thread_row(const float* __restrict__ A, const float* __restrict__ x,
                                float* __restrict__ y, int M, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= M) return;
    float acc = 0.f;
    for(int j = 0; j < N; j++) acc += A[(size_t)N * r + j] * x[j];
    y[r] = acc;
}

__global__ void gemv_warp_row(const float* __restrict__ A, const float* __restrict__ x,
                              float* __restrict__ y, int M, int N) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if(r >= M) return;
    const float* Ar = A + r * N;
    float acc = 0.f;
    if((N & 3) == 0) {
        const float4* a4 = reinterpret_cast<const float4*>(Ar);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for(int j = lane; j < N / 4; j += 32) {
            acc += a4[j].x * x4[j].x
                + a4[j].y * x4[j].y
                + a4[j].z * x4[j].z
                + a4[j].w * x4[j].w;
        }
    } else {
        for(int j = lane; j < N; j += 32) {
            acc += Ar[j] * x[j];
        }
    }
    acc = warp_sum(acc);
    if(lane == 0) y[r] = acc;
}

static void gemv_cpu(const float* A, const float* x, float* y, int M, int N) {
    for(int i = 0; i < M; i++) {
        double acc = 0;
        for(int j = 0; j < N; j++) acc += A[i * N + j] * x[j];
        y[i] = (float)acc;
    }
}

int main(int argc, char** argv) {
    std::string mode = run_mode(argc, argv);
    cublasHandle_t h;
    cublasCreate(&h);
    const float one = 1.f, zero = 0.f;
    for (auto shape: std::vector<std::pair<int, int>>{{8192, 8192}, {4096, 1023}, {1, 4096}}) {
        int M = shape.first, N = shape.second;
        std::vector<float> hA((size_t)N * M), hx(N), ref(M);
        fill_rand(hA, -1, 1, 1);
        fill_rand(hx, -1, 1, 2);
        gemv_cpu(hA.data(), hx.data(), ref.data(), M, N);
        DevBuf<float> A(hA), x(hx), y(M);

        auto naive = [&] { gemv_thread_row<<<(M + 255) / 256, 256>>>(A.d, x.d, y.d, M, N); };
        auto warp = [&] { gemv_warp_row<<<(M * 32 + 255) / 256, 256>>>(A.d, x.d, y.d , M, N); };
        auto cublas = [&] { cublasSgemv(h, CUBLAS_OP_T, N, M, &one, A.d, N, x.d, 1, &zero, y.d, 1); };

        naive();
        double e0 = rel_err(y.down(), ref);
        warp();
        double e1 = rel_err(y.down(), ref);
        cublas();
        double e2 = rel_err(y.down(), ref);
        printf("gemv M=%d N=%d  err naive %.2e warp %.2e cublas %.2e\n", M, N, e0, e1, e2);

        if (mode != "check" && M == 8192) {
        double bytes = 4.0 * ((size_t)M * N + N + M);  // 读 A、读 x、写 y
        float t0 = time_kernel(naive), t1 = time_kernel(warp), t2 = time_kernel(cublas);
        printf("  naive %.1f us %.0f GB/s | warp+float4 %.1f us %.0f GB/s | cublasSgemv %.1f us %.0f GB/s\n",
                t0 * 1e3, gbps(bytes, t0), t1 * 1e3, gbps(bytes, t1), t2 * 1e3, gbps(bytes, t2));
        }   
    }
    cublasDestroy(h);
    return 0;
}
