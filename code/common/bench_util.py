"""各算子 bench_torch.py 共用的 PyTorch 计时与误差工具，口径和 common.cuh 一致：
warmup 20、计时 100、取中位数；误差 max|got - ref| / max|ref|，参考值用 float64 在 GPU 上算。
结果张量尽量预先分配（out=），不把显存分配算进计时。
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = False  # 纯 FP32，和 cuBLAS 对照组的 CUBLAS_DEFAULT_MATH 一致
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"


def header():
    """打印 GPU 型号和 torch / CUDA 版本，方便和 CUDA 版 benchmark 的输出对照。"""
    print(torch.cuda.get_device_name(0), "torch", torch.__version__, "cuda", torch.version.cuda)


def time_ms(fn, warmup=20, iters=100):
    """返回 fn 在 GPU 上耗时的中位数（ms），和 common.cuh 的 time_kernel 同一套做法。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(iters):
        t0.record()  # event 记录在当前 stream 上，和 fn 里的 kernel 按顺序执行
        fn()
        t1.record()
        t1.synchronize()  # 等这一次跑完再读耗时
        ms.append(t0.elapsed_time(t1))
    ms.sort()
    return ms[iters // 2]


def rel_err(got, ref):
    """max|got - ref| / max|ref|，都转成 float64 再算。"""
    return ((got.double() - ref.double()).abs().max() / ref.double().abs().max()).item()


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def tflops(flop, ms):
    return flop / (ms * 1e-3) / 1e12


def uniform(shape, lo, hi, seed, dtype=torch.float32):
    """在 GPU 上生成 [lo, hi) 均匀分布的张量。每次调用用独立的 Generator，同一个 seed 结果相同。"""
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.empty(shape, device=dev, dtype=dtype).uniform_(lo, hi, generator=g)
