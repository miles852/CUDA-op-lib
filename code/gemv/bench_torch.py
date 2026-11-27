"""GEMV 的 PyTorch 对照组：8192×8192 FP32，torch.mv(out=)。  python3 bench_torch.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import torch  # noqa: E402
from bench_util import gbps, header, rel_err, time_ms, uniform, dev  # noqa: E402,F401


M = N = 8192
A, x = uniform((M, N), -1, 1, 1), uniform((N,), -1, 1, 2)
y = torch.empty(M, device=dev)
ref = A.double() @ x.double()
header()
torch.mv(A, x, out=y)
t = time_ms(lambda: torch.mv(A, x, out=y))
print(f"gemv {M}x{N}  torch.mv {t * 1e3:.1f} us {gbps(4.0 * (M * N + N + M), t):.0f} GB/s  err {rel_err(y, ref):.1e}")
