## 性能与竞品对比
| 实现(8192*8192, FP32) | 耗时 | 有效带宽 | 加速比 |
| :--- | :--- | :--- | :---|
| thread-per-row | 867.1 us | 288GB/s | 3.04x |
| warp-per-row + float4 | 284.4 us | 879GB/s | --- |
| cublasSgemv | 286.7 us | 879GB/s | 1.01x |
| torch.mv | 291.8 us | 856GB/s |  1.03x |
## 正确性
| 实现 | 8192 * 8192 | 4096 * 1023 | 1 * 4096 |
| :--- | :--- | :--- | :---|
| thread-per-row | 2.94e-06 | 8.80e-07 | 5.68e-07 |
| warp-per-row + float4 | 2.12e-07 | 3.41e-07 | 3.41e-07 |
| cublasSgemv | 3.43e-07 | 2.45e-07 | 0.00e+00 |
| torch.mv | 2.8e-07 |