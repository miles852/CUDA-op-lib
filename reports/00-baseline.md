- FP32理论峰值: 84.0 TFLOPS
- 理论带宽: 1TB/s 实测copy带宽: 0.899 TB/s
- 脊点: 84 FLOP/B

## Roofline
| 算子 | FLOPs | 最少字节 | I ( FLOP/B ) | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| GEMV | 2MN | 4(MN + M + N) | 0.5 | 访存受限 |