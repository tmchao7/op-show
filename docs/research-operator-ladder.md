# 算子练习序列调研（2026-11-15）

目标：为 Vector Add → Reduce → GEMV → GEMM → FlashAttention v2 建立可逐步检查的源码约束示例。本轮补齐 CUDA Reduce、GEMV 和 Triton CUDA FlashAttention-2 风格前向。当前 macOS arm64 环境没有 `nvcc` 或 NVIDIA GPU；源码是教学算子，页面只展示逻辑 tile，不声称设备运行结果。

| 来源 | 可复用部分与取舍 | 许可或使用限制 |
| --- | --- | --- |
| [NVIDIA CUDA Programming Guide：Warp Shuffle](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions) | 核对 warp 内 `__shfl_down_sync` 的参与线程与同步语义。Reduce/GEMV 各用完整 32-lane warp 处理一行；无跨 warp 求和。 | 官方 API 文档，仅依据语义编写独立示例，不复制示例代码。 |
| [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) | 检查现成 `matrixMul`、reduction 及 CUB 方案；生产用途优先使用维护中的库/样例。OP/SHOW 需要极短且可逐行引用的源码，故保留独立教学算子。 | 仓库 [LICENSE](https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE) 指向 CUDA EULA；不复制源码。 |
| [NVIDIA CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples) | 检查现成 cuBLAS/cuSPARSE 入口；它们适合执行算子，不提供本项目需要的逐 tile 源码约束动画。 | Apache 2.0；未复用代码。 |
| [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | 核对 FlashAttention v2 风格的在线 softmax、Q/K/V 分片和输出归一化；不把示意阶段称为真实设备调度。 | 官方教程；2026-11-15 核对当前页面，上游 Triton 仓库为 MIT 许可；不复制代码。 |
| [Dao-AILab FlashAttention](https://github.com/Dao-AILab/flash-attention) 与 [FlashAttention-2 论文](https://tridao.me/publications/flash2/flash2.pdf) | 核对 v2 的算法边界与并行划分。先评估现有库/工具，OP/SHOW 只补可视化说明层。 | 上游源码与论文各自保留版权；不复制代码、图像或实测数据。 |

本轮模型复用 v1：输出行按 4 行一个逻辑 tile，`blockIdx.x` 是执行单元 ID，tile ID 是输出行组。K 循环按 32 元素切片；尾场景同时包含输出行尾块与 K 尾分片。地址为连续行主序元素偏移。新增的 `row_k_view` 沿用 `matrix-frame`、`matrix-axis` 和现有暖白配色，分别画 Reduce 的 X → Y 与 GEMV 的 A × x → y；它只显示逻辑数据位置，不声称 lane 的真实发射时刻。

## FlashAttention-2 教学示例范围

2026-11-15 核对 [Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)、[上游教程源码](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)及 [FlashAttention-2 论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html)。上游 Triton 仓库为 MIT 许可；这里不复制教程源码。教程明确是 FlashAttention v2 算法实现，论文说明其输出块并行、在线 softmax 与更细的 warp 工作划分。本项目只借算法关系重写一个单 head、非 causal、无 dropout、固定 `D=16` 的教学前向算子：每个 program 负责一个 Q 输出行块，依次读 K/V 块，用行最大值与行和维护在线归一化状态，并累积 O。它没有生产级 FA2 的 warp/流水线优化，也没有编译或运行验证。
