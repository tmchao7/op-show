# Tensor Core GEMM 示例调研（2026-11-09）

目标：用 OP/SHOW 演示一个 CUDA WMMA GEMM 的输出 tile、A/B 输入分片、K 循环与 M/N/K 尾部 mask。当前环境为 macOS arm64，无 nvcc 或 NVIDIA GPU；交付源码约束的逻辑模型，不声称已在设备上编译或执行。

| 来源 | 可复用内容 | 与本任务的关系及许可 |
| --- | --- | --- |
| [NVIDIA CUDA Programming Guide §5.4.11 Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions)（在线查阅；本地 CUDA 13.3 快照） | WMMA fragment、16×16×16 half × half → float、warp 一致调用、load/store_matrix_sync 的 256-bit 指针对齐与 leading dimension 要求。 | 用作 API 事实来源，不复制示例代码。 |
| [NVIDIA cudaTensorCoreGemm 样例](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/3_CUDA_Features/cudaTensorCoreGemm) | 官方 WMMA GEMM 实现和运行环境参考。 | 功能完整但代码规模及优化层级不适合逐格讲解；本项目写一个独立的小型教学算子。仓库元数据未给出可直接引用的 SPDX 标识，因此不复制其代码。 |
| [NVIDIA CUTLASS GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api.md) | 生产 GEMM 中线程块、warp、MMA 分层和 tile iterator 的现有方案。 | 后续若需生产性能优先评估 CUTLASS；当前教学模型无需引入该依赖或模板体系。 |
| [CUDA Tiled GEMM Visualizer](https://www.joelfernandes.org/simulators/cuda-tiled-gemm/index.html) | A/B/C 矩阵与当前 K 分片联动的交互结构。 | 重画关系与配色，不复制图像、代码或页面资产。 |

方案：每个 CUDA block 只有一个 32-lane warp，负责一个 16×16 的 C tile。每次将 A 的 16×16、B 的 16×16 分片按边界零填充到 32-byte 对齐的 shared memory，再做 wmma::load_matrix_sync、wmma::mma_sync；结束后把 accumulator 写入对齐的 shared memory，再按 M/N 边界写回 C。A/B 为行主序 half，C 为行主序 float。该示例追求可检查性，不用于性能比较。
