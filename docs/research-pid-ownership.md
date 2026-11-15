# Launch / PID 任务认领视图调研（2026-11-15）

目标：在查看单个 tile 的访存前，先讲清 launch 网格、PID 数、PID 到逻辑任务的映射、同一 PID 的串行认领，以及跨 PID 的可并行关系。GQA 的“按 Q 输出任务分配”与“按 KV head 分配后循环多个 Q 任务”是两种示例策略，不是对所有 GQA 内核的统一结论。

| 来源 | 可复用的语义/呈现 | 与 OP/SHOW 的差异、许可 |
| --- | --- | --- |
| [Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | `program_id`、二维/三维 grid 与 Q block、head 的关系；K/V 迭代在 program 内 | 官方示例不是本项目的两策略对照；只核对语义，未复制源码或图像。 |
| [Triton 官方源码 06-fused-attention](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) | 进一步核对 `tl.program_id(0)` 对应输出行块、head 轴与内层 key 循环 | 本地 skill 中的上游 checkout 也可复核；无代码复制。 |
| [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention) | GQA 中多个 Q head 映射到较少 KV head；任务划分与 IO 复杂性需要分开讲 | 本项目只展示小尺寸逻辑认领，不宣称某策略更快；仓库许可见其 LICENSE，未复制代码。 |
| [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | grid/block 独立执行，实际调度顺序不确定 | 仅以源码可证的依赖表达“可并行”，不画实测 SM 波次。 |
| [NVIDIA WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions) | 一个 warp 协同执行矩阵片段的 load、MMA 与 store；对照仓库的 32 线程 WMMA 教学源码 | 仅把 launch 与 K 循环的源码关系写入现有示例，不把 Tensor Core 阶段画成实测时长。 |
| 仓库现有 v1/v2 | v1 已有 `worker_grid`、tile 索引、阶段和来源行；v2 已有 QKV PID 认领表 | v1 原先强制一 worker 一 tile。沿用其确定性展开与界面色板，扩展为可选的每 PID 多任务认领；不复制 v2 的算子专属逻辑。 |

实现采用可选的 v1 `claim_loop`：同一 worker 的 `claim_i` 按顺序展开为多个输出 tile，CLI 检查覆盖恰好一次并添加同 PID 的串行依赖。可选 `parallelism.task_group` 用确定性公式给任务标注共享输入组（如 KV head），同样校验范围和源码引用。界面从展开后的 worker、任务和依赖生成视图；说明文字须有源码引用。Q/K/V 的共享归属不等于该数据必然常驻缓存，也不代表设备的真实调度时序。
