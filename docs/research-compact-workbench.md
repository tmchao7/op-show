# 同屏算子工作台调研（2026-11-15）

目标：在 1440×900 的普通桌面视口同时阅读 PID 认领、输入到输出的矩阵关系和逻辑进度，减少为了核对同一任务而上下滚动。先检查了仓库已有的 ownership、GEMM dependency map、v2 trace、inspector 和播放器；这些数据与交互可以直接重排，JSON schema 无需扩展。

| 来源 | 链接 | 可借鉴处 | 与 OP/SHOW 的差异、使用限制 |
| --- | --- | --- | --- |
| NVIDIA cuBLASDx GEMM 文档 | https://docs.nvidia.com/cuda/cublasdx/using_cublasdx.html | A/B/C 形状、分块与执行层次的官方术语 | 仅核对讲解用语；本项目不调用 cuBLASDx，也不展示其性能数据。 |
| NVIDIA CUTLASS Efficient GEMM | https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md | threadblock 输出 tile、K 方向循环与层级关系 | 用于核对 PID/block 与 C tile 的解释；不复制图文或代码，页面仍依据具体示例源码。 |
| CUDA Tiled GEMM Visualizer | https://www.joelfernandes.org/simulators.html | 输入、输出、当前阶段和源码联动 | 借鉴阅读顺序；OP/SHOW 的模型由用户源码约束，重画矩阵，不复用其页面资产。 |
| cuda-matmul-study | https://github.com/balaji-ch/cuda-matmul-study | 把逐级搬运分阶段展示 | 只借动画叙事方式；本轮没有新增硬件层级模拟。开源仓库的具体资产与许可证须在复用前另查。 |
| 知乎与哔哩哔哩社区检索 | https://www.zhihu.com/search?type=content&q=CUDA%20GEMM%20tile%20%E7%A4%BA%E6%84%8F%E5%9B%BE · https://search.bilibili.com/all?keyword=CUDA%20GEMM%20tile | 寻找讲解时反复出现的 PID、输入分片、输出落点、阶段这几类信息 | 本次访问完整页面遇到 403/412；未据此认定或复制任何具体画面。 |

采用的布局是左侧 PID 认领、中间主矩阵、底部播放器、右侧可切换的摘要/地址/逐格/任务/源码。QKV 将 Producer 与 Consumer 矩阵放在同一行，任务路径压成一条可读的步骤带；较长的 PID 认领表与完整路径放在右侧任务标签。矩阵自身滚动，桌面整页保持固定，窄屏回到纵向滚动与双主图标签。

这里只借鉴公开资料的信息组织方法。没有复制第三方图片、文字、图标或代码。暖白色值沿用仓库自己的设计变量；逻辑 tick 仍表示依赖层级，不表示真实设备耗时。
