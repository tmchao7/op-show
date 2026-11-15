# 算子可视化参考调研（2026-11-09）

本轮目标：从仓库既有参考入口出发，扩大检索范围（GitHub、知乎、CSDN、NVIDIA、昇腾 CANN/msOpProf 等），为 OP/SHOW 选择可复用的**讲解与交互模式**，并把工具做得更完整。原则仍是 [AGENTS.md](../AGENTS.md) 的要求：**OP/SHOW 只做源码支撑的逻辑讲解，不伪造设备耗时、吞吐、利用率或实测 trace**。

## 检索方式与局限

- 使用网页搜索 + 直接抓取；结果按 2026-11-09 的返回内容记录。
- GitHub raw / PyPI 部分页面被超时或人机校验拦截；知乎正文返回 403。这些条目只作为**线索与概念参考**，未读取全文，也没有复制其代码、图片或文案。
- 所有新实现均为本项目重写的 HTML/CSS/JS；未引入第三方依赖、字体、图标或图片资源。

## 参考清单

| # | 来源 | 链接 | 可借鉴的部分 | 与 OP/SHOW 的差异 / 边界 | 许可与抓取情况 |
| --- | --- | --- | --- | --- | --- |
| 1 | TileLens（Triton 可视化工具包） | https://github.com/Deep-Learning-Profiling-Tools/TileLens · https://pypi.org/project/tilelens/ | 以 Triton program/tile 为中心、把源码位置与 tile 对应起来展示 | 它基于运行时/编译器产物；本项目是预计算 JSON，无运行时插桩 | GitHub raw 与 PyPI 抓取失败（超时/校验）；未复用代码，仅参考交互概念 |
| 2 | TritonParse（Meta） | https://github.com/meta-pytorch/tritonparse | 把 Triton 编译/运行产物整理成可浏览的 HTML，便于从源码跳到中间产物 | 本项目不做编译与 IR；只做手工声明模型与源码行证据 | GitHub 抓取超时；未复用代码 |
| 3 | Joel Fernandes · Tiled CUDA GEMM 模拟器 | https://www.joelfernandes.org/simulators/cuda-tiled-gemm/index.html | C tile 选择、阶段步进、共享内存 vs global 的讲解顺序、播放/单步/重置、键盘快捷键；明确“模拟器非实测” | 本项目不模拟共享内存容量、global load 次数或硬件缓存行为；我们的数据来自模型声明 | 已抓取页面；个人站点，许可未标注；未复制其配色、文案或代码 |
| 4 | 昇腾 msOpProf 典型案例（指令流水图） | https://raw.atomgit.com/zzzsss1234/msopprof/raw/master/docs/zh/best_practices/typical_cases.md · 官方仓库 https://github.com/Ascend/msopprof | **多 lane 指令流水图**的表达：MTE2 搬运与 Vector/Cube 计算的并行、double buffer 后的重叠；用时间线解释“谁在等谁” | 它是真实 profiling 数据 + 硬件流水；我们是逻辑 tick 的阶段占位，不标吞吐/耗时 | 文档可抓取，版权归华为/昇腾；仅学习表达方式，不复制图表与文案 |
| 5 | msOpProf 用户指南 | https://www.hiascend.com/en-sg/document/detail/en/CANNCommunityEdition/910/devaids/optool/docs/en/user_guide/msopprof_user_guide.md | 仿真数据导入 MindStudio Insight、按流水线定位瓶颈的工作流 | 本项目不接仿真器、不产生性能结论 | 官方文档；未复制内容 |
| 6 | CANN ops-blas GEMM tiling 讲解（CSDN） | https://blog.csdn.net/m0_68635815/article/details/153204714 | 先讲 Cube/L1 buffer 与 tile 尺寸约束，再解释 tiling 策略的叙述顺序；16×16 最小 tile 等背景 | 本项目不引入 Cube 单元/带宽数字；示例参数是演示值 | 已抓取正文；社区文章，仅参考讲解结构 |
| 7 | 知乎《CUTLASS: 高效矩阵乘实现拆解》 | https://zhuanlan.zhihu.com/p/30058292010 | CUTLASS 分层/tile 讲解的社区线索 | 未读全文（403）；未使用其中图表或代码 | 403 拦截；仅记录线索 |
| 8 | Triton-Ascend profiling 文档 | https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/debug_guide/profiling.md | 昇腾后端 profiling 视图与术语 | 抓取失败；仅记录入口，本项目不引入实测 profile | 抓取失败（网络） |
| 9 | NVIDIA Nsight Systems / Nsight Compute 文档 | https://docs.nvidia.com/nsight-systems/UserGuide/index.html · https://docs.nvidia.com/nsight-compute/ | 时间线/内存层级/roofline 等 profiling 可视化范式 | 本项目明确不展示 roofline、利用率或带宽；只借“按 lane 看并行”的阅读方式 | NVIDIA 官方文档；未复制内容 |
| 10 | 仓库既有入口 | AGENTS.md 的 TileLens/TritonParse、CUDA Tiled GEMM Visualizer、CUDA Tile Scheduling Studio、cuda-matmul-study、Triton-Ascend issue #1908 | 已验证的参考清单与设计边界 | 同上 | 见 AGENTS.md |

## 采纳到 OP/SHOW 的设计

1. **全局长条时间线（v1）**：每个 tile 一条 lane，横轴是逻辑 tick，阶段用色块表示。它借鉴 profiling 工具的多 lane 流水图与 CUDA GEMM 模拟器的“逐阶段查看”模式，但**明确标注为逻辑占位**：不表示耗时、吞吐、SM 占用或硬件流水重叠。点击色块可跳到对应阶段，依赖等待区用斜纹显示。
2. **源码引用索引**：把每个 source.refs 行与引用它的 mapping 字段（grid、tile axis/size、step、loop、address、gemm、v2 formula/event/param）建立索引。源码面板中被引用的行会有标记，点击行显示 “used by” 列表；借鉴 TileLens/TritonParse 的 source↔tile 双向定位思路，但只使用本项目已有的字面证据。
3. **可分享链接**：把模型、场景、选中 tile/task、逻辑 tick、语言编码进 URL hash，并提供“分享视图”按钮。方便在 issue/文档/聊天中指向同一画面；不引入任何后端状态。
4. **可访问性**：时间线色块是 button 并带 aria-label；被引用源码行可 Tab 聚焦、Enter/Space 选择；分享按钮有 title/aria-label；文案中英双语。

## 明确不做的事

- 不展示实测耗时、吞吐、带宽、SM/AIV 利用率、roofline 或“谁更快”的结论。
- 不模拟共享内存容量、缓存命中、double buffer 或硬件调度；这些只在参考来源中出现。
- 不把第三方可视化工具的截图、配色或代码搬进仓库；新 UI 全部为本项目重写。
- 不新增构建步骤或前端依赖。

## 许可结论

- 本调研只借鉴**交互与信息组织方式**，未复制源代码、图片或大段文案。
- 新增的 HTML/CSS/JS 由本项目编写；仓库已移除此前捆绑的第三方算子源码，不再捆绑第三方算子实现。
- 上表中官方/社区页面的版权归各自作者；后续若要复用其具体资产，需单独确认许可。