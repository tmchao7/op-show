# OP/SHOW 开发规则

本文件适用于仓库内的后续 agent。先读 `README.md`、`docs/model-format.md` 和相关源码，再改动项目。

## 增加功能前先查现有方案

1. 先确定需求涉及的算子、后端、数据模型和交互，再检查仓库内是否已有可复用实现。
2. 搜索 GitHub 上的开源项目、现成工具和官方插件；同时查阅 NVIDIA CUDA/Triton、Ascend/Triton-Ascend 等**对应后端的最新官方文档与示例**。知乎、CSDN 等技术社区可用于寻找示意图和实践经验，关键语义仍以源码和官方资料核对。
3. 在动手前留下简短调研记录：链接、查阅日期、可复用部分、与 OP/SHOW 的差异、许可证或使用限制。能复用成熟实现、格式或交互模式时优先复用；确有缺口再实现本项目所需的最小部分。不要照搬受版权保护的图片、文字或代码。
4. 官方插件或工具若能完成需求，先评估其接口、维护状态及适配成本。借鉴视觉呈现时重画数据关系，并在代码或文档中写清借鉴来源。

### 已整理的参考入口

这些链接是既有调研的起点。开始新功能时重新检查文档版本、维护状态和许可证，并继续搜索更贴近该需求的实现；不要把下表当作穷尽列表。

| 类型 | 链接 | 可借鉴的部分 |
| --- | --- | --- |
| NVIDIA 官方 | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | block、thread、内存与同步语义的权威依据。 |
| Triton 官方 | [Triton Programming Guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) | program 与数据 tile 的关系及图示。 |
| Ascend 官方/上游 | [Triton-Ascend](https://github.com/triton-lang/triton-ascend)、[msOpProf](https://github.com/Ascend/msopprof) | 后端实现与模拟器流水线视图；先确认适用版本。 |
| 源码驱动可视化 | [TileLens](https://github.com/Deep-Learning-Profiling-Tools/tilelens)、[TritonParse](https://github.com/meta-pytorch/tritonparse) | tile 访问、源码映射及编译过程的现有方案，评估能否直接接入。 |
| 动画交互 | [CUDA Tiled GEMM Visualizer](https://www.joelfernandes.org/simulators/cuda-tiled-gemm/index.html)、[CUDA Tile Scheduling Studio](https://cudacourseh100.github.io/raster/)、[CUDA Visualizer](https://poojithdevan4d.github.io/cuda-visualizer/) | 输入/输出矩阵联动、PID/wave/padding、访存与执行动画。 |
| 逐级数据搬运 | [cuda-matmul-study](https://github.com/balaji-ch/cuda-matmul-study) | 分步骤展现 tile 在内存层级之间的路径。 |
| 社区示意 | [知乎算子讲解](https://zhuanlan.zhihu.com/p/2016054257409867829)、[CSDN 算子文章](https://blog.csdn.net/m0_61864577/article/details/167215457) | 寻找静态矩阵示意和讲解方式，再按本项目源码重画。 |
| 暖色与矩阵配色 | [Rosé Pine Dawn](https://github.com/rose-pine/rose-pine-palette)、[react-matrix](https://github.com/bronz3beard/react-matrix) | 暖白分层、柔和色组和浅色矩阵阅读体验。 |

[Triton-Ascend 的 TileLens/Triton-Viz 讨论](https://github.com/triton-lang/triton-ascend/issues/1908)记录了相关工具在 Ascend 上的适配问题；引用时应核对 issue 的最新进展。

## 源码约束与逻辑正确性

- 先读算子 kernel、launcher、形状和 stride 来源，再定义 tile、PID、地址、mask 和事件依赖。执行单元 ID 与逻辑 tile ID 必须分开。每条确定映射都应有可核对的源码引用。
- v1 描述规则输出 tile；v2 当前只支持 `qkv-head-grouped` 逻辑任务轨迹。扩展格式前先确认现有 schema 是否足够，避免为单个示例做无法复用的分支。
- 不确定的索引、同步或指针关系使用 `mapping.status: "unknown"`，不能播放猜测轨迹。示例尺寸、连续布局和 AIV 数须标为示例，不能称为实测调度或性能。
- Wave 是源码中的每 PID 循环片段，不是跨 PID 全局屏障。远端路径遵守 store → 本 PID wave 任务结束 → fence → notify → wait → consume → output；本地直写没有 wait。

## 前端视觉与交互

- 沿用暖白界面：页面 `#FBF7F1`、面板 `#FFFDF9`、正文 `#342A24`、分隔线 `#E7D8C8`、主色 `#B85437`。优先使用 `op_show/web/styles.css` 中的语义变量，避免零散硬编码和经典深色开发者界面。
- 排版先建立清楚的阅读顺序和对齐关系：标题、控制器、矩阵、时间轴、详情、源码。控制留白、字号层级、卡片宽度、边框与间距；新视图要与旧视图协调。
- G0–G4 的柔和颜色只表达**数据分组**；执行状态用明暗和描边表达；选中态用深色外描边；tail/masked 用文字或斜纹表达。不要只靠颜色传递含义。
- 保持中文默认与中英切换，新文案两种语言都要补齐。检查对比度、键盘焦点、悬停、长文本、横向滚动和窄屏标签页。桌面与手机宽度都要实际检查；若环境无法模拟窄屏，要明确记录未验证项。
- 不随意引入前端框架或构建步骤。确需依赖时先调研已有轻量方案及维护成本。

## 交付检查

- 每个示例核对规则尺寸与尾 tile、任务覆盖、地址和 mask；交互检查选中联动、播放/暂停、前后单步、拖动、Wave 跳转、Q/K/V 与源码定位。
- 运行 `python3 -m unittest discover -s tests -v`、`node --check op_show/web/app.js`、`node --check op_show/web/trace.js`、`git diff --check`。新增校验逻辑时写能抓住真实错误的测试，并保留 v1 回归。
- 交付时说明改了什么、参考了什么、实际验证了什么，以及尚未验证的边界。提交或推送遵循当前用户任务的授权与仓库流程。
