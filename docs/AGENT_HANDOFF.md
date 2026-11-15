# 给下一位开发 agent 的交接提示词

可将下面的提示词直接复制到新会话，并在最后补上本次具体需求。

```text
你将继续开发本地仓库 /Users/tmchao7/repos/02_personal/op-show（公开仓库 https://github.com/tmchao7/op-show）。先读取仓库根目录 AGENTS.md，并遵守其中的调研、源码约束、前端设计和验收规则；再读 README.md、docs/model-format.md、skills/op-show/SKILL.md，以及与你要改动的文件。

项目目标：根据用户算子源码生成可审查的 Tile 矩阵和数据流动画，不依赖 GPU/NPU，也不把理想逻辑进度冒充真实设备调度或耗时。CLI 是 `python3 -m op_show validate <json>` 和 `python3 -m op_show serve <json>`。v1 是规则输出 tile；v2 是 `qkv-head-grouped` 任务轨迹格式。仓库内有七个 v1 模型和一个 Triton-Ascend v2 QKV 示例；v2 上游源码附有许可证声明。

当前已实现：op_show/trace.py 展开并检查 v2 任务的 PID、flat_idx、地址、signal slot、输出覆盖和事件依赖；op_show/web/trace.js、index.html、styles.css 展示 Producer 数据矩阵、PID 认领、peer_mem 路径、Consumer 输出矩阵、Wave 播放与 Q/K/V 切换。examples/ascend_qkv_alltoall.json 是 v2 演示；tests/test_trace.py 另有合成测试源。v1 示例覆盖 Vector Add、Reduce、GEMV、GEMM 和 FlashAttention-2 风格前向。

v2 核对要点（自定义模型）：Producer/Consumer 的 PID 划分、flat_idx 与 consumer_flat_idx、peer Q/K/V 偏移、signal slot、输出覆盖；远端是本 PID 的 wave 任务循环完成后 fence，再逐任务 notify；Consumer wait/consume 后写输出；本地目标直写，无 wait；Wave 不是全局屏障。连续 stride 与核心数/序列长度由模型作者声明，不得称为实测值。

增加新功能前，先使用 AGENTS.md 中整理的官方文档、开源可视化项目、知乎/CSDN 和配色参考链接作为起点，再搜索更贴近需求的最新方案。记录链接、查阅日期、许可、差异和采用理由，再实现缺失部分。优先沿用仓库暖白配色及 G0–G4 数据分组语义，认真检查排版、对比度、中英切换和手机布局。不要复制第三方图片资产；不要把无法从源码证明的映射画成确定结果。

验证入口：`python3 -m unittest discover -s tests -v`、两个 JS 语法检查和 `python3 -m op_show validate examples/<name>.json`。2026-11-15 的本地检查为 32 项通过、1 项 GPU 测试因缺少 PyTorch/Triton 而跳过；FlashAttention 尾块在桌面和 390px 宽度下检查了矩阵、语言、单步、拖动和源码定位。后续改动仍需重新验收。macOS arm64 无 CUDA/NPU，请勿声称做过设备运行验证。

请先检查 `git status` 和最新提交，理解当前实现，再针对下面的新需求完成调研、设计、代码、测试和可视化验收。保留现有功能。完成后报告调研依据、实现、测试结果及剩余限制。

本次需求：[在这里填写]
```
