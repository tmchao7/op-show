# 给下一位开发 agent 的交接提示词

可将下面的提示词直接复制到新会话，并在最后补上本次具体需求。

```text
你将继续开发本地仓库 /Users/tmchao7/repos/02_personal/op-show（公开仓库 https://github.com/tmchao7/op-show）。先读取仓库根目录 AGENTS.md，并遵守其中的调研、源码约束、前端设计和验收规则；再读 README.md、docs/model-format.md、skills/op-show/SKILL.md，以及与你要改动的文件。

项目目标：根据用户算子源码生成可审查的 Tile 矩阵和数据流动画，不依赖 GPU/NPU，也不把理想逻辑进度冒充真实设备调度或耗时。CLI 是 `python3 -m op_show validate <json>` 和 `python3 -m op_show serve <json>`。v1 是规则输出 tile；v2 当前支持 Triton-Ascend `kernel_qkv_fuse_a2a_head_grouped` 的 QKV All-to-All 任务轨迹。

当前已实现：v2 模型在 examples/ascend_qkv_alltoall.json，完整示例源码与上游许可证在 examples/sources/；op_show/trace.py 展开并检查 PID、flat_idx、地址、signal slot、输出覆盖和事件依赖；op_show/web/trace.js、index.html、styles.css 展示 Producer 数据矩阵、PID 认领、peer_mem 路径、Consumer 输出矩阵、Wave 播放与 Q/K/V 切换。中文默认、可切英文。v1 示例仍可用。

关键核对点：R0 来源、B1、目标 R2、G3 → flat_idx 57 → Producer P09 / pid 18；Consumer C01 / pid 3；S=1300 的最后一个 block 只有 68/112 token 有效。远端是本 PID 的 wave 任务循环完成后 fence，再逐任务 notify；Consumer wait/consume 后写输出。本地目标直写，无 wait。Wave 不是全局屏障。示例假设 Q/K/V 与输出张量连续，32 AIV 和序列长度不是实测值。

增加新功能前，先使用 AGENTS.md 中整理的官方文档、开源可视化项目、知乎/CSDN 和配色参考链接作为起点，再搜索更贴近需求的最新方案。记录链接、查阅日期、许可、差异和采用理由，再实现缺失部分。优先沿用仓库暖白配色及 G0–G4 数据分组语义，认真检查排版、对比度、中英切换和手机布局。不要复制第三方图片资产；不要把无法从源码证明的映射画成确定结果。

现有验证：14 项 Python 测试通过，app.js 与 trace.js 语法检查通过；桌面浏览器验证了任务联动、尾部 mask、Q/K/V、中英切换、播放/暂停、单步和源码定位。上一轮未完成真实窄屏视觉验收；涉及前端时请补做。macOS arm64 无 CUDA/NPU，请勿声称做过设备运行验证。

请先检查 `git status` 和最新提交，理解当前实现，再针对下面的新需求完成调研、设计、代码、测试和可视化验收。保留现有功能。完成后报告调研依据、实现、测试结果及剩余限制。

本次需求：[在这里填写]
```
