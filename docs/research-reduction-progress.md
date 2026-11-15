# 结果 Tile 归约进度色阶调研

查阅日期：2026-11-15。需求是让同一个输出区域随 K/Key 循环逐轮加深，同时保留源码阶段、尾部 mask 和写回状态。

| 来源 | 借鉴点 | 与 OP/SHOW 的差异及使用边界 |
| --- | --- | --- |
| [Carbon 顺序色规范](https://www.carbondesignsystem.com/building-blocks/data-visualization/color-palettes) | 浅色主题里用同色系由浅到深表示有序量；分类色和顺序色分工。 | 本项目用含零值在内的七档暖色表达有效归约位置比例，不复制 Carbon 色值、图片或代码。 |
| [Carbon 进度条规范](https://www.carbondesignsystem.com/building-blocks/core/components/progress-bar/guidelines) | 有明确分母时同时给出进度标签和数值。 | 本项目输出精确的 `16/19`，而不是设备执行百分比；只借交互原则。 |
| [W3C WCAG 2.2：颜色使用](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html) | 颜色不能是传递信息的唯一方式。 | 色阶旁保留文字分数；mask 仍用斜纹，选中和完成仍用独立描边与文字。 |
| [MDN color-mix()](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/color_value/color-mix)、[reduced motion](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/%40media/prefers-reduced-motion) | 浏览器原生混色与减少动画偏好。 | 沿用项目已有的 CSS 能力，不增加前端依赖；不复制文档示例代码。 |
| [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)、[融合 Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)、[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/efficient_gemm.html) | 核对 K 分片逐轮累加与 Attention 在线状态更新的语义。 | 具体检查点仍引用仓库示例源码；官方文档不替代本地源码证据。 |
| [VisualMatmul](https://github.com/gmlayer0/VisualMatmul) | 已有逐步展示矩阵乘累加的思路。 | PyQt/OpenGL 架构不适合直接嵌入本地网页；未核实许可证，因此不复制任何实现或资产。 |

色阶表示“已纳入计算的有效归约位置 ÷ 总有效位置”。它不表示累加结果大小、输出已经写回、真实硬件调度或耗时。单轮循环也需要一个贡献检查点；即使分数达到满值，后续的规约、归一化与 store 仍是独立阶段。
