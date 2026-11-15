# 矩阵/表格化算子视图的前端 UI 调研（2026-11-09）

本轮目标：在 tile / GEMM / trace 三类矩阵视图上，统一“表格切分”的阅读方式，并借鉴成熟的前端表格与数据可视化实现来美化、协调界面。约束仍是仓库的暖白视觉、无框架/无构建步骤，以及“不伪造实测数据”。

## 检索方式与局限

- 查阅日期：2026-11-09。优先读取官方设计系统与可访问性规范，再参考数据表实践文章。
- 只借鉴信息组织、交互和视觉规则；没有复制任何第三方代码、图片或字体，也没有引入前端框架。
- 个别页面受登录、JS 壳或网络限制，仅记录为线索，未据此实现。

## 参考清单

| # | 来源 | 链接 | 可复用部分 | 在 OP/SHOW 的落地 / 差异 | 许可与抓取 |
| --- | --- | --- | --- | --- | --- |
| 1 | W3C WAI-ARIA APG · Grid Pattern | https://www.w3.org/WAI/ARIA/apg/patterns/grid/ | 数据网格的语义、roving tabindex（整张表一个 Tab 停靠点）、方向键/Home/End 导航、焦点管理 | tile 矩阵改为“整张表一个 tab stop + 方向键移动选中 tile”；保留 button 语义，不伪造完整 ARIA grid 行列结构 | 已抓取；W3C 文档许可；未复制示例代码 |
| 2 | W3C APG · Data Grid Examples | https://www.w3.org/WAI/ARIA/apg/patterns/grid/examples/data-grids/ | 数据网格的键盘交互细节和可访问性取舍 | 作为方向键/Home/End 行为的线索；OP/SHOW 是静态阅读器，不需要单元格编辑 | 仅记录入口（本轮未单独读取正文） |
| 3 | Nielsen Norman Group · Data Tables: Four Major User Tasks | https://www.nngroup.com/articles/data-tables/ | 表格适合“比较”，卡片/详情适合“深入阅读单条”；表格需要持续可见的表头/标识 | 保留 tile 矩阵做横向比较，把详细归属/地址放在右侧 Inspector；不把卡片堆成唯一视图 | 已抓取；仅参考文章观点 |
| 4 | Material Design · Data tables | https://m1.material.io/components/data-tables.html | 表头、行悬停、选中态、密度、数值/文本对齐；表格卡片化 | 统一矩阵单元 gap/半径/描边；为 GEMM 增加有效/masked 图例与悬停反馈；未复制 Material 组件或图标 | 已抓取；Material 1 已停止维护，仅取通用规则 |
| 5 | IBM Carbon Design System · Color palettes | https://www.carbondesignsystem.com/data-visualization/color-palettes/ | 分类色与顺序色的区别、颜色顺序需保证相邻可辨、颜色不能是唯一编码 | A/B/C 继续用分类色区分矩阵身份；valid/masked 用纹理 + 图例双编码，不依赖颜色；配色改为项目暖色 | 已抓取；IBM Carbon 许可；未使用其色值/资产 |
| 6 | IBM Carbon Design System · Chart anatomy | https://www.carbondesignsystem.com/data-visualization/chart-anatomy/ | 标题、轴标题、刻度、图例、图形 frame 的层级关系 | 矩阵统一加 axis label（行/列名与范围）、GEMM 图例、矩阵 frame；轴标签保持 8px mono，避免抢内容 | 已抓取 |
| 7 | Figr · Data Table Design: Layout, States and Accessibility（2026-10） | https://figr.design/blog/data-table-design | 上下文/标题、冻结标识、状态与可访问性、表格零件职责 | 增加矩阵方向说明、选中态、焦点环、粘性行列标签；不复制其具体布局 | 已抓取；社区文章，仅参考观点 |
| 8 | raw.studio · 5 Quick Checks to Fix Data Table UX（2026-09） | https://raw.studio/blog/data-table-ux | 标识列优先、粘性表头、语义结构、过滤可见 | trace 矩阵增加粘性列头/行头；tile 矩阵提供坐标轴和方向说明 | 已抓取 |
| 9 | Material Design 3 · Grids & spacing（密度） | https://m3.material.io/foundations/layout/grids-spacing/density | 密度与间距的系统化，而非每个视图单独调 | 用 --matrix-gap / --matrix-axis 等 token 统一 tile、GEMM、trace、inspector 的几何 | 仅记录入口（页面为 JS 壳，未完整读取） |

## 采纳的设计规则

1. **统一矩阵几何**：新增 matrix-frame、matrix-axis、--matrix-gap、--matrix-axis 等 token；tile 网格、Inspector 单元矩阵、GEMM A/B/C 分片、trace 矩阵共享同一套间距、圆角、轴标签字号和边框语义。
2. **先比较，再深入**：tile 矩阵保留为比较视图，列方向增加 N0/N1… 轴标签，grid-context 增加“M ↓ 行 / N → 列”方向说明；单 tile 的详细边界和地址仍放在 Inspector，避免把矩阵塞成信息卡片。
3. **键盘可达**：tile 矩阵采用 roving tabindex，整张表只有一个 Tab 停靠点；方向键移动选中 tile，Home/End 跳到行首/行尾，焦点随选中移动。与全局播放快捷键隔离，避免方向键同时推进播放。
4. **矩阵身份与状态分离**：A/B/C 的分类色只表示“这是哪个矩阵”，有效/masked 用实色/斜纹 + 图例表示；执行状态继续用明暗/描边，不用新颜色承载状态。
5. **轴与图例**：GEMM 分片加行/列轴标签（如 K[16:19]）和 有效/masked 图例；trace 矩阵的表头在纵向滚动时保持可见，行标签在横向滚动时保持可见。
6. **更安静的交互**：去掉 trace 单元 hover 的 scale 放大，改用描边/阴影，避免矩阵在鼠标移动时抖动；保留 reduced-motion 分支。
7. **密度与响应式**：桌面保持 2px 单元 gap、8px 轴标签；窄屏缩小 padding，tile 列轴与网格一起横向滚动；不引入虚拟滚动或框架。

## 明确不做的事

- 不生成“热力值/带宽/耗时”颜色，不把颜色映射成没有数据支撑的量。
- 不复制 Material/Carbon/Figr 的图片、色板或组件代码；不引入 UI 框架。
- 不为了 ARIA grid 的完整性而伪造行列 DOM；当前用 button + roving tabindex 提供等效键盘体验。
- 不做超出演示规模的虚拟化；极端 tile 数量下的性能仍未实测。

## 许可结论

- 本调研只使用公开文档中的**规则与交互模式**，未复制代码或资产；新增 CSS/HTML/JS 为本项目编写。
- 各来源版权归原作者；后续若要复用具体代码/图标/色板，需单独确认许可。
- 暖白配色沿用仓库既有的 #FBF7F1 / #FFFDF9 / #342A24 / #E7D8C8 / #B85437 体系，未照搬任何第三方主题。

## 第二轮：工作台层级与表格配色（2026-11-09）

本轮目标：把首屏从“介绍页”收敛成“工作台”，并统一矩阵/表格的排布、字号、间距和配色。查证日期均为 2026-11-09。

| 来源 | 链接 | 借鉴点 | 与 OP/SHOW 的差异 |
| --- | --- | --- | --- |
| GitHub Primer · Color usage | https://primer.style/product/getting-started/foundations/color-usage/ | 三层 token；accent / success / attention / danger 语义角色；背景分 default / muted / inset；边框与文字使用中性色阶 | 不引入 Primer 变量或主题，只把相同职责映射到本项目暖白 token |
| GitHub Primer · DataTable | https://primer.style/components/data-table | 文本默认左对齐、数字右对齐；用 cell padding 控制密度；行选择用稳定 ID；表头/行头语义 | 本项目矩阵是二维坐标，不是记录表，单元格保持居中；只借对齐与密度规则 |
| Ant Design · 色彩 | https://ant.design/docs/spec/colors-cn | 12 主色 × 10 层级；主色取第 6 阶；功能色保持一致；中性色承担文字/背景/边框；企业界面用色克制 | 保留暖白主色，不采用 Ant 蓝或组件库 |
| AntV · 色板规范 | https://antv.antgroup.com/zh/specification/language/palette/ | 分类色按色相区分且相邻明暗有别；顺序色同色相 5–7 阶；语义色遵循约定；颜色不宜过多；照顾色觉差异 | A/B/C 仍用现有暖色分类，不新增数据含义色 |
| IBM Carbon · Color palettes | https://www.carbondesignsystem.com/data-visualization/color-palettes/ | 分类色按固定顺序使用以拉开相邻差异；顺序色表达有序数据；警示色有固定语义 | 不照搬 Carbon 色值；masked 继续用斜纹而非仅换色 |
| CSDN / 知乎检索（2026-11-09） | https://so.csdn.net/so/search?q=%E6%95%B0%E6%8D%AE%E8%A1%A8%E6%A0%BC%20%E8%AE%BE%E8%AE%A1%20%E9%85%8D%E8%89%B2 · https://www.zhihu.com/search?type=content&q=B%E7%AB%AF%E8%A1%A8%E6%A0%BC%E8%AE%BE%E8%AE%A1 | 社区文章的共识：文本左对齐、数字右对齐、表头跟随列；浅色横线替代重网格；表头加重；大表才用斑马纹；留白即分隔；状态色语义统一 | 只采纳排布与配色规则，未复制文章内容或配图 |
| W3C APG · Grid | https://www.w3.org/WAI/ARIA/apg/patterns/grid/ | roving tabindex、方向键/Home/End、焦点跟随选中 | 继续用 button + roving tabindex，不伪造完整 grid 行列 DOM |

本轮采纳的规则：

1. **工具优先的页面层级**：删除营销式大标题，页面标题压到 24–27px；算子、形状、指标进入首屏；大字号只留给关键数值。
2. **中性色优先的 60-30-10**：暖白/米色承担大部分面积，accent 只用于主操作、选中与焦点；运行/完成状态改用柔和底色 + 文字 + 进度条，不再整块铺满主色。
3. **表格排布**：表格的数值列右对齐，文本列左对齐；本页指标条是摘要卡，标签与数值同侧左对齐；矩阵单元格因是二维坐标而居中；用浅横线和面板分组，不画重网格；小矩阵不加斑马纹。
4. **颜色规范**：surface / border / text / accent / success / warning 语义 token 分开；A/B/C 分类色保持色相与明度都不同；masked 用斜纹双编码；颜色不作为唯一状态编码。
5. **密度与节奏**：8px 节奏；面板 padding 16–20px；矩阵 gap 2px、轴标签 8px；指标条与时间线保持紧凑，避免首屏被说明文字占满。
6. **窄屏**：控制条换行或纵向排列，分段控件不得溢出卡片；触控目标不小于 24×24；时间线/阶段条只在自身容器内滚动。
7. **不引入**：前端框架、第三方字体/图标、构建步骤；新增样式均由本项目编写。

许可结论：本轮只参考公开文档和社区文章中的排布、层级与配色规则，没有复制代码、色板、图片或大段文案；所有改动仍使用仓库自有暖白体系。

## GEMM 依赖布局（2026-11-15）

用户提供的 GEMM 示意图把完整 A 放左下、B 放右上、C 放右下；A 的 M 行带与 B 的 N 列带在当前 K 分片相交，指向 C 中同一输出 tile。现有 OP/SHOW 只把三个 16×16 分片平铺，缺少完整矩阵坐标，难以看出输出位置依赖哪两条输入带。

| 来源 | 可借鉴部分 | 取舍与许可 |
| --- | --- | --- |
| 用户本轮提供的 GEMM 示意图 | A/B/C 的空间朝向、M 行带 × N 列带 → C tile、K 循环沿 A/B 的方向 | 只借关系，页面重新绘制；不复制图片资产或原图配色。 |
| [Triton 官方 Programming Guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)（2026-11-15 可访问） | program ID 与输出 tile 对应、循环中的输入分片 | 用于核对逻辑关系；不复制图像。 |
| [CUDA Tiled GEMM Visualizer](https://www.joelfernandes.org/simulators/cuda-tiled-gemm/index.html)（2026-11-15 可访问） | 当前 A/B 分片和 C 输出联动 | 已有动画偏内置 GEMM；本项目用源码模型驱动，重画概览与分片。 |

实施时保留当前逐格 A/B/C 分片作为 mask 细节，在其上增加完整 A[M,K]、B[K,N]、C[M,N] 的依赖概览。概览只画真实维度与选中范围，不假装显示设备中的 fragment；当前 K 分片在 A/B 中移动，C tile 位置保持固定。两层都用既有暖白和 A/B/C 分类色。
