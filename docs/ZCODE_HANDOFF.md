# OP/SHOW 浏览器验收交接提示词（给 Zcode）

你将接手一个本地、无 GPU 依赖的算子可视化工具 **OP/SHOW**，用 computer use 做真实浏览器验收和“怎么用”的交接。请直接操作界面，不要只读代码。

## 一、仓库与当前状态

- 仓库：`/Users/tmchao7/repos/02_personal/op-show`
- 先用 `git status` 查看当前 HEAD 和工作区；验收期间不要 commit、push、reset 或清理工作区。
- 内置示例为 7 个 v1：Vector Add、Reduce、GEMV、三个 GEMM 和 FlashAttention-2 风格前向；另有 Triton-Ascend QKV All-to-All v2 演示及附许可证的上游源码。`serve examples` 菜单应显示 8 个模型。
- 先读：`README.md`、`AGENTS.md`、`docs/research-ui-matrix.md`、`docs/research-visualization.md`。
- **重要**：`serve` 在启动时把 `index.html`、`app.js`、`styles.css` 读入内存。若你或用户改了前端文件，必须重启服务才会生效。

## 二、启动与基础检查

```sh
cd /Users/tmchao7/repos/02_personal/op-show
python3 -m unittest discover -s tests -v
node --check op_show/web/app.js
node --check op_show/web/trace.js
python3 -m op_show serve examples --no-browser --port 8765
```

打开 `http://127.0.0.1:8765/`。若端口被占，换 `8766`；不要杀不属于你的进程。验收结束后关闭你启动的服务。

预期：32 个 Python 测试通过，缺少 PyTorch/Triton/NVIDIA GPU 时设备数值对照测试跳过；两个 JS 语法检查通过，浏览器无 404/JS 报错。以当前测试输出为准。

## 三、桌面验收清单（建议 1440×900 或 1440×1000）

1. **Tile 矩阵**：默认 Tensor Core 模型下，确认 tile 网格上方有 `N0/N1…` 列轴标签和“M ↓ 行 / N → 列”方向说明；点击四个 tile，右侧 Inspector 的编号、边界、地址应联动更新。
2. **键盘导航**：用 Tab 进入 tile 网格（整张表只应有一个 Tab 停靠点）；按方向键移动选中 tile，Home/End 跳到行首/行尾；焦点环要清晰可见；按方向键时播放进度不应前进。
3. **播放与时间线**：播放/暂停、单步、拖动进度条、点击时间线色块，确认 tile 状态、当前阶段、时间线和 Inspector 同步；确认界面文案仍明确“逻辑进度，非实测耗时”。
4. **源码引用**：点击源码面板中带圆点标记的源码行，确认出现 “USED BY / 引用位置” chips，并能从引用列表跳到对应行。
5. **分享视图**：点击“分享视图”，确认复制链接或地址栏 hash 包含模型/场景/tile/逻辑 tick/语言；在新标签页打开该链接，确认恢复同一视图；切换中文/EN 后分享，确认语言也被恢复。
6. **GEMM 面板**（Tensor Core 模型）：向下滚动到 A × B → C 面板，确认 A/B/C 都有行列轴标签（如 `M[0:16)`、`K[16:19)`）、“有效 / 被 mask”图例、有效格与斜纹 masked 格；hover 时不应出现布局跳动或横向溢出。
7. **所有模型**：用顶部“算子”下拉切换 8 个模型，并在“输入形状”中切换 aligned / padded 场景；确认没有空白面板、重叠文字、异常滚动条或 console error。v2 还要检查 Producer/Consumer 四视图、Wave、Q/K/V 切换。
8. **语言**：中文/EN 切换后，静态文案、矩阵轴说明、GEMM 图例、源码面板、分享链接都应按当前语言显示，不能出现明显漏翻。
9. **视觉协调**：重点看矩阵的间距、轴标签对齐、图例、选中/焦点态、暖白配色对比度；确认新矩阵样式与旧面板没有“两套设计语言”。

## 四、移动/窄屏验收

用浏览器 DevTools 或窗口缩放到 **390×844** 左右，检查：

- `document.documentElement.scrollWidth === window.innerWidth`（不应有页面级横向溢出）；
- 顶部栏、算子/形状下拉、CUDA C++ 标签不被截断；
- tile 矩阵的列轴与卡片对齐，卡片不跑出面板；
- GEMM A/B/C 在窄屏下不溢出面板，轴标签可读；
- 时间线/阶段条等横向滚动容器只在自身内部滚动，不把整页撑宽；
- 键盘焦点和触控目标仍可用。

此前已用 headless Chrome 在 1440 和 390 宽度做过截图与方向键测试，但仍需你用真实 computer use 复核，尤其是浏览器差异、字体、触控板/鼠标 hover 和实际视觉观感。

## 五、可访问性与约束

- 检查焦点可见、按钮/单元格可读、图例不只用颜色区分状态（masked 有斜纹）；如方便，用屏幕阅读器或系统辅助功能抽查关键控件。
- 不要求你修改代码；发现问题的第一优先级是**准确复现并报告**（截图、viewer 宽度/浏览器版本、操作步骤、期望/实际）。
- 不要运行 GPU/NPU 代码，不安装前端框架，不修改 `examples/sources/`，不 commit/push，不删除/重置工作区。
- 若用户明确让你修复：先读 `AGENTS.md`，保持暖白配色、无构建步骤、中英双语；改完按仓库检查项重新跑测试和语法检查。

## 六、最终交接报告格式

请输出：

1. 环境：OS、浏览器/版本、窗口/模拟尺寸、服务 URL、启动命令；
2. 检查表：逐项 PASS / FAIL，FAIL 附最小复现步骤和截图路径；
3. 控制台/网络：是否有 JS error、404、布局警告；
4. 视觉问题：按严重程度列出（阻断/明显/轻微），说明是否影响“使用交接”；
5. 结论：可以直接交给用户使用，还是需要先修复哪几项；
6. 关闭你启动的服务，并说明是否有未提交改动/未覆盖项。

重点验收不是“代码看起来对”，而是用户能否在真实浏览器里顺畅完成：看 tile → 选 tile → 播放/单步 → 看 GEMM 分片 → 点源码证据 → 分享/恢复视图。
