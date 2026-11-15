const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[character]));

const translations = {
  zh: {
    topCaption: '无需硬件的算子图解', local: '本地', eyebrow: '源码 → TILE 映射 → 执行',
    workspaceTitle: 'Tile 执行图解',
    intro: '从 program 或 block ID 找到它负责的数据，再逐步查看哪些 tile 可以并行、哪些阶段必须依次执行。',
    logical: '逻辑执行进度', logicalNote: '无依赖的 tile 同时推进。时间轴展示依赖关系，不代表设备耗时。',
    settings: '可视化设置', operator: '算子', inputShape: '输入形状', statistics: '模型统计',
    detailTitle: '当前任务', detailTabs: '详情视图', detailSummary: '摘要', detailAddresses: '地址', detailMask: '逐格', detailTasks: '任务', detailSource: '源码',
    sectionTiles: '01 / TILE 归属', tileGrid: '输出 tile 网格', outputTiles: '输出 tile',
    ownershipSection: '00 / LAUNCH · PID', ownershipTitle: '先看谁认领任务', ownershipNote: '同一 PID 内的任务按源码循环顺序执行；不同 PID 仅在没有声明依赖时可并行。这里不表示实际设备调度或耗时。',
    launchGrid: 'Launch 网格', pidCount: '个 PID', taskCount: '个输出任务', claimsPerPid: '每 PID 认领', claimUnit: '认领单位', claimOrder: '源码循环顺序', claimReady: '可并行启动的 PID', claimSelected: '当前选中', claimSource: '查看认领源码', groupTasks: '{count} 个输出任务', groupPids: '{count} 个 PID 参与',
    pending: '待执行', active: '进行中', complete: '已完成', tailMasked: '尾部 / mask',
    sectionExecution: '02 / 理想化执行', parallelTitle: '无依赖的 tile 同时推进',
    reset: '重置', play: '播放', pause: '暂停', stepForward: '单步前进', executionProgress: '执行进度',
    speed: '速度', stageSequence: '阶段顺序', serialWithinTile: 'tile 内依次执行',
    progressNote: '进度按已完成的 tile 阶段计算，不代表实际耗时或 GPU/NPU 利用率。',
    sectionInspector: '03 / 详情', sectionSource: '04 / 源码依据', sourceTitle: '映射依据来自哪里',
    footerWalkthrough: '基于源码的逻辑图解', footerNoExecution: '页面不会执行算子源码',
    metricShape: '输入形状', metricTileSize: 'TILE 大小', metricOutputTiles: '输出 TILE',
    metricMasked: 'MASK 单元', elements: '个元素', workItems: '个逻辑工作项',
    tailPadding: '尾 tile 填充', fullyAligned: '完全整除', tiles: '个 tile',
    workers: '个执行单元', validCells: '个有效单元', axisRows: '行', axisCols: '列', gridParallel: '{count} 个 tile 可并行启动',
    gridHint: '点击 tile 查看归属与地址偏移 ↗', waiting: '等待依赖', stageComplete: '已完成',
    seekTo: '跳转到 {label}', stepNumber: '第 {number} 步',
    tooManyCells: 'tile 太大，无法逐格展示；上方列出了范围与数量。',
    validCell: '有效', maskedCell: '被 mask', logicalTile: '逻辑 TILE', tailTile: '尾 TILE',
    ownership: '归属与边界', worker: '执行单元', output: '输出范围', valid: '有效单元',
    dependencies: '依赖', current: '当前阶段', noDependencies: '无 · 可并行',
    validAndMask: '有效单元 / MASK', cellsLegend: '实色单元在边界内；斜纹单元被 mask。',
    maskedTail: '{count} 个被 mask 的尾部单元不会写入输出。', noPadding: '此 tile 没有输出填充。',
    loopSlice: '{axis} 分片 {number}：{valid}/{total} 个有效元素。',
    addressOffsets: '首元素地址偏移', offset: '偏移', viewSource: '查看源码行 ↗',
    noAddress: '未提供地址示例。', sourceEvidence: '依据，不是执行结果',
    sourceExplanation: '这些代码行与本地源码逐字对应。公式由 agent 提出，校验器检查描述与覆盖范围。使用前请对照源码复核公式。',
    mappingUnknown: '映射未知：{note}。已禁用播放。', loadError: '无法加载描述或源码',
    tracePrev: '上一步', traceNext: '下一步', traceProgress: '逻辑进度',
    share: '分享当前视图', shareLabel: '分享视图', shareCopied: '已复制', sharePrompt: '复制此链接',
    sectionTimeline: '02b / 时间线', timelineTitle: '全局长条时间线',
    timelineHint: '每一行是一个 tile，横轴是逻辑 tick。同一列同时亮起的 tile 同时推进；点击色块跳到对应阶段。',
    timelineNote: '色块只表示逻辑阶段占位，不是设备时间、吞吐或性能测量。',
    sourceUsage: '引用位置', sourceUsageEmpty: '没有映射字段引用这一行',
    catalogError: '无法加载算子列表', pageTitle: 'OP/SHOW — 算子 Tile 执行图解',
    traceHeading: '01 / 数据如何流动', traceTitle: '从 Producer Tile 到 Consumer 输出', sourceRank: '来源 Rank', targetRank: '目标 Rank', tensor: '输出张量', producerMatrix: 'Producer 数据矩阵', claimTable: 'PID 认领表', peerPath: 'peer_mem 数据路径', consumerMatrix: 'Consumer 输出矩阵', producerHint: '行是 block，列是目标 Rank 与 head group；颜色区分 G0–G4。', claimHint: '每行一个 Producer PID；每列是本 Wave 中按步长 16 认领的任务。', consumerHint: '行是来源 Rank 与 block，列是 head group；Q/K/V 使用相同坐标。', logicalEvents: '逻辑事件与依赖', traceTiming: '每一步同时完成所有依赖已满足的事件；Wave 不是跨 PID 的全局屏障，也不代表设备耗时。',
    gemmTitle: 'C tile 依赖哪些 A / B 数据', gemmHint: '选中上方 C tile；沿时间轴查看 A 的行带与 B 的列带如何按 K 分片相乘并累加。矩阵面积表示逻辑坐标范围，不代表硬件耗时。', gemmMaskLegend: 'A/B 的斜纹格：越界后填零；C 的斜纹格：不写入输出。', gemmPhase: 'K 分片 {current}/{total}', gemmValid: '{count}/{total} 个有效元素',
    gemmDetailTitle: '当前分片 · 逐格检查', gemmDetailHint: '下方显示实际 tile 尺寸及边界 mask', gemmMapHint: '选中的 A 行带 × B 列带 → C tile；深色交点是当前 K 分片', gemmMapTail: '尾 tile：斜纹元素见下方逐格图',
    rowkReduceTitle: '按行归约的数据分片', rowkGemvTitle: '矩阵向量乘的数据分片', rowkHint: '选择输出 tile，再沿时间轴查看当前 K 分片、有效输入和输出行。每个 warp 负责一行；格子是逻辑数据位置。', rowkMaskLegend: '斜纹输入格被边界条件跳过；斜纹输出行不会写回。',
    attentionTitle: '当前 Attention 数据分片', attentionHint: '选择输出 tile，再沿时间轴查看 Q 行块如何依次读取 K/V 分片、更新在线 softmax 状态并写出 O。', attentionMaskLegend: '斜纹 Q/O 行属于尾块；K/V 的斜纹行超出 N，S 的对应列被设为 −∞。补齐的 Q 行不会写回 O。', attentionPhase: 'Key 分片 {current}/{total}', attentionScores: 'S 是逻辑 QKᵀ 分数矩阵，不是全局内存张量。', attentionRecurrence: '在线 softmax：m′ = max(m, rowmax(S))；α = exp(m − m′)；l′ = αl + rowsum(exp(S − m′))；acc′ = αacc + P × V；O = acc / l。'
  },
  en: {
    topCaption: 'HARDWARE-FREE KERNEL WALKTHROUGH', local: 'LOCAL',
    eyebrow: 'SOURCE → TILE MAP → EXECUTION',
    workspaceTitle: 'Tile execution walkthrough',
    intro: 'Follow each tile from its program or block ID to the data it owns. Step through the work that happens in parallel and the steps that must happen in order.',
    logical: 'Logical execution', logicalNote: 'All independent tiles advance together. The timeline shows dependencies, not device timing.',
    settings: 'Visualization settings', operator: 'OPERATOR', inputShape: 'INPUT SHAPE', statistics: 'Model statistics',
    detailTitle: 'Selected task', detailTabs: 'Detail views', detailSummary: 'Summary', detailAddresses: 'Offsets', detailMask: 'Cells', detailTasks: 'Tasks', detailSource: 'Source',
    sectionTiles: '01 / TILE OWNERSHIP', tileGrid: 'Output tile grid', outputTiles: 'Output tiles',
    ownershipSection: '00 / LAUNCH · PID', ownershipTitle: 'Who claims each task', ownershipNote: 'Claims within a PID follow source loop order. Different PIDs can advance together only when no dependency is declared. This is not device scheduling or elapsed time.',
    launchGrid: 'Launch grid', pidCount: 'PIDs', taskCount: 'output tasks', claimsPerPid: 'Claims per PID', claimUnit: 'Claim unit', claimOrder: 'Source loop order', claimReady: 'PIDs ready in parallel', claimSelected: 'Selected', claimSource: 'View claim source', groupTasks: '{count} output tasks', groupPids: '{count} PIDs involved',
    pending: 'Pending', active: 'Active', complete: 'Complete', tailMasked: 'Tail / masked',
    sectionExecution: '02 / IDEAL EXECUTION', parallelTitle: 'All independent tiles advance together',
    reset: 'Reset', play: 'Play', pause: 'Pause', stepForward: 'Step forward', executionProgress: 'Execution progress',
    speed: 'SPEED', stageSequence: 'STAGE SEQUENCE', serialWithinTile: 'within each tile · serial',
    progressNote: 'Progress is measured in completed tile stages. It is not elapsed time or measured GPU/NPU utilization.',
    sectionInspector: '03 / INSPECTOR', sectionSource: '04 / SOURCE EVIDENCE', sourceTitle: 'Where this mapping came from',
    footerWalkthrough: 'Source-backed logical walkthrough', footerNoExecution: 'No code is executed in this viewer',
    metricShape: 'INPUT SHAPE', metricTileSize: 'TILE SIZE', metricOutputTiles: 'OUTPUT TILES',
    metricMasked: 'MASKED CELLS', elements: 'elements', workItems: 'logical work items',
    tailPadding: 'tail tile padding', fullyAligned: 'fully aligned', tiles: 'tiles',
    workers: 'workers', validCells: 'valid cells', axisRows: 'rows', axisCols: 'columns', gridParallel: '{count} tiles can start in parallel',
    gridHint: 'Click a tile to inspect ownership and offsets ↗', waiting: 'Waiting for dependencies',
    stageComplete: 'Complete', seekTo: 'Seek to {label}', stepNumber: 'STEP {number}',
    tooManyCells: 'Tile too large for individual cells. Bounds and counts are shown above.',
    validCell: 'Valid', maskedCell: 'Masked', logicalTile: 'LOGICAL TILE', tailTile: 'TAIL TILE',
    ownership: 'OWNERSHIP & BOUNDS', worker: 'WORKER', output: 'OUTPUT', valid: 'VALID',
    dependencies: 'DEPENDENCIES', current: 'CURRENT', noDependencies: 'None · can run in parallel',
    validAndMask: 'VALID CELLS / MASK', cellsLegend: 'Solid cells = in bounds; striped cells = masked.',
    maskedTail: '{count} masked tail cells do not write output.', noPadding: 'No output padding in this tile.',
    loopSlice: '{axis} slice {number}: {valid}/{total} valid elements.',
    addressOffsets: 'ADDRESS OFFSETS · FIRST ELEMENT', offset: 'offset', viewSource: 'View source line ↗',
    noAddress: 'No address sample provided.', sourceEvidence: 'Evidence, not execution',
    sourceExplanation: 'These lines are literal matches to the local source. The agent proposes the formulas; the validator checks the model and coverage. Review the formulas against the source before relying on them.',
    mappingUnknown: 'Mapping unknown: {note}. Playback is disabled.', loadError: 'Could not load model or source',
    tracePrev: 'Previous step', traceNext: 'Next step', traceProgress: 'Logical progress',
    share: 'Share this view', shareLabel: 'Share view', shareCopied: 'Copied', sharePrompt: 'Copy this link',
    sectionTimeline: '02b / TIMELINE', timelineTitle: 'Global stage timeline',
    timelineHint: 'One lane per tile; the horizontal axis is logical ticks. Tiles lit up in the same column advance together; click a segment to jump to that stage.',
    timelineNote: 'Segments are logical stage slots, not device time, throughput, or a performance measurement.',
    sourceUsage: 'USED BY', sourceUsageEmpty: 'No mapping fields cite this line',
    catalogError: 'Could not load catalog', pageTitle: 'OP/SHOW — Tile execution explorer',
    traceHeading: '01 / DATA FLOW', traceTitle: 'Producer tiles to Consumer output', sourceRank: 'SOURCE RANK', targetRank: 'TARGET RANK', tensor: 'OUTPUT TENSOR', producerMatrix: 'Producer data matrix', claimTable: 'PID claim table', peerPath: 'peer_mem data path', consumerMatrix: 'Consumer output matrix', producerHint: 'Rows are blocks; columns are target ranks and head groups. G0–G4 colors identify data groups.', claimHint: 'Each row is one Producer PID; columns show stride-16 claims in this wave.', consumerHint: 'Rows are source ranks and blocks; columns are head groups. Q/K/V share coordinates.', logicalEvents: 'Logical events and dependencies', traceTiming: 'Each step completes all events with satisfied dependencies. A wave is not a global PID barrier or device time.',
    gemmTitle: 'Which A / B data feeds this C tile', gemmHint: 'Select a C tile above. Step through K slices to see an A row band and B column band multiply and accumulate. Matrix area shows logical coordinates, not hardware time.', gemmMaskLegend: 'Striped A/B cells are zero-filled; striped C cells are not stored.', gemmPhase: 'K slice {current}/{total}', gemmValid: '{count}/{total} valid elements',
    gemmDetailTitle: 'Current slices · cell inspection', gemmDetailHint: 'Actual tile size and boundary masks appear below', gemmMapHint: 'Selected A row band × B column band → C tile; dark intersection is the current K slice', gemmMapTail: 'Tail tile: see striped cells in the detail below',
    rowkReduceTitle: 'Row reduction data slices', rowkGemvTitle: 'Matrix-vector data slices', rowkHint: 'Select an output tile, then follow the timeline through each K slice, valid input, and output row. Each warp owns one row; cells show logical data positions.', rowkMaskLegend: 'Striped input cells are skipped by bounds checks; striped output rows are not stored.',
    attentionTitle: 'Current Attention data slices', attentionHint: 'Select an output tile, then follow each K/V slice as the Q rows update online softmax state and write O.', attentionMaskLegend: 'Striped Q/O rows are tail rows; striped K/V rows exceed N and their S columns become −∞. Padded Q rows are not stored to O.', attentionPhase: 'Key slice {current}/{total}', attentionScores: 'S is the logical QKᵀ score matrix, not a global-memory tensor.', attentionRecurrence: 'Online softmax: m′ = max(m, rowmax(S)); α = exp(m − m′); l′ = αl + rowsum(exp(S − m′)); acc′ = αacc + P × V; O = acc / l.'
  }
};

function savedLanguage() {
  try { return localStorage.getItem('op-show-language') === 'en' ? 'en' : 'zh'; }
  catch { return 'zh'; }
}
function t(key, values = {}) {
  return translations[state.language][key].replace(/\{(\w+)\}/g, (_, name) => values[name] ?? '');
}
function displayLabel(item, field = 'label') {
  return state.language === 'zh' && item?.[`${field}_zh`] ? item[`${field}_zh`] : item?.[field] || '';
}
function applyLanguage() {
  document.documentElement.lang = state.language === 'zh' ? 'zh-CN' : 'en';
  document.title = t('pageTitle');
  document.querySelectorAll('[data-i18n]').forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll('[data-i18n-title]').forEach((node) => { node.title = t(node.dataset.i18nTitle); });
  document.querySelectorAll('[data-i18n-aria-label]').forEach((node) => {
    node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel));
  });
  document.querySelectorAll('[data-lang]').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.lang === state.language));
  });
}

const state = { catalog: [], model: null, modelIndex: null, source: '', scenario: null, selected: null,
  tick: 0, timer: null, speed: 650, highlightedRef: null, language: savedLanguage(),
  timelineKey: null, timelineSegments: [], refIndex: null, shareTimer: null, detailTab: 'summary' };

function setupWorkbench() {
  const rail = $('detail-rail-content');
  rail.append(document.querySelector('.inspector-panel'), document.querySelector('.trace-detail'), $('source-panel'));
  const gemmMask = document.createElement('div');
  gemmMask.id = 'gemm-mask-detail';
  gemmMask.append(document.querySelector('.gemm-detail-heading'), $('gemm-matrices'),
    document.querySelector('.gemm-panel > .playback-footnote'));
  rail.append(gemmMask);
  $('trace-task-detail').append(document.querySelector('[data-trace-view="claim"]'), document.querySelector('[data-trace-view="path"]'));
  $('timeline-drawer').append($('timeline-panel'));
}
setupWorkbench();

function openDetailTab(name) {
  const allowed = ['summary', 'addresses', 'mask', 'source', 'tasks'];
  if (!allowed.includes(name) || (name === 'tasks' && !isTraceModel())) name = 'summary';
  state.detailTab = name;
  $('detail-rail').dataset.activeTab = name;
  document.querySelectorAll('[data-detail-tab]').forEach((button) => {
    button.setAttribute('aria-selected', String(button.dataset.detailTab === name));
    button.tabIndex = button.dataset.detailTab === name ? 0 : -1;
  });
  $('detail-rail-content').scrollTop = 0;
  if (name === 'source' && state.model) {
    delete $('source-code').dataset.lastLine;
    renderSource();
  }
}

function jumpToSource(ref) {
  state.highlightedRef = ref;
  openDetailTab('source');
}

function isTraceModel() {
  return state.model?.schema_version === 2 && state.model.status !== 'unknown';
}

// Map every source line to the mapping fields that cite it, so the source panel
// can explain why a line matters and let readers jump from code to the model.
function buildRefIndex(model) {
  const index = { usages: {}, lines: {} };
  const add = (ref, label) => {
    if (!ref || !model.refs || !model.refs[ref]) return;
    const usages = index.usages[ref] || (index.usages[ref] = []);
    if (!usages.includes(label)) usages.push(label);
    const line = model.refs[ref].line;
    const refs = index.lines[line] || (index.lines[line] = []);
    if (!refs.includes(ref)) refs.push(ref);
  };
  if (model.schema_version === 2) {
    const scenario = model.scenarios[0];
    for (const [key, ref] of Object.entries(scenario?.formula_refs || {})) add(ref, `formula ${key}`);
    for (const [key, ref] of Object.entries(model.param_refs || {})) add(ref, `param ${key}`);
    const kinds = new Set();
    for (const event of scenario?.events || []) {
      const label = `event ${event.kind}`;
      if (kinds.has(label)) continue;
      kinds.add(label);
      add(event.source_ref, label);
    }
  } else {
    add(model.grid_source_ref, 'launch grid');
    const scenario = model.scenarios[0];
    for (const ref of scenario?.claim_source_refs || []) add(ref, 'PID claim loop');
    for (const ref of model.parallelism?.source_refs || []) add(ref, 'parallel strategy');
    add(scenario?.task_group?.source_ref, 'task input group');
    for (const dimension of scenario?.dimensions || []) {
      add(dimension.source_ref, `axis ${dimension.name}`);
      add(dimension.size_ref, `size ${dimension.name}`);
    }
    for (const step of scenario?.steps || []) {
      add(step.source_ref, `step ${step.id}`);
      add(step.loop_source_ref, `loop ${step.loop_axis || ''}`.trim());
    }
    for (const address of scenario?.tiles?.[0]?.addresses || []) {
      add(address.source_ref, `address ${displayLabel(address)}`);
    }
    const gemm = scenario?.gemm_view;
    if (gemm) {
      for (const [key, ref] of Object.entries(gemm.source_refs || {})) add(ref, `gemm ${key}`);
    }
    const rowk = scenario?.row_k_view;
    if (rowk) {
      for (const [key, ref] of Object.entries(rowk.source_refs || {})) add(ref, `row/K ${key}`);
    }
    const attention = scenario?.attention_view;
    if (attention) {
      for (const [key, ref] of Object.entries(attention.source_refs || {})) add(ref, `attention ${key}`);
    }
  }
  for (const key of Object.keys(model.refs || {})) {
    if (!index.usages[key]) add(key, `ref ${key}`);
  }
  return index;
}

const STAGE_COLOR_COUNT = 6;

function renderTimeline() {
  const panel = $('timeline-panel');
  const scenario = currentScenario();
  if (!scenario || state.model.schema_version === 2) { panel.hidden = true; return; }
  panel.hidden = false;
  const total = Math.max(1, scenario.total_ticks);
  const key = [state.model.kernel, scenario.id, total, scenario.tiles.length, scenario.steps.length, state.language].join('|');
  const host = $('timeline');
  if (state.timelineKey !== key) {
    state.timelineKey = key;
    const rulerStep = Math.max(1, Math.ceil(total / 10));
    const ruler = [];
    for (let tick = 0; tick <= total; tick += rulerStep) {
      ruler.push(`<span style="left:${tick / total * 100}%">${tick}</span>`);
    }
    const stageIds = [...new Set(scenario.steps.map((step) => step.id))];
    const stageColor = new Map(stageIds.map((id, index) => [id, index % STAGE_COLOR_COUNT]));
    const rows = [`<div class="timeline-row timeline-ruler"><span class="timeline-name"></span><div class="timeline-track">${ruler.join('')}</div></div>`];
    for (const tile of scenario.tiles) {
      const tileId = tileKey(tile);
      const segments = scenario.steps.map((step, index) => {
        const start = tile.start + index;
        const label = `${tileLabel(tile)} · ${displayLabel(step)} · tick ${start}–${start + 1}`;
        return `<button type="button" class="timeline-seg stage-c${stageColor.get(step.id)}${step.iteration === null ? '' : ' loop'}" data-timeline-tile="${escapeHtml(tileId)}" data-timeline-start="${start}" style="left:${start / total * 100}%;width:${100 / total}%" title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}"></button>`;
      });
      const wait = tile.start > 0 ? `<span class="timeline-wait" style="width:${tile.start / total * 100}%"></span>` : '';
      rows.push(`<div class="timeline-row timeline-lane" data-timeline-lane="${escapeHtml(tileId)}"><button type="button" class="timeline-name" data-timeline-tile="${escapeHtml(tileId)}" data-timeline-start="${tile.start}">${escapeHtml(tileLabel(tile))}</button><div class="timeline-track">${wait}${segments.join('')}</div></div>`);
    }
    host.innerHTML = rows.join('');
    state.timelineSegments = host.querySelectorAll ? [...host.querySelectorAll('[data-timeline-start]')] : [];
  }
  $('timeline-tick').textContent = `${state.tick} / ${scenario.total_ticks}`;
  for (const node of state.timelineSegments) {
    const start = Number(node.dataset.timelineStart);
    node.dataset.state = state.tick > start ? 'done' : state.tick === start ? 'current' : 'pending';
  }
  const lanes = host.querySelectorAll ? host.querySelectorAll('[data-timeline-lane]') : [];
  for (const lane of lanes) lane.classList.toggle('selected', lane.dataset.timelineLane === state.selected);
}

function parseHash() {
  const raw = String(window.location?.hash || '').replace(/^#/, '');
  if (!raw) return null;
  const params = new URLSearchParams(raw);
  return {
    model: params.get('m'), scenario: params.get('s'), tile: params.get('tile'),
    task: params.get('task'), tick: params.get('tick'), language: params.get('lang'),
    detail: params.get('detail'),
  };
}

function buildShareHash() {
  const params = new URLSearchParams();
  if (state.modelIndex !== null) params.set('m', String(state.modelIndex));
  if (state.scenario) params.set('s', state.scenario);
  if (state.language) params.set('lang', state.language);
  if (state.detailTab !== 'summary') params.set('detail', state.detailTab);
  if (isTraceModel()) {
    const trace = TraceView.getState();
    if (trace.selected) params.set('task', trace.selected);
    params.set('tick', String(trace.tick));
  } else if (state.model) {
    if (state.selected) params.set('tile', state.selected);
    params.set('tick', String(state.tick));
  }
  return '#' + params.toString();
}

function shareUrl() {
  const origin = window.location?.origin || '';
  const path = window.location?.pathname || '';
  return `${origin}${path}${buildShareHash()}`;
}

function flashShare(button, label) {
  const labelNode = button.querySelector ? button.querySelector('span') : null;
  if (!labelNode) return;
  labelNode.textContent = label;
  button.classList.add('copied');
  clearTimeout(state.shareTimer);
  state.shareTimer = setTimeout(() => {
    labelNode.textContent = t('shareLabel');
    button.classList.remove('copied');
  }, 1600);
}

function copyShareLink() {
  const url = shareUrl();
  try { window.history?.replaceState(null, '', buildShareHash()); } catch (error) { /* ignore */ }
  const done = () => flashShare($('share-button'), t('shareCopied'));
  const fallback = () => {
    try { window.prompt?.(t('sharePrompt'), url); } catch (error) { /* ignore */ }
    done();
  };
  if (window.navigator?.clipboard?.writeText) {
    window.navigator.clipboard.writeText(url).then(done, fallback);
  } else {
    fallback();
  }
}

function applyHashState(hash) {
  if (!hash) return;
  if ((hash.language === 'zh' || hash.language === 'en') && hash.language !== state.language) {
    setLanguage(hash.language);
  }
  if (hash.detail) openDetailTab(hash.detail);
  const scenario = state.model?.scenarios.find((item) => item.id === hash.scenario);
  if (scenario && scenario.id !== state.scenario) {
    state.scenario = scenario.id;
    state.selected = isTraceModel() ? null : tileKey(scenario.tiles[0]);
    state.tick = 0;
    if (isTraceModel()) TraceView.scenarioChange(scenario);
    render();
  }
  const tick = Number.parseInt(hash.tick, 10);
  if (isTraceModel()) {
    state.highlightedRef = null;
    TraceView.setState({
      selected: hash.task || undefined,
      tick: Number.isFinite(tick) ? tick : undefined,
    });
    renderSource();
    return;
  }
  const current = currentScenario();
  if (hash.tile && current?.tiles.some((tile) => tileKey(tile) === hash.tile)) state.selected = hash.tile;
  if (Number.isFinite(tick) && current) state.tick = Math.max(0, Math.min(current.total_ticks, tick));
  render();
}

function currentScenario() {
  return state.model?.scenarios.find((item) => item.id === state.scenario);
}

function tileKey(tile) { return tile.coords.join(','); }
function tileLabel(tile) { return `T[${tile.coords.join(', ')}]`; }
function backendLabel(backend) {
  return { cuda: 'CUDA C++', 'triton-cuda': 'TRITON / CUDA',
    'triton-ascend': 'TRITON / ASCEND' }[backend] || backend;
}

function selectedTile() {
  const scenario = currentScenario();
  return scenario?.tiles.find((tile) => tileKey(tile) === state.selected) || scenario?.tiles[0];
}

function tileState(tile) {
  if (state.tick >= tile.end) return 'done';
  if (state.tick <= tile.start) return 'pending';
  return 'active';
}

function stepIndex(tile) {
  const scenario = currentScenario();
  if (!scenario || !tile) return -1;
  return Math.max(0, Math.min(scenario.steps.length - 1, state.tick - tile.start));
}

function activeStep(tile = selectedTile()) {
  const scenario = currentScenario();
  return tile && scenario && state.tick >= tile.start && state.tick < tile.end
    ? scenario.steps[stepIndex(tile)] : null;
}

// Last step that belongs to a repeated loop group; -1 when the program has no loop.
function lastLoopStepIndex(steps) {
  for (let index = steps.length - 1; index >= 0; index--) {
    if (steps[index].iteration !== null && steps[index].iteration !== undefined) return index;
  }
  return -1;
}

// The K-slice phase shown by the inspector and the GEMM view: the active step's
// iteration, the last iteration once the loop is over (epilogue / finished tile),
// and 0 before the loop starts. Works for loop-free programs (phase 0).
function phaseIndex(scenario, tile) {
  const steps = scenario?.steps || [];
  const lastLoop = lastLoopStepIndex(steps);
  const lastIteration = lastLoop >= 0 ? steps[lastLoop].iteration : 0;
  if (!tile || steps.length === 0) return 0;
  if (state.tick >= tile.end) return lastIteration;
  if (state.tick < tile.start) return 0;
  const index = Math.max(0, Math.min(steps.length - 1, state.tick - tile.start));
  const iteration = steps[index]?.iteration;
  if (iteration !== null && iteration !== undefined) return iteration;
  return index > lastLoop ? lastIteration : 0;
}

function syncPlayButton() {
  const playing = Boolean(state.timer);
  const label = t(playing ? 'pause' : 'play');
  $('play-button').innerHTML = `${playing ? 'Ⅱ' : '▶'} <span>${label}</span>`;
  $('play-button').setAttribute('aria-label', label);
}

function pause() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  syncPlayButton();
}

function play() {
  const scenario = currentScenario();
  if (!scenario) return;
  if (state.timer) { pause(); return; }
  if (state.tick >= scenario.total_ticks) state.tick = 0;
  state.timer = setInterval(() => {
    state.tick = Math.min(state.tick + 1, scenario.total_ticks);
    render();
    if (state.tick >= scenario.total_ticks) pause();
  }, state.speed);
  syncPlayButton();
  render();
}

function seek(value) {
  const scenario = currentScenario();
  if (!scenario) return;
  state.tick = Math.min(Math.max(Number(value) || 0, 0), scenario.total_ticks);
  if (state.tick >= scenario.total_ticks) pause();
  render();
}

function renderScenarioSwitch() {
  $('scenario-switch').innerHTML = state.model.scenarios.map((scenario) =>
    `<button type="button" data-scenario="${escapeHtml(scenario.id)}" class="${scenario.id === state.scenario ? 'active' : ''}" aria-pressed="${scenario.id === state.scenario}">${escapeHtml(displayLabel(scenario))}</button>`
  ).join('');
}

function renderCatalog() {
  const selected = $('model-select').value;
  $('model-select').innerHTML = state.catalog.map((entry) =>
    `<option value="${entry.id}">${escapeHtml(displayLabel(entry, 'title'))}</option>`
  ).join('');
  if (selected) $('model-select').value = selected;
}

function setLanguage(language) {
  if (!translations[language]) return;
  state.language = language;
  try { localStorage.setItem('op-show-language', language); } catch { /* Storage may be unavailable. */ }
  state.refIndex = state.model ? buildRefIndex(state.model) : null;
  state.timelineKey = null;
  applyLanguage();
  renderCatalog();
  syncPlayButton();
  render();
}

function renderMetrics() {
  const scenario = currentScenario();
  const sliceView = scenario.gemm_view || scenario.row_k_view;
  const shape = scenario.dimensions.map((dim) => `${dim.name}=${dim.extent}`)
    .concat(sliceView ? [`K=${sliceView.k_extent}`] : [])
    .concat(scenario.attention_view ? [`N=${scenario.attention_view.n_extent}`] : []).join(' · ');
  const sizes = scenario.dimensions.map((dim) => dim.size)
    .concat(sliceView ? [sliceView.k_size] : [])
    .concat(scenario.attention_view ? [scenario.attention_view.n_size] : []).join(' × ');
  const pad = scenario.tiles.reduce((sum, tile) => sum + tile.total_cells - tile.valid_cells, 0);
  const metrics = [
    [t('metricShape'), escapeHtml(shape), ''],
    [t('metricTileSize'), escapeHtml(sizes), t('elements')],
    [t('metricOutputTiles'), scenario.tiles.length, t('workItems')],
    [t('metricMasked'), pad, pad ? t('tailPadding') : t('fullyAligned')]
  ];
  $('metrics').innerHTML = metrics.map(([label, value, note]) => `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${escapeHtml(note)}</div></div>`).join('');
}

function renderGrid() {
  const scenario = currentScenario();
  const dimensions = scenario.dimensions;
  const columns = dimensions.length === 2 ? dimensions[1].count : dimensions[0].count;
  const columnAxis = dimensions.length === 2 ? dimensions[1] : dimensions[0];
  const template = `repeat(${columns}, minmax(0, 1fr))`;
  $('tile-grid').style.gridTemplateColumns = template;
  $('tile-column-axis').style.gridTemplateColumns = template;
  $('tile-column-axis').innerHTML = Array.from({ length: columns }, (_, index) =>
    `<span>${escapeHtml(columnAxis.name)}${index}</span>`).join('');
  const orientation = dimensions.length === 2
    ? `<span>${escapeHtml(dimensions[0].name)} ↓ ${t('axisRows')}</span><span>${escapeHtml(columnAxis.name)} → ${t('axisCols')}</span>`
    : `<span>${escapeHtml(dimensions[0].name)} → ${t('axisCols')}</span>`;
  $('grid-context').innerHTML = `<span class="grid-dimensions">${dimensions.map((dim) =>
    `<span class="grid-dimension">${escapeHtml(dim.name)}: ${dim.extent} / ${dim.size} = ${dim.count} ${t('tiles')}</span>`
  ).join('<span aria-hidden="true">·</span>')}</span><span class="grid-orientation">${orientation}</span><span>${scenario.grid.x} × ${scenario.grid.y} ${t('workers')}</span>`;
  // docs/model-format: the first 2D tile axis is the grid row and the second is the
  // column, independent of the worker iteration order the backend stores tiles in.
  const tiles = scenario.tiles.slice().sort((a, b) =>
    (a.coords[0] - b.coords[0]) || ((a.coords[1] ?? 0) - (b.coords[1] ?? 0)));
  $('tile-grid').innerHTML = tiles.map((tile) => {
    const status = tileState(tile);
    const tail = tile.valid_cells < tile.total_cells;
    const isSelected = tileKey(tile) === state.selected;
    const progress = Math.max(0, Math.min(1, (state.tick - tile.start) / tile.step_count));
    const range = tile.bounds.map((bound) => `${bound.axis}[${bound.start}:${bound.end}]`).join(' · ');
    const statusLabel = t(status === 'done' ? 'complete' : status);
    return `<button type="button" tabindex="${isSelected ? '0' : '-1'}" class="tile-card ${status} ${tail ? 'tail' : ''} ${isSelected ? 'selected' : ''}" data-tile="${tileKey(tile)}" aria-label="${escapeHtml(tileLabel(tile))}, ${escapeHtml(range)}, ${statusLabel}">
      <span class="tile-card-top"><span class="tile-id">${escapeHtml(tileLabel(tile))}</span><span class="tile-card-status">${statusLabel}</span></span>
      <span class="tile-card-sub">${escapeHtml(range)}</span><span class="tile-card-sub">${tile.valid_cells}/${tile.total_cells} ${t('validCells')}</span>
      <span class="tile-card-progress" style="width:${Math.round(progress * 100)}%"></span></button>`;
  }).join('');
  const independent = scenario.tiles.filter((tile) => !tile.dependencies.length).length;
  $('grid-footer').innerHTML = `<span>${t('gridParallel', {count: independent})}</span><span>${t('gridHint')}</span>`;
}

function renderOwnership() {
  const scenario = currentScenario();
  const groups = new Map();
  for (const tile of scenario.tiles) {
    const key = [tile.worker.x, tile.worker.y, tile.worker.z].join(',');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(tile);
  }
  const axes = ['x', 'y', 'z'].filter((axis) => axis === 'x' || scenario.grid[axis] > 1);
  const gridFormula = axes.map((axis) => state.model.worker_grid_formulas?.[axis] ?? '1').join(', ');
  const gridValues = axes.map((axis) => scenario.grid[axis]).join(' × ');
  const readyPids = [...groups.values()].filter((tiles) => tiles[0].start === 0).length;
  const strategy = state.model.parallelism;
  const refs = [state.model.grid_source_ref, ...(scenario.claim_source_refs || []), ...(strategy?.source_refs || [])];
  const uniqueRefs = [...new Set(refs)];
  const selected = selectedTile();
  const groupMap = scenario.task_group ? `<div class="ownership-group-map">${Array.from({length: scenario.task_group.count}, (_, index) => {
    const tasks = scenario.tiles.filter((tile) => tile.task_group === index);
    const axisIndex = scenario.dimensions.findIndex((axis) => axis.name === scenario.task_group.member_axis);
    const members = axisIndex < 0 ? [] : [...new Set(tasks.map((tile) => `${scenario.task_group.member_axis}${tile.coords[axisIndex]}`))];
    const pids = new Set(tasks.map((tile) => [tile.worker.x, tile.worker.y, tile.worker.z].join(',')));
    return `<div class="ownership-input-group ${selected?.task_group === index ? 'selected' : ''}"><strong>${escapeHtml(displayLabel(scenario.task_group))}${index}</strong>${members.length ? `<span>← ${escapeHtml(members.join(', '))}</span>` : ''}<small>${t('groupTasks', {count: tasks.length})} · ${t('groupPids', {count: pids.size})}</small></div>`;
  }).join('')}</div>` : '';
  $('ownership-summary').innerHTML = `<div class="ownership-stats">
      <div><span>${t('launchGrid')}</span><strong>(${escapeHtml(gridFormula)})</strong><small>→ ${gridValues} = ${groups.size} ${t('pidCount')}</small></div>
      <div><span>${t('taskCount')}</span><strong>${scenario.tiles.length}</strong><small>${t('claimsPerPid')} ${scenario.claim_count}</small></div>
      <div><span>${t('claimReady')}</span><strong>${readyPids}</strong><small>${t('claimUnit')}: ${escapeHtml(strategy ? displayLabel(strategy, 'claim_unit') : t('logicalTile'))}</small></div>
    </div>${strategy ? `<p class="ownership-strategy">${escapeHtml(displayLabel(strategy, 'summary'))}</p>` : ''}${groupMap}
    <div class="ownership-refs">${uniqueRefs.map((ref) => `<button type="button" data-source-ref="${escapeHtml(ref)}" title="${escapeHtml(t('claimSource'))}">${escapeHtml(ref)} · L${state.model.refs[ref].line} ↗</button>`).join('')}</div>`;
  $('ownership-claims').innerHTML = [...groups].map(([key, tiles]) => {
    tiles.sort((a, b) => a.claim_index - b.claim_index);
    const worker = tiles[0].worker;
    const pid = axes.map((axis) => `${state.model.worker_labels[axis] || `wid_${axis}`}=${worker[axis]}`).join(' · ');
    const selected = tiles.some((tile) => tileKey(tile) === state.selected);
    return `<div class="ownership-lane ${selected ? 'selected' : ''}"><div class="ownership-pid"><strong>PID (${axes.map((axis) => worker[axis]).join(', ')})</strong><small>${escapeHtml(pid)}</small></div>
      <div class="ownership-task-list" role="group" aria-label="${escapeHtml(pid)}">${tiles.map((tile, index) => {
        const active = tileKey(tile) === state.selected;
        const range = tile.bounds.map((bound) => `${bound.axis}[${bound.start}:${bound.end})`).join(' × ');
        const group = scenario.task_group && tile.task_group !== null ? `${displayLabel(scenario.task_group)}${tile.task_group}` : '';
        const status = tileState(tile);
        return `${index ? '<span class="ownership-arrow" aria-hidden="true">→</span>' : ''}<button type="button" class="ownership-task ${status} ${active ? 'selected' : ''}" data-claim-tile="${tileKey(tile)}" aria-label="${escapeHtml(`${tileLabel(tile)} ${range} ${group}`)}" aria-pressed="${active}"><span>${index + 1}. ${escapeHtml(tileLabel(tile))}</span><small>${escapeHtml(range)}</small>${group ? `<em>${escapeHtml(group)}</em>` : ''}</button>`;
      }).join('')}</div></div>`;
  }).join('');
}

function renderGemmView() {
  const scenario = currentScenario();
  const view = scenario.gemm_view;
  $('gemm-view').hidden = !view;
  if (!view) return;
  $('gemm-legend').innerHTML = `<span><i class="matrix-swatch"></i>${t('validCell')}</span><span><i class="matrix-swatch masked"></i>${t('maskedCell')}</span>`;
  const tile = selectedTile();
  const [m, n] = tile.bounds;
  const current = activeStep(tile);
  const phase = phaseIndex(scenario, tile);
  const kStart = phase * view.k_size;
  const kEnd = Math.min(kStart + view.k_size, view.k_extent);
  const roles = view.step_roles[current?.id] || [];
  $('gemm-phase').textContent = t('gemmPhase', {
    current: phase + 1, total: Math.ceil(view.k_extent / view.k_size)
  });
  const mExtent = scenario.dimensions[0].extent;
  const nExtent = scenario.dimensions[1].extent;
  const kExtent = view.k_extent;
  const percentage = (value, extent) => `${(value / extent * 100).toFixed(4)}%`;
  const rect = (className, top, left, height, width, rowExtent, colExtent) =>
    `<span class="gemm-map-region ${className}" style="top:${percentage(top, rowExtent)};left:${percentage(left, colExtent)};height:${percentage(height, rowExtent)};width:${percentage(width, colExtent)}"></span>`;
  function overview(name, rows, cols, rowAxis, colAxis, band, active, ref) {
    const range = `${rowAxis}[${active.row}:${active.row + active.height}) × ${colAxis}[${active.col}:${active.col + active.width})`;
    const outputTiles = name === 'C' ? scenario.tiles.map((item) => {
      const [row, col] = item.bounds;
      const picked = tileKey(item) === state.selected;
      return `<button type="button" class="gemm-output-tile ${tileState(item)} ${picked ? 'selected' : ''}" data-gemm-tile="${tileKey(item)}" tabindex="${picked ? 0 : -1}" aria-pressed="${picked}" aria-label="${escapeHtml(tileLabel(item))}, ${row.axis}[${row.start}:${row.end}) × ${col.axis}[${col.start}:${col.end})" style="top:${percentage(row.start, rows)};left:${percentage(col.start, cols)};height:${percentage(row.end - row.start, rows)};width:${percentage(col.end - col.start, cols)}">${escapeHtml(tileLabel(item))}</button>`;
    }).join('') : '';
    return `<div class="gemm-map-card ${name.toLowerCase()} ${roles.includes(name) ? 'focused' : ''}">
      <div class="gemm-map-heading"><strong>${name}</strong><span>${rows} × ${cols}</span></div>
      <div class="matrix-frame gemm-map-frame">
        <span class="matrix-axis col">${colAxis} [0:${cols}) →</span>
        <span class="matrix-axis row">${rowAxis} [0:${rows}) ↓</span>
        <div class="gemm-map-plot" role="${name === 'C' ? 'group' : 'img'}" aria-label="${name}: ${escapeHtml(range)}">
          ${rect('band', band.row, band.col, band.height, band.width, rows, cols)}
          ${name === 'C' ? '' : rect('active', active.row, active.col, active.height, active.width, rows, cols)}
          ${outputTiles}
        </div>
      </div>
      <div class="gemm-map-coordinates">${escapeHtml(range)}</div>
      <button type="button" class="gemm-map-source" data-source-ref="${escapeHtml(ref)}">L${state.model.refs[ref].line} ↗</button>
    </div>`;
  }
  const tail = m.end < m.tile_end || n.end < n.tile_end || kEnd - kStart < view.k_size;
  $('gemm-dependency').innerHTML = `<div class="gemm-map-intro"><span class="gemm-map-equation">A[M,K] × B[K,N] → C[M,N]</span>
      <p>${t('gemmMapHint')}</p>${tail ? `<span class="gemm-map-tail">${t('gemmMapTail')}</span>` : ''}</div>
    ${overview('B', kExtent, nExtent, 'K', view.n_axis,
      {row: 0, col: n.start, height: kExtent, width: n.end - n.start},
      {row: kStart, col: n.start, height: kEnd - kStart, width: n.end - n.start}, view.source_refs.b)}
    ${overview('A', mExtent, kExtent, view.m_axis, 'K',
      {row: m.start, col: 0, height: m.end - m.start, width: kExtent},
      {row: m.start, col: kStart, height: m.end - m.start, width: kEnd - kStart}, view.source_refs.a)}
    ${overview('C', mExtent, nExtent, view.m_axis, view.n_axis,
      {row: m.start, col: n.start, height: m.end - m.start, width: n.end - n.start},
      {row: m.start, col: n.start, height: m.end - m.start, width: n.end - n.start}, view.source_refs.c)}`;
  function matrix(name, rowStart, colStart, rows, cols, rowEnd, colEnd, rowAxis, colAxis, ref) {
    let cells = '';
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const valid = rowStart + r < rowEnd && colStart + c < colEnd;
      const coord = `${rowAxis}${rowStart + r}, ${colAxis}${colStart + c}`;
      cells += `<span class="gemm-cell ${valid ? '' : 'masked'}" title="${escapeHtml(coord)} · ${t(valid ? 'validCell' : 'maskedCell')}"></span>`;
    }
    const count = (rowEnd - rowStart) * (colEnd - colStart);
    const rowLabel = `${rowAxis}[${rowStart}:${rowEnd})`;
    const colLabel = `${colAxis}[${colStart}:${colEnd})`;
    return `<div class="gemm-matrix-card ${name.toLowerCase()} ${roles.includes(name) ? 'focused' : ''}" data-matrix="${name}">
      <div class="gemm-matrix-top"><strong>${name}</strong><button type="button" data-source-ref="${escapeHtml(ref)}">L${state.model.refs[ref].line} ↗</button></div>
      <div class="matrix-frame gemm-matrix-frame">
        <span class="matrix-axis col">${escapeHtml(colLabel)}</span>
        <span class="matrix-axis row">${escapeHtml(rowLabel)}</span>
        <div class="gemm-cell-grid" style="grid-template-columns:repeat(${cols},1fr)" role="img" aria-label="${name}: ${t('gemmValid', {count, total: rows * cols})}">${cells}</div>
      </div>
      <div class="gemm-matrix-valid">${t('gemmValid', {count, total: rows * cols})}</div></div>`;
  }
  $('gemm-matrices').innerHTML = [
    matrix('A', m.start, kStart, m.tile_end - m.start, view.k_size, m.end, kEnd,
      view.m_axis, 'K', view.source_refs.a),
    '<div class="gemm-operator" aria-hidden="true">×</div>',
    matrix('B', kStart, n.start, view.k_size, n.tile_end - n.start, kEnd, n.end,
      'K', view.n_axis, view.source_refs.b),
    '<div class="gemm-operator" aria-hidden="true">→</div>',
    matrix('C', m.start, n.start, m.tile_end - m.start, n.tile_end - n.start, m.end, n.end,
      view.m_axis, view.n_axis, view.source_refs.c)
  ].join('');
}

function renderRowKView() {
  const scenario = currentScenario();
  const view = scenario.row_k_view;
  $('rowk-view').hidden = !view;
  if (!view) return;
  const tile = selectedTile();
  const row = tile.bounds[0];
  const phase = phaseIndex(scenario, tile);
  const kStart = phase * view.k_size;
  const kEnd = Math.min(kStart + view.k_size, view.k_extent);
  const role = view.step_roles[activeStep(tile)?.id] || [];
  const gemv = view.kind === 'gemv';
  $('rowk-title').textContent = t(gemv ? 'rowkGemvTitle' : 'rowkReduceTitle');
  $('rowk-equation').textContent = gemv ? 'A × x → y' : 'X → Y';
  $('rowk-hint').textContent = t('rowkHint');
  $('rowk-phase').textContent = t('gemmPhase', {
    current: phase + 1, total: Math.ceil(view.k_extent / view.k_size)
  });

  function card(name, matrixRole, rowStart, colStart, rows, cols, rowEnd, colEnd, rowAxis, colAxis, ref) {
    let cells = '';
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const valid = rowStart + r < rowEnd && colStart + c < colEnd;
      const coord = rowAxis && colAxis ? `${rowAxis}${rowStart + r}, ${colAxis}${colStart + c}`
        : rowAxis ? `${rowAxis}${rowStart + r}` : `${colAxis}${colStart + c}`;
      cells += `<span class="gemm-cell ${valid ? '' : 'masked'}" title="${escapeHtml(coord)} · ${t(valid ? 'validCell' : 'maskedCell')}"></span>`;
    }
    const count = (rowEnd - rowStart) * (colEnd - colStart);
    const rowLabel = rowAxis ? `${rowAxis}[${rowStart}:${rowEnd})` : '';
    const colLabel = colAxis ? `${colAxis}[${colStart}:${colEnd})` : '';
    return `<div class="gemm-matrix-card rowk-card ${matrixRole} ${role.includes(matrixRole) ? 'focused' : ''}">
      <div class="gemm-matrix-top"><strong>${name}</strong><button type="button" data-source-ref="${escapeHtml(ref)}">L${state.model.refs[ref].line} ↗</button></div>
      <div class="matrix-frame gemm-matrix-frame"><span class="matrix-axis col">${escapeHtml(colLabel)}</span><span class="matrix-axis row">${escapeHtml(rowLabel)}</span>
      <div class="gemm-cell-grid rowk-cell-grid" style="grid-template-columns:repeat(${cols},1fr)" role="img" aria-label="${name}: ${t('gemmValid', {count, total: rows * cols})}">${cells}</div></div>
      <div class="gemm-matrix-valid">${t('gemmValid', {count, total: rows * cols})}</div></div>`;
  }
  const matrix = card(gemv ? 'A' : 'X', 'matrix', row.start, kStart,
    row.tile_end - row.start, view.k_size, row.end, kEnd,
    view.output_axis, 'K', view.source_refs.matrix);
  const vector = gemv ? card('x', 'vector', 0, kStart, 1, view.k_size, 1, kEnd,
    '', 'K', view.source_refs.vector) : '';
  const output = card(gemv ? 'y' : 'Y', 'output', row.start, 0,
    row.tile_end - row.start, 1, row.end, 1,
    view.output_axis, '', view.source_refs.output);
  $('rowk-matrices').className = `rowk-matrices ${view.kind}`;
  $('rowk-matrices').innerHTML = gemv
    ? `${matrix}<div class="gemm-operator" aria-hidden="true">×</div>${vector}<div class="gemm-operator" aria-hidden="true">→</div>${output}`
    : `${matrix}<div class="gemm-operator" aria-hidden="true">→</div>${output}`;
}

function renderAttentionView() {
  const scenario = currentScenario();
  const view = scenario.attention_view;
  $('attention-view').hidden = !view;
  if (!view) return;
  const tile = selectedTile();
  const [m, d] = tile.bounds;
  const phase = phaseIndex(scenario, tile);
  const nStart = phase * view.n_size;
  const nEnd = Math.min(nStart + view.n_size, view.n_extent);
  const roles = view.step_roles[activeStep(tile)?.id] || [];
  $('attention-phase').textContent = t('attentionPhase', {
    current: phase + 1, total: Math.ceil(view.n_extent / view.n_size)
  });

  function card(name, rowStart, colStart, rows, cols, rowEnd, colEnd, rowAxis, colAxis, ref) {
    let cells = '';
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const valid = rowStart + r < rowEnd && colStart + c < colEnd;
      const coordinate = `${rowAxis}${rowStart + r}, ${colAxis}${colStart + c}`;
      cells += `<span class="gemm-cell ${valid ? '' : 'masked'}" title="${escapeHtml(coordinate)} · ${t(valid ? 'validCell' : 'maskedCell')}"></span>`;
    }
    const count = (rowEnd - rowStart) * (colEnd - colStart);
    return `<div class="gemm-matrix-card attention-card ${name.toLowerCase()} ${roles.includes(name) ? 'focused' : ''}">
      <div class="gemm-matrix-top"><strong>${name}</strong><button type="button" data-source-ref="${escapeHtml(ref)}">L${state.model.refs[ref].line} ↗</button></div>
      <div class="matrix-frame gemm-matrix-frame"><span class="matrix-axis col">${escapeHtml(colAxis)}[${colStart}:${colEnd})</span><span class="matrix-axis row">${escapeHtml(rowAxis)}[${rowStart}:${rowEnd})</span>
      <div class="gemm-cell-grid attention-cell-grid" style="grid-template-columns:repeat(${cols},1fr)" role="img" aria-label="${name}: ${t('gemmValid', {count, total: rows * cols})}">${cells}</div></div>
      <div class="gemm-matrix-valid">${t('gemmValid', {count, total: rows * cols})}</div></div>`;
  }
  $('attention-matrices').innerHTML = [
    card('Q', m.start, d.start, m.tile_end - m.start, d.tile_end - d.start,
      m.end, d.end, view.q_axis, view.d_axis, view.source_refs.q),
    card('K', nStart, d.start, view.n_size, d.tile_end - d.start,
      nEnd, d.end, 'N', view.d_axis, view.source_refs.k),
    card('V', nStart, d.start, view.n_size, d.tile_end - d.start,
      nEnd, d.end, 'N', view.d_axis, view.source_refs.v),
    card('S', m.start, nStart, m.tile_end - m.start, view.n_size,
      m.end, nEnd, view.q_axis, 'N', view.source_refs.score),
    card('O', m.start, d.start, m.tile_end - m.start, d.tile_end - d.start,
      m.end, d.end, view.q_axis, view.d_axis, view.source_refs.o)
  ].join('');
  $('attention-recurrence').innerHTML = `<p>${t('attentionScores')}</p><p>${t('attentionRecurrence')}</p>
    <button type="button" data-source-ref="${escapeHtml(view.source_refs.softmax)}">L${state.model.refs[view.source_refs.softmax].line} ↗</button>`;
}

function renderPlayback() {
  const scenario = currentScenario();
  const tile = selectedTile();
  const finished = scenario.tiles.reduce((sum, item) => sum + Math.max(0, Math.min(item.step_count, state.tick - item.start)), 0);
  const total = scenario.tiles.reduce((sum, item) => sum + item.step_count, 0);
  $('progress-count').textContent = `${Math.round(finished / total * 100)}%`;
  $('tick-label').textContent = `${state.tick} / ${scenario.total_ticks}`;
  $('scrubber').max = scenario.total_ticks;
  $('scrubber').value = state.tick;
  const active = activeStep();
  $('current-stage').textContent = active ? displayLabel(active) : state.tick < tile.start ? t('waiting') : t('stageComplete');
  $('stage-strip').innerHTML = scenario.steps.map((step, index) => {
    const stageTick = tile.start + index;
    const stageStatus = state.tick > stageTick ? 'complete' : state.tick === stageTick ? 'current' : '';
    const repeatLabel = step.iteration === null ? t('stepNumber', {number: index + 1}) :
      `${step.loop_axis}${step.iteration} · ${t('stepNumber', {number: index + 1})}`;
    const label = displayLabel(step);
    return `<button type="button" class="stage-chip ${stageStatus}" data-step="${index}" title="${escapeHtml(t('seekTo', {label}))}"><b>${escapeHtml(label)}</b><span>${escapeHtml(repeatLabel)}</span></button>`;
  }).join('');
  // Keep the active stage chip visible inside its own horizontal scroller.
  const strip = $('stage-strip');
  const current = strip.querySelector('.stage-chip.current');
  if (current) {
    const left = current.offsetLeft;
    const right = left + current.offsetWidth;
    if (left < strip.scrollLeft) strip.scrollLeft = Math.max(0, left - 8);
    else if (right > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = right - strip.clientWidth + 8;
  }
}

function renderCells(tile, scenario) {
  const bounds = tile.bounds;
  const rows = bounds[0].tile_end - bounds[0].start;
  const cols = bounds.length > 1 ? bounds[1].tile_end - bounds[1].start : 1;
  if (rows * cols > 512) return `<p class="cell-summary">${t('tooManyCells')}</p>`;
  let cells = '';
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const valid = bounds[0].start + row < bounds[0].end &&
        (bounds.length === 1 || bounds[1].start + col < bounds[1].end);
      const coordinate = bounds.length === 1 ? `${bounds[0].start + row}` :
        `${bounds[0].start + row}, ${bounds[1].start + col}`;
      cells += `<span class="cell ${valid ? '' : 'masked'}" title="${valid ? t('validCell') : t('maskedCell')} [${coordinate}]"></span>`;
    }
  }
  const rowAxis = bounds[0];
  const colAxis = bounds.length > 1 ? bounds[1] : null;
  const label = `${tile.valid_cells} / ${tile.total_cells} ${t('validCells')}`;
  return `<div class="matrix-frame inspector-matrix">
    ${colAxis ? `<span class="matrix-axis col">${escapeHtml(colAxis.axis)}[${colAxis.start}:${colAxis.end})</span>` : ''}
    <span class="matrix-axis row">${escapeHtml(rowAxis.axis)}[${rowAxis.start}:${rowAxis.end})</span>
    <div class="cell-grid" style="grid-template-columns:repeat(${bounds.length === 1 ? rows : cols},1fr)" role="img" aria-label="${escapeHtml(label)}">${cells}</div>
  </div>`;
}

function renderInspector() {
  const scenario = currentScenario();
  const tile = selectedTile();
  const step = activeStep();
  const bounds = tile.bounds.map((bound) => `${bound.axis}[${bound.start}:${bound.end})`).join(' × ');
  const worker = Object.entries(tile.worker).filter(([key, value]) => value || key === 'x')
    .map(([key, value]) => `${state.model.worker_labels[key] || `wid_${key}`}=${value}`).join(' · ');
  const tail = tile.total_cells > tile.valid_cells;
  $('inspector-title').textContent = tileLabel(tile);
  const loopSlice = step && step.loop_axis && step.loop_valid != null && step.iteration != null
    ? `<div class="cell-summary" style="margin-top:8px">${escapeHtml(t('loopSlice', {axis: step.loop_axis, number: step.iteration + 1, valid: step.loop_valid, total: scenario.params[step.loop_tile]}))}</div>` : '';
  const phase = phaseIndex(scenario, tile);
  const addresses = tile.addresses.map((address) => {
    const sample = address.samples[Math.min(phase, address.samples.length - 1)];
    const substitutions = Object.entries(sample.variables).map(([key, value]) => `${key}=${value}`).join(', ');
    return `<div class="formula-card"><div class="formula-title">${escapeHtml(displayLabel(address))}</div><div class="formula-expr">${escapeHtml(address.expr)}</div><div class="formula-result">${escapeHtml(substitutions)} → ${t('offset')} <strong>${sample.offset}</strong></div><button class="formula-source" data-source-ref="${escapeHtml(address.source_ref)}">${t('viewSource')}</button></div>`;
  }).join('');
  $('inspector-content').innerHTML = `
    <div class="inspector-section" data-detail-section="summary"><div class="inspector-label">${t('logicalTile')}</div><span class="inspector-id">${escapeHtml(tileLabel(tile))}</span>${tail ? `<span class="inspector-badge">${t('tailTile')}</span>` : ''}</div>
    <div class="inspector-section" data-detail-section="summary"><div class="inspector-label">${t('ownership')}</div><div class="detail-table">
      <span class="key">${t('worker')}</span><span class="value">${escapeHtml(worker)}</span>
      <span class="key">${t('output')}</span><span class="value">${escapeHtml(bounds)}</span>
      <span class="key">${t('valid')}</span><span class="value">${tile.valid_cells} / ${tile.total_cells} ${t('validCells')}</span>
      <span class="key">${t('dependencies')}</span><span class="value">${tile.dependencies.length ? escapeHtml(tile.dependencies.map((dep) => `T[${dep.join(', ')}]`).join(', ')) : t('noDependencies')}</span>
      <span class="key">${t('current')}</span><span class="value">${state.tick >= tile.end ? t('stageComplete') : state.tick < tile.start ? t('waiting') : escapeHtml(displayLabel(step))}</span>
    </div></div>
    <div class="inspector-section" data-detail-section="mask"><div class="inspector-label">${t('validAndMask')}</div>${renderCells(tile, scenario)}<div class="cell-summary">${t('cellsLegend')} ${tail ? `<strong>${t('maskedTail', {count: tile.total_cells - tile.valid_cells})}</strong>` : t('noPadding')}</div>${loopSlice}</div>
    <div class="inspector-section inspector-bottom" data-detail-section="addresses"><div class="inspector-label">${t('addressOffsets')}</div>${addresses || `<div class="cell-summary">${t('noAddress')}</div>`}</div>`;
}

function renderSource() {
  const scenario = currentScenario();
  const step = scenario && state.model.schema_version !== 2 ? activeStep() : null;
  const refKey = state.model.schema_version === 2 && state.model.status !== 'unknown' ? TraceView.currentRef() :
    (step?.source_ref || state.model.grid_source_ref || Object.keys(state.model.refs)[0]);
  const highlighted = state.highlightedRef || refKey;
  const line = state.model.refs[highlighted]?.line;
  const refIndex = state.refIndex || { usages: {}, lines: {} };
  const usages = refIndex.usages[highlighted] || [];
  const lineRefs = refIndex.lines[line] || [];
  $('source-path').textContent = state.model.source_path;
  $('source-explainer').innerHTML = `<div><strong>${t('sourceEvidence')}</strong>${t('sourceExplanation')}</div><div class="source-ref-list">${Object.entries(state.model.refs).slice(0, 7).map(([key, value]) => `<button data-source-ref="${escapeHtml(key)}">${escapeHtml(key)} · L${value.line}</button>`).join('')}</div><div class="source-usage"><div class="usage-title">${t('sourceUsage')}</div><div class="usage-chips">${usages.length ? usages.map((usage) => `<span>${escapeHtml(usage)}</span>`).join('') : `<span class="usage-empty">${t('sourceUsageEmpty')}</span>`}</div>${lineRefs.length > 1 ? `<div class="usage-chips" style="margin-top:6px">${lineRefs.map((key) => `<button data-source-ref="${escapeHtml(key)}">${escapeHtml(key)}</button>`).join('')}</div>` : ''}</div>`;
  $('source-code').innerHTML = state.source.split('\n').map((text, index) => {
    const cited = (refIndex.lines[index + 1] || []).length > 0;
    const interactive = cited ? ` tabindex="0" role="button" aria-label="${escapeHtml(t('sourceUsage'))}"` : '';
    return `<span class="code-line${index + 1 === line ? ' highlight' : ''}${cited ? ' cited' : ''}" id="line-${index + 1}" data-line="${index + 1}"${interactive}><span class="line-no">${String(index + 1).padStart(2, '0')}</span>${escapeHtml(text || ' ')}</span>`;
  }).join('');
  const target = $(`line-${line}`);
  if (target && state.detailTab === 'source' && $('source-code').dataset.lastLine !== String(line)) {
    const viewport = $('source-code').parentElement;
    viewport.scrollTop += target.getBoundingClientRect().top - viewport.getBoundingClientRect().top - viewport.clientHeight / 2;
    $('source-code').dataset.lastLine = String(line);
  }
}

function showErrorPanel(message) {
  pause();
  TraceView.pause();
  $('workbench-layout').dataset.modelKind = 'error';
  $('scenario-switch').innerHTML = '';
  $('unknown-panel').hidden = false;
  $('unknown-panel').className = 'unknown';
  $('unknown-panel').textContent = message;
  $('visualization').hidden = true;
  $('trace-visualization').hidden = true;
  $('metrics').hidden = true;
  $('metrics').innerHTML = '';
}

function render() {
  if (!state.model) return;
  $('backend-pill').textContent = backendLabel(state.model.backend);
  const unknown = state.model.status === 'unknown';
  const trace = state.model.schema_version === 2;
  $('workbench-layout').dataset.modelKind = unknown ? 'unknown' : trace ? 'trace' :
    currentScenario()?.gemm_view ? 'gemm' : currentScenario()?.attention_view ? 'attention' :
    currentScenario()?.row_k_view ? 'rowk' : 'tile';
  if (state.detailTab === 'tasks' && !trace) openDetailTab('summary');
  $('visualization').hidden = unknown || trace;
  $('trace-visualization').hidden = unknown || !trace;
  $('metrics').hidden = unknown || trace;
  if (unknown || trace) $('metrics').innerHTML = '';
  $('unknown-panel').hidden = !unknown;
  $('timeline-panel').hidden = unknown || trace;
  if (unknown) {
    $('scenario-switch').innerHTML = '';
    $('unknown-panel').className = 'unknown';
    $('unknown-panel').textContent = t('mappingUnknown', {note: displayLabel(state.model, 'note')});
    if (state.detailTab !== 'source') openDetailTab('source');
    renderSource();
    return;
  }
  renderScenarioSwitch();
  if (trace) { TraceView.render(); renderSource(); return; }
  renderMetrics();
  renderOwnership();
  renderGrid();
  renderGemmView();
  renderRowKView();
  renderAttentionView();
  renderPlayback();
  renderTimeline();
  renderInspector();
  renderSource();
}

let loadToken = 0;

async function loadModel(index) {
  const token = ++loadToken;
  pause();
  TraceView.pause();
  try {
    const [modelResponse, sourceResponse] = await Promise.all([
      fetch(`/models/${index}.json`), fetch(`/sources/${index}.txt`)
    ]);
    if (token !== loadToken) return;
    if (!modelResponse.ok || !sourceResponse.ok) throw new Error(t('loadError'));
    const model = await modelResponse.json();
    const source = await sourceResponse.text();
    if (token !== loadToken) return;
    state.model = model;
    state.source = source;
    state.refIndex = buildRefIndex(state.model);
    state.timelineKey = null;
    state.scenario = state.model.scenarios[0]?.id || null;
    state.selected = state.model.schema_version === 2 ? null :
      (state.model.scenarios[0]?.tiles[0] ? tileKey(state.model.scenarios[0].tiles[0]) : null);
    state.tick = 0;
    state.highlightedRef = null;
    openDetailTab(state.model.schema_version === 2 && state.model.status !== 'unknown' ? 'tasks' : 'summary');
    delete $('source-code').dataset.lastLine;
    $('model-select').value = String(index);
    if (state.model.schema_version === 2 && state.model.status !== 'unknown') {
      TraceView.init(state.model, currentScenario(), (ref, scroll) => {
        state.highlightedRef = ref;
        renderSource();
        if (scroll) openDetailTab('source');
      }, () => state.language);
    }
    render();
    state.modelIndex = Number(index);
  } catch (error) {
    if (token !== loadToken) return;
    showErrorPanel(t('loadError'));
    if (state.modelIndex !== null) $('model-select').value = String(state.modelIndex);
  }
}

async function initialize() {
  try {
    const response = await fetch('/catalog.json');
    if (!response.ok) throw new Error(t('catalogError'));
    state.catalog = await response.json();
    const hash = parseHash();
    if (hash?.language && translations[hash.language]) {
      state.language = hash.language;
      applyLanguage();
    }
    renderCatalog();
    const requested = Number.parseInt(hash?.model ?? '', 10);
    const index = Number.isInteger(requested) && requested >= 0 && requested < state.catalog.length ? requested : 0;
    await loadModel(index);
    applyHashState(hash);
  } catch (error) {
    showErrorPanel(t('catalogError'));
  }
}

$('model-select').addEventListener('change', (event) => loadModel(event.target.value));
document.querySelector('.language-switch').addEventListener('click', (event) => {
  const button = event.target.closest('[data-lang]');
  if (button) setLanguage(button.dataset.lang);
});
$('scenario-switch').addEventListener('click', (event) => {
  const button = event.target.closest('[data-scenario]');
  if (!button) return;
  pause();
  state.scenario = button.dataset.scenario;
  if (state.model.schema_version === 2) TraceView.scenarioChange(currentScenario());
  else state.selected = tileKey(currentScenario().tiles[0]);
  state.tick = 0;
  state.highlightedRef = null;
  render();
});
$('tile-grid').addEventListener('click', (event) => {
  const button = event.target.closest('[data-tile]');
  if (!button) return;
  state.selected = button.dataset.tile;
  state.highlightedRef = null;
  render();
  // render() replaces the grid; keep keyboard focus on the tile the user picked.
  $('tile-grid').querySelector(`[data-tile="${state.selected}"]`)?.focus({preventScroll: true});
});
$('ownership-view').addEventListener('click', (event) => {
  const tileButton = event.target.closest('[data-claim-tile]');
  if (tileButton) {
    state.selected = tileButton.dataset.claimTile;
    state.highlightedRef = null;
    render();
    $('ownership-view').querySelector(`[data-claim-tile="${state.selected}"]`)?.focus({preventScroll: true});
    return;
  }
  const sourceButton = event.target.closest('[data-source-ref]');
  if (sourceButton) {
    jumpToSource(sourceButton.dataset.sourceRef);
  }
});
// One tab stop for the whole tile matrix; arrow keys move selection (WAI-ARIA grid pattern).
$('tile-grid').addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
  if (!event.target.closest?.('[data-tile]')) return;
  const scenario = currentScenario();
  const tile = selectedTile();
  if (!scenario || !tile) return;
  event.preventDefault();
  event.stopPropagation();
  const dimensions = scenario.dimensions;
  const coords = [...tile.coords];
  if (dimensions.length === 1) {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') coords[0] -= 1;
    else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') coords[0] += 1;
    else coords[0] = event.key === 'Home' ? 0 : dimensions[0].count - 1;
  } else {
    if (event.key === 'ArrowLeft') coords[1] -= 1;
    if (event.key === 'ArrowRight') coords[1] += 1;
    if (event.key === 'ArrowUp') coords[0] -= 1;
    if (event.key === 'ArrowDown') coords[0] += 1;
    if (event.key === 'Home') coords[1] = 0;
    if (event.key === 'End') coords[1] = dimensions[1].count - 1;
  }
  const next = scenario.tiles.find((item) => item.coords.every((value, index) => value === coords[index]));
  if (!next || tileKey(next) === state.selected) return;
  state.selected = tileKey(next);
  state.highlightedRef = null;
  render();
  const buttons = $('tile-grid').querySelectorAll ? [...$('tile-grid').querySelectorAll('[data-tile]')] : [];
  buttons.find((node) => node.dataset.tile === state.selected)?.focus();
});
$('gemm-view').addEventListener('click', (event) => {
  const tileButton = event.target.closest('[data-gemm-tile]');
  if (tileButton) {
    state.selected = tileButton.dataset.gemmTile;
    state.highlightedRef = null;
    render();
    $('gemm-view').querySelector(`[data-gemm-tile="${state.selected}"]`)?.focus({preventScroll: true});
    return;
  }
  const button = event.target.closest('[data-source-ref]');
  if (!button) return;
  jumpToSource(button.dataset.sourceRef);
});
$('gemm-view').addEventListener('keydown', (event) => {
  if (!event.target.closest?.('[data-gemm-tile]') ||
      !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
  const scenario = currentScenario();
  const coords = [...selectedTile().coords];
  const axis = scenario.dimensions[1];
  if (event.key === 'ArrowLeft') coords[1] -= 1;
  if (event.key === 'ArrowRight') coords[1] += 1;
  if (event.key === 'ArrowUp') coords[0] -= 1;
  if (event.key === 'ArrowDown') coords[0] += 1;
  if (event.key === 'Home') coords[1] = 0;
  if (event.key === 'End') coords[1] = axis.count - 1;
  const next = scenario.tiles.find((item) => item.coords.every((value, index) => value === coords[index]));
  event.preventDefault();
  event.stopPropagation();
  if (!next) return;
  state.selected = tileKey(next);
  state.highlightedRef = null;
  render();
  $('gemm-view').querySelector(`[data-gemm-tile="${state.selected}"]`)?.focus({preventScroll: true});
});
$('rowk-matrices').addEventListener('click', (event) => {
  const button = event.target.closest('[data-source-ref]');
  if (!button) return;
  jumpToSource(button.dataset.sourceRef);
});
$('attention-view').addEventListener('click', (event) => {
  const button = event.target.closest('[data-source-ref]');
  if (!button) return;
  jumpToSource(button.dataset.sourceRef);
});
$('stage-strip').addEventListener('click', (event) => {
  const button = event.target.closest('[data-step]');
  if (!button) return;
  pause();
  seek(selectedTile().start + Number(button.dataset.step));
});
$('inspector-content').addEventListener('click', (event) => {
  const button = event.target.closest('[data-source-ref]');
  if (!button) return;
  jumpToSource(button.dataset.sourceRef);
});
$('gemm-mask-detail').addEventListener('click', (event) => {
  const button = event.target.closest('[data-source-ref]');
  if (button) jumpToSource(button.dataset.sourceRef);
});
document.querySelector('.detail-tabs').addEventListener('click', (event) => {
  const button = event.target.closest('[data-detail-tab]');
  if (button) openDetailTab(button.dataset.detailTab);
});
document.querySelector('.detail-tabs').addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
  const buttons = [...document.querySelectorAll('[data-detail-tab]')].filter((node) => getComputedStyle(node).display !== 'none');
  const index = buttons.findIndex((node) => node.dataset.detailTab === state.detailTab);
  const next = buttons[(index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length];
  if (!next) return;
  event.preventDefault();
  openDetailTab(next.dataset.detailTab);
  next.focus();
});
$('source-explainer').addEventListener('click', (event) => {
  const button = event.target.closest('[data-source-ref]');
  if (!button) return;
  state.highlightedRef = button.dataset.sourceRef;
  renderSource();
});
$('play-button').addEventListener('click', play);
$('step-button').addEventListener('click', () => { pause(); seek(state.tick + 1); });
$('reset-button').addEventListener('click', () => { pause(); seek(0); });
$('scrubber').addEventListener('input', (event) => { pause(); seek(event.target.value); });
$('speed-select').addEventListener('change', (event) => {
  state.speed = Number(event.target.value);
  if (state.timer) { pause(); play(); }
});
window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const active = document.activeElement;
  if (active && (active.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(active.tagName))) return;
  if (!state.model) return;
  const trace = state.model.schema_version === 2 && state.model.status !== 'unknown';
  if (event.code === 'Space') {
    event.preventDefault();
    if (trace) TraceView.toggle(); else play();
  }
  if (event.code === 'ArrowRight') {
    event.preventDefault();
    if (trace) TraceView.step(1); else { pause(); seek(state.tick + 1); }
  }
  if (event.code === 'ArrowLeft') {
    event.preventDefault();
    if (trace) TraceView.step(-1); else { pause(); seek(state.tick - 1); }
  }
});
$('share-button').addEventListener('click', copyShareLink);
$('timeline').addEventListener('click', (event) => {
  const button = event.target.closest('[data-timeline-tile]');
  if (!button) return;
  pause();
  state.selected = button.dataset.timelineTile;
  seek(Number(button.dataset.timelineStart) || 0);
});
function pickSourceLine(node) {
  if (!node) return;
  const refs = state.refIndex?.lines?.[Number(node.dataset.line)];
  if (!refs || !refs.length) return;
  state.highlightedRef = refs[0];
  renderSource();
}
$('source-code').addEventListener('click', (event) => pickSourceLine(event.target.closest('.code-line')));
$('source-code').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  const node = event.target.closest?.('.code-line');
  if (!node || !node.classList?.contains('cited')) return;
  event.preventDefault();
  pickSourceLine(node);
});
applyLanguage();
initialize();
