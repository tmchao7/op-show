const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[character]));

const translations = {
  zh: {
    topCaption: '无需硬件的算子图解', local: '本地', eyebrow: '源码 → TILE 映射 → 执行',
    heroLead: '看清每个 tile', heroAccent: '如何工作。',
    intro: '从 program 或 block ID 找到它负责的数据，再逐步查看哪些 tile 可以并行、哪些阶段必须依次执行。',
    logical: '逻辑执行进度', logicalNote: '无依赖的 tile 同时推进。时间轴展示依赖关系，不代表设备耗时。',
    settings: '可视化设置', operator: '算子', inputShape: '输入形状', statistics: '模型统计',
    sectionTiles: '01 / TILE 归属', tileGrid: '输出 tile 网格', outputTiles: '输出 tile',
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
    workers: '个执行单元', validCells: '个有效单元', gridParallel: '{count} 个 tile 可并行启动',
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
    catalogError: '无法加载算子列表', pageTitle: 'OP/SHOW — 算子 Tile 执行图解',
    traceHeading: '01 / 数据如何流动', traceTitle: '从 Producer Tile 到 Consumer 输出', sourceRank: '来源 Rank', targetRank: '目标 Rank', tensor: '输出张量', producerMatrix: 'Producer 数据矩阵', claimTable: 'PID 认领表', peerPath: 'peer_mem 数据路径', consumerMatrix: 'Consumer 输出矩阵', producerHint: '行是 block，列是目标 Rank 与 head group；颜色区分 G0–G4。', claimHint: '每行一个 Producer PID；每列是本 Wave 中按步长 16 认领的任务。', consumerHint: '行是来源 Rank 与 block，列是 head group；Q/K/V 使用相同坐标。', logicalEvents: '逻辑事件与依赖', traceTiming: '每一步同时完成所有依赖已满足的事件；Wave 不是跨 PID 的全局屏障，也不代表设备耗时。',
    gemmTitle: '当前 Tensor Core 矩阵分片', gemmHint: '点击上方 C tile，再按时间轴查看 A/B 的 K 分片如何累加到 C。斜纹元素在 shared memory 中填零或在输出时跳过。', gemmMaskLegend: 'A/B 的斜纹格：越界后填零；C 的斜纹格：不写入输出。', gemmPhase: 'K 分片 {current}/{total}', gemmValid: '{count}/{total} 个有效元素'
  },
  en: {
    topCaption: 'HARDWARE-FREE KERNEL WALKTHROUGH', local: 'LOCAL',
    eyebrow: 'SOURCE → TILE MAP → EXECUTION', heroLead: 'See the work', heroAccent: 'take shape.',
    intro: 'Follow each tile from its program or block ID to the data it owns. Step through the work that happens in parallel and the steps that must happen in order.',
    logical: 'Logical execution', logicalNote: 'All independent tiles advance together. The timeline shows dependencies, not device timing.',
    settings: 'Visualization settings', operator: 'OPERATOR', inputShape: 'INPUT SHAPE', statistics: 'Model statistics',
    sectionTiles: '01 / TILE OWNERSHIP', tileGrid: 'Output tile grid', outputTiles: 'Output tiles',
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
    workers: 'workers', validCells: 'valid cells', gridParallel: '{count} tiles can start in parallel',
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
    catalogError: 'Could not load catalog', pageTitle: 'OP/SHOW — Tile execution explorer',
    traceHeading: '01 / DATA FLOW', traceTitle: 'Producer tiles to Consumer output', sourceRank: 'SOURCE RANK', targetRank: 'TARGET RANK', tensor: 'OUTPUT TENSOR', producerMatrix: 'Producer data matrix', claimTable: 'PID claim table', peerPath: 'peer_mem data path', consumerMatrix: 'Consumer output matrix', producerHint: 'Rows are blocks; columns are target ranks and head groups. G0–G4 colors identify data groups.', claimHint: 'Each row is one Producer PID; columns show stride-16 claims in this wave.', consumerHint: 'Rows are source ranks and blocks; columns are head groups. Q/K/V share coordinates.', logicalEvents: 'Logical events and dependencies', traceTiming: 'Each step completes all events with satisfied dependencies. A wave is not a global PID barrier or device time.',
    gemmTitle: 'Current Tensor Core matrix slices', gemmHint: 'Select a C tile above, then use the timeline to see successive A/B K slices accumulate into C. Striped cells are zero-filled in shared memory or skipped on output.', gemmMaskLegend: 'Striped A/B cells are zero-filled; striped C cells are not stored.', gemmPhase: 'K slice {current}/{total}', gemmValid: '{count}/{total} valid elements'
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

const state = { catalog: [], model: null, source: '', scenario: null, selected: null,
  tick: 0, timer: null, speed: 650, highlightedRef: null, language: savedLanguage() };

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
  if (state.tick < tile.start) return 'pending';
  return 'active';
}

function stepIndex(tile) {
  const scenario = currentScenario();
  if (!scenario || !tile) return -1;
  return Math.max(0, Math.min(scenario.steps.length - 1, state.tick - tile.start));
}

function activeStep() {
  const tile = selectedTile();
  const scenario = currentScenario();
  return tile && scenario && state.tick >= tile.start && state.tick < tile.end
    ? scenario.steps[stepIndex(tile)] : null;
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
  applyLanguage();
  renderCatalog();
  syncPlayButton();
  if (state.model?.schema_version === 2) TraceView.render();
  render();
}

function renderMetrics() {
  const scenario = currentScenario();
  const shape = scenario.dimensions.map((dim) => `${dim.name}=${dim.extent}`)
    .concat(scenario.gemm_view ? [`K=${scenario.gemm_view.k_extent}`] : []).join(' · ');
  const sizes = scenario.dimensions.map((dim) => dim.size)
    .concat(scenario.gemm_view ? [scenario.gemm_view.k_size] : []).join(' × ');
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
  $('tile-grid').style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
  $('grid-context').innerHTML = `<span class="grid-dimensions">${dimensions.map((dim) =>
    `<span class="grid-dimension">${escapeHtml(dim.name)}: ${dim.extent} / ${dim.size} = ${dim.count} ${t('tiles')}</span>`
  ).join('<span aria-hidden="true">·</span>')}</span><span>${scenario.grid.x} × ${scenario.grid.y} ${t('workers')}</span>`;
  $('tile-grid').innerHTML = scenario.tiles.map((tile) => {
    const status = tileState(tile);
    const tail = tile.valid_cells < tile.total_cells;
    const progress = Math.max(0, Math.min(1, (state.tick - tile.start) / tile.step_count));
    const range = tile.bounds.map((bound) => `${bound.axis}[${bound.start}:${bound.end}]`).join(' · ');
    const statusLabel = t(status === 'done' ? 'complete' : status);
    return `<button type="button" class="tile-card ${status} ${tail ? 'tail' : ''} ${tileKey(tile) === state.selected ? 'selected' : ''}" data-tile="${tileKey(tile)}" aria-label="${escapeHtml(tileLabel(tile))}, ${escapeHtml(range)}, ${statusLabel}">
      <span class="tile-card-top"><span class="tile-id">${escapeHtml(tileLabel(tile))}</span><span class="tile-card-status">${statusLabel}</span></span>
      <span class="tile-card-sub">${escapeHtml(range)}</span><span class="tile-card-sub">${tile.valid_cells}/${tile.total_cells} ${t('validCells')}</span>
      <span class="tile-card-progress" style="width:${Math.round(progress * 100)}%"></span></button>`;
  }).join('');
  const independent = scenario.tiles.filter((tile) => !tile.dependencies.length).length;
  $('grid-footer').innerHTML = `<span>${t('gridParallel', {count: independent})}</span><span>${t('gridHint')}</span>`;
}

function renderGemmView() {
  const scenario = currentScenario();
  const view = scenario.gemm_view;
  $('gemm-view').hidden = !view;
  if (!view) return;
  const tile = selectedTile();
  const [m, n] = tile.bounds;
  const loopSteps = scenario.steps.filter((step) => step.iteration !== null);
  const current = activeStep();
  const phase = current?.iteration ?? (state.tick > tile.start + loopSteps.length - 1 ?
    loopSteps[loopSteps.length - 1].iteration : 0);
  const kStart = phase * view.k_size;
  const kEnd = Math.min(kStart + view.k_size, view.k_extent);
  const roles = view.step_roles[current?.id] || [];
  $('gemm-phase').textContent = t('gemmPhase', {
    current: phase + 1, total: Math.ceil(view.k_extent / view.k_size)
  });
  function matrix(name, rowStart, colStart, rows, cols, rowEnd, colEnd, rowAxis, colAxis, ref) {
    let cells = '';
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const valid = rowStart + r < rowEnd && colStart + c < colEnd;
      const coord = `${rowAxis}${rowStart + r}, ${colAxis}${colStart + c}`;
      cells += `<span class="gemm-cell ${valid ? '' : 'masked'}" title="${escapeHtml(coord)} · ${t(valid ? 'validCell' : 'maskedCell')}"></span>`;
    }
    const count = (rowEnd - rowStart) * (colEnd - colStart);
    return `<div class="gemm-matrix-card ${name.toLowerCase()} ${roles.includes(name) ? 'focused' : ''}">
      <div class="gemm-matrix-top"><strong>${name}</strong><button type="button" data-source-ref="${escapeHtml(ref)}">L${state.model.refs[ref].line} ↗</button></div>
      <div class="gemm-matrix-range">${rowAxis}[${rowStart}:${rowEnd}) × ${colAxis}[${colStart}:${colEnd})</div>
      <div class="gemm-cell-grid" style="grid-template-columns:repeat(${cols},1fr)" role="img" aria-label="${name}: ${t('gemmValid', {count, total: rows * cols})}">${cells}</div>
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
      `K${step.iteration} · ${t('stepNumber', {number: index + 1})}`;
    const label = displayLabel(step);
    return `<button type="button" class="stage-chip ${stageStatus}" data-step="${index}" title="${escapeHtml(t('seekTo', {label}))}"><b>${escapeHtml(label)}</b><span>${escapeHtml(repeatLabel)}</span></button>`;
  }).join('');
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
  return `<div class="cell-grid" style="grid-template-columns:repeat(${bounds.length === 1 ? rows : cols},1fr)">${cells}</div>`;
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
  const phase = step?.iteration ?? (state.tick >= scenario.total_ticks ? Math.max(...scenario.steps.map((item) => item.iteration ?? 0)) : 0);
  const addresses = tile.addresses.map((address) => {
    const sample = address.samples[Math.min(phase, address.samples.length - 1)];
    const substitutions = Object.entries(sample.variables).map(([key, value]) => `${key}=${value}`).join(', ');
    return `<div class="formula-card"><div class="formula-title">${escapeHtml(displayLabel(address))}</div><div class="formula-expr">${escapeHtml(address.expr)}</div><div class="formula-result">${escapeHtml(substitutions)} → ${t('offset')} <strong>${sample.offset}</strong></div><button class="formula-source" data-source-ref="${escapeHtml(address.source_ref)}">${t('viewSource')}</button></div>`;
  }).join('');
  $('inspector-content').innerHTML = `
    <div class="inspector-section"><div class="inspector-label">${t('logicalTile')}</div><span class="inspector-id">${escapeHtml(tileLabel(tile))}</span>${tail ? `<span class="inspector-badge">${t('tailTile')}</span>` : ''}</div>
    <div class="inspector-section"><div class="inspector-label">${t('ownership')}</div><div class="detail-table">
      <span class="key">${t('worker')}</span><span class="value">${escapeHtml(worker)}</span>
      <span class="key">${t('output')}</span><span class="value">${escapeHtml(bounds)}</span>
      <span class="key">${t('valid')}</span><span class="value">${tile.valid_cells} / ${tile.total_cells} ${t('validCells')}</span>
      <span class="key">${t('dependencies')}</span><span class="value">${tile.dependencies.length ? escapeHtml(tile.dependencies.map((dep) => `T[${dep.join(', ')}]`).join(', ')) : t('noDependencies')}</span>
      <span class="key">${t('current')}</span><span class="value">${state.tick >= tile.end ? t('stageComplete') : state.tick < tile.start ? t('waiting') : escapeHtml(displayLabel(step))}</span>
    </div></div>
    <div class="inspector-section"><div class="inspector-label">${t('validAndMask')}</div>${renderCells(tile, scenario)}<div class="cell-summary">${t('cellsLegend')} ${tail ? `<strong>${t('maskedTail', {count: tile.total_cells - tile.valid_cells})}</strong>` : t('noPadding')}</div>${step?.loop_axis ? `<div class="cell-summary" style="margin-top:8px">${escapeHtml(t('loopSlice', {axis: step.loop_axis, number: step.iteration + 1, valid: step.loop_valid, total: scenario.params[step.loop_tile]}))}</div>` : ''}</div>
    <div class="inspector-section inspector-bottom"><div class="inspector-label">${t('addressOffsets')}</div>${addresses || `<div class="cell-summary">${t('noAddress')}</div>`}</div>`;
}

function renderSource() {
  const scenario = currentScenario();
  const step = scenario && state.model.schema_version !== 2 ? activeStep() : null;
  const refKey = state.model.schema_version === 2 && state.model.status !== 'unknown' ? TraceView.currentRef() :
    (step?.source_ref || state.model.grid_source_ref || Object.keys(state.model.refs)[0]);
  const highlighted = state.highlightedRef || refKey;
  const line = state.model.refs[highlighted]?.line;
  $('source-path').textContent = state.model.source_path;
  $('source-explainer').innerHTML = `<div><strong>${t('sourceEvidence')}</strong>${t('sourceExplanation')}</div><div class="source-ref-list">${Object.entries(state.model.refs).slice(0, 7).map(([key, value]) => `<button data-source-ref="${escapeHtml(key)}">${escapeHtml(key)} · L${value.line}</button>`).join('')}</div>`;
  $('source-code').innerHTML = state.source.split('\n').map((text, index) => `<span class="code-line ${index + 1 === line ? 'highlight' : ''}" id="line-${index + 1}"><span class="line-no">${String(index + 1).padStart(2, '0')}</span>${escapeHtml(text || ' ')}</span>`).join('');
  const target = $(`line-${line}`);
  if (target && $('source-code').dataset.lastLine !== String(line)) {
    const viewport = $('source-code').parentElement;
    viewport.scrollTop += target.getBoundingClientRect().top - viewport.getBoundingClientRect().top - viewport.clientHeight / 2;
    $('source-code').dataset.lastLine = String(line);
  }
}

function render() {
  if (!state.model) return;
  $('backend-pill').textContent = backendLabel(state.model.backend);
  const unknown = state.model.status === 'unknown';
  const trace = state.model.schema_version === 2;
  $('visualization').hidden = unknown || trace;
  $('trace-visualization').hidden = unknown || !trace;
  $('metrics').hidden = unknown;
  $('unknown-panel').hidden = !unknown;
  if (unknown) {
    $('unknown-panel').className = 'unknown';
    $('unknown-panel').textContent = t('mappingUnknown', {note: displayLabel(state.model, 'note')});
    renderSource();
    return;
  }
  renderScenarioSwitch();
  if (trace) { TraceView.render(); renderSource(); return; }
  renderMetrics();
  renderGrid();
  renderGemmView();
  renderPlayback();
  renderInspector();
  renderSource();
}

async function loadModel(index) {
  pause();
  TraceView.pause();
  const [modelResponse, sourceResponse] = await Promise.all([
    fetch(`/models/${index}.json`), fetch(`/sources/${index}.txt`)
  ]);
  if (!modelResponse.ok || !sourceResponse.ok) throw new Error(t('loadError'));
  state.model = await modelResponse.json();
  state.source = await sourceResponse.text();
  state.scenario = state.model.scenarios[0]?.id || null;
  state.selected = state.model.schema_version === 2 ? null :
    (state.model.scenarios[0]?.tiles[0] ? tileKey(state.model.scenarios[0].tiles[0]) : null);
  state.tick = 0;
  state.highlightedRef = null;
  $('model-select').value = String(index);
  if (state.model.schema_version === 2 && state.model.status !== 'unknown') {
    TraceView.init(state.model, currentScenario(), (ref, scroll) => {
      state.highlightedRef = ref;
      renderSource();
      if (scroll) $('source-panel').scrollIntoView({behavior:'smooth', block:'start'});
    }, () => state.language);
  }
  render();
}

async function initialize() {
  try {
    const response = await fetch('/catalog.json');
    if (!response.ok) throw new Error(t('catalogError'));
    state.catalog = await response.json();
    renderCatalog();
    await loadModel(0);
  } catch (error) {
    $('unknown-panel').hidden = false;
    $('unknown-panel').className = 'unknown';
    $('unknown-panel').textContent = error.message;
    $('visualization').hidden = true;
    $('metrics').hidden = true;
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
});
$('gemm-matrices').addEventListener('click', (event) => {
  const button = event.target.closest('[data-source-ref]');
  if (!button) return;
  state.highlightedRef = button.dataset.sourceRef;
  renderSource();
  $('source-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
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
  state.highlightedRef = button.dataset.sourceRef;
  renderSource();
  $('source-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
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
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); play(); }
  if (event.code === 'ArrowRight') { event.preventDefault(); pause(); seek(state.tick + 1); }
  if (event.code === 'ArrowLeft') { event.preventDefault(); pause(); seek(state.tick - 1); }
});
applyLanguage();
initialize();
