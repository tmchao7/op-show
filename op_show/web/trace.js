/* Source-backed logical QKV trace. The level is a DAG frontier, not device time. */
const TraceView = (() => {
  let timer = null;
  let tab = 'producer';
  let sourceRank = 0;
  let targetRank = 0;
  let tensor = 'Q';
  let selected = null;
  let tick = 0;
  let eventMap = new Map();
  let taskMap = new Map();
  let model = null;
  let scenario = null;
  let sourceJump = () => {};
  let lang = () => 'zh';
  const $t = (id) => document.getElementById(id);
  const h = (value) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const words = {
    zh: {pending:'待执行',active:'进行中',done:'完成',direct:'本地直写',peer_store:'远端写入',load:'加载 Q/K/V',fence:'fence',notify:'notify',wait:'wait',consume:'consume',output_store:'输出 Q/K/V',masked:'masked',skip:'跳过',valid:'有效',source:'来源',target:'目标',group:'分组',block:'Block',producer:'Producer',consumer:'Consumer',slot:'信号槽',offset:'元素偏移',head:'Head 范围',tokens:'有效 token',jump:'定位源码',play:'播放',pause:'暂停',example:'示例参数',notMeasured:'非设备测量值',none:'无',route:'路径',peer:'目标 peer_mem',output:'目标 qkv_out',input:'输入 q/k/v',wave:'Wave',task:'任务',pid:'物理 PID',claim:'认领序号',current:'当前事件',all:'全部'},
    en: {pending:'Pending',active:'In progress',done:'Complete',direct:'Local direct write',peer_store:'Remote store',load:'Load Q/K/V',fence:'fence',notify:'notify',wait:'wait',consume:'consume',output_store:'Write Q/K/V',masked:'masked',skip:'Skipped',valid:'valid',source:'Source',target:'Target',group:'Group',block:'Block',producer:'Producer',consumer:'Consumer',slot:'Signal slot',offset:'Element offset',head:'Head range',tokens:'Valid tokens',jump:'View source',play:'Play',pause:'Pause',example:'Example inputs',notMeasured:'not device measurements',none:'None',route:'Route',peer:'Target peer_mem',output:'Target qkv_out',input:'Input q/k/v',wave:'Wave',task:'Task',pid:'Physical PID',claim:'Claim index',current:'Current event',all:'All'}
  };
  const w = key => words[lang()][key] || key;
  const task = () => taskMap.get(selected) || scenario?.tasks[0];
  const ev = id => eventMap.get(id);
  const done = id => ev(id) && ev(id).level < tick;
  function status(item) {
    const final = item.route === 'local' ? `${item.id}:direct` : `${item.id}:output_store`;
    if (done(final)) return 'done';
    if (item.event_ids.some(done)) return 'active';
    return 'pending';
  }
  const tileClass = item => `trace-cell g${item.head_group} ${status(item)} ${item.id === selected ? 'selected' : ''} ${item.valid_tokens < item.total_tokens ? 'masked' : ''} ${item.route === 'local' ? 'local' : ''}`;
  const label = item => `R${item.source_rank} → R${item.target_rank} · B${item.block_id} · G${item.head_group}`;
  function cell(item, content, extra='') {
    return `<button type="button" class="${tileClass(item)}" data-task="${h(item.id)}" aria-label="${h(label(item))}, ${w(status(item))}" title="${h(label(item))}">${content}${extra}</button>`;
  }
  function pause() { if (timer) clearInterval(timer); timer = null; const btn=$t('trace-play'); btn.innerHTML=`▶ <span>${w('play')}</span>`; btn.setAttribute('aria-label',w('play')); }
  function seek(value) { const previous=currentRef(); tick=Math.max(0,Math.min(scenario.total_ticks,Number(value)||0)); if(tick===scenario.total_ticks) pause(); render(); if(currentRef()!==previous) sourceJump(currentRef(),false); }
  function play() {
    if(timer) { pause(); return; }
    if(tick===scenario.total_ticks) tick=0;
    const btn=$t('trace-play'); btn.innerHTML=`Ⅱ <span>${w('pause')}</span>`; btn.setAttribute('aria-label',w('pause'));
    timer=setInterval(()=>seek(tick+1),420);
    render();
  }
  function select(id) {
    const item=taskMap.get(id); if(!item) return;
    selected=id; sourceRank=item.source_rank; targetRank=item.target_rank;
    render(); sourceJump(currentRef(),false);
  }
  function currentRef() {
    const item=task(); if(!item) return 'role';
    const events=[...item.event_ids,item.fence_id].map(ev).filter(Boolean).filter(e=>e.level<tick).sort((a,b)=>b.level-a.level);
    return events[0]?.source_ref || 'producer_loop';
  }
  function rankPickers() {
    const n=scenario.params.rank_size;
    for(const [id,value] of [['trace-source-rank',sourceRank],['trace-target-rank',targetRank]]) {
      const node=$t(id);
      node.innerHTML=Array.from({length:n},(_,i)=>`<option value="${i}">R${i}</option>`).join(''); node.value=String(value);
    }
    $t('trace-tensor').value=tensor;
  }
  function renderProducer() {
    const s=scenario, p=s.params, rows=[];
    rows.push('<div class="trace-grid-label"></div>');
    for(let target=0;target<p.rank_size;target++) for(let group=0;group<s.head_groups;group++) rows.push(`<div class="trace-grid-label group g${group}">R${target}<small>G${group}</small></div>`);
    for(let block=0;block<s.num_blocks;block++) {
      rows.push(`<div class="trace-row-label">B${block}<small>W${Math.floor(block/p.s_wave)}</small></div>`);
      for(let target=0;target<p.rank_size;target++) for(let group=0;group<s.head_groups;group++) {
        const item=taskMap.get(`s${sourceRank}-t${target}-b${block}-g${group}`);
        rows.push(cell(item,`<b>G${group}</b><small>${item.route==='local'?'↘':'→'} R${target}</small>`,item.valid_tokens<p.block_s?'<i class="trace-tail">◩</i>':''));
      }
    }
    $t('trace-producer').innerHTML=`<div class="trace-matrix producer-matrix" style="--cols:${p.rank_size*s.head_groups}">${rows.join('')}</div>`;
  }
  function renderConsumer() {
    const s=scenario,p=s.params,rows=['<div class="trace-grid-label"></div>'];
    for(let group=0;group<s.head_groups;group++) rows.push(`<div class="trace-grid-label group g${group}">G${group}</div>`);
    for(let source=0;source<p.rank_size;source++) for(let block=0;block<s.num_blocks;block++) {
      rows.push(`<div class="trace-row-label">R${source} · B${block}<small>W${Math.floor(block/p.s_wave)}</small></div>`);
      for(let group=0;group<s.head_groups;group++) {
        const item=taskMap.get(`s${source}-t${targetRank}-b${block}-g${group}`);
        rows.push(cell(item,`<b>${tensor} · G${group}</b><small>${item.route==='local'?w('direct'):`R${source} → R${targetRank}`}</small>`,item.valid_tokens<p.block_s?'<i class="trace-tail">◩</i>':''));
      }
    }
    $t('trace-consumer').innerHTML=`<div class="trace-matrix consumer-matrix" style="--cols:${s.head_groups}">${rows.join('')}</div>`;
  }
  function renderClaims() {
    const s=scenario, p=s.params, current=task(), wave=current.wave;
    const rows=[];
    for(let logical=0;logical<p.producer_cores;logical++) {
      const owned=s.tasks.filter(item=>item.source_rank===sourceRank && item.wave===wave && item.producer_logical===logical).sort((a,b)=>a.flat_idx-b.flat_idx);
      rows.push(`<div class="trace-claim-row"><div class="trace-claim-pid"><b>P${String(logical).padStart(2,'0')}</b><span>pid ${s.producer_pids[logical]}</span></div><div class="trace-claim-items">${owned.map(item=>cell(item,`<b>${item.flat_idx}</b><small>B${item.block_id} · R${item.target_rank} · G${item.head_group}</small>`)).join('')}</div></div>`);
    }
    $t('trace-claims').innerHTML=rows.join('');
  }
  function renderPath() {
    const x=task(), p=scenario.params, route=x.route==='local';
    const path=[
      {title:w('input'),value:`q/k/v [B${x.block_id}, R${x.target_rank}, G${x.head_group}]`,sub:`${w('offset')} ${x.input_offset}`,kind:'load'},
      {title:`P${String(x.producer_logical).padStart(2,'0')} / pid ${x.producer_pid}`,value:`flat_idx ${x.flat_idx}`,sub:`${w('wave')} ${x.wave}`,kind:'load'},
      route?{title:w('direct'),value:`${tensor}_out [R${x.source_rank}, B${x.block_id}, G${x.head_group}]`,sub:`${w('offset')} ${x.output_offset}`,kind:'direct'}:{title:w('peer'),value:`peer_mem(R${x.target_rank}, B${x.block_id}, G${x.head_group})`,sub:`${tensor} ${w('offset')} ${x.peer_offsets[tensor.toLowerCase()]}`,kind:'peer_store'},
      ...(route?[]:[
        {title:'fence → notify',value:`slot ${x.signal_slot} · +${x.signal_offset}`,sub:`${w('wave')} ${x.wave}`,kind:'notify'},
        {title:`C${String(x.consumer_logical).padStart(2,'0')} / pid ${x.consumer_pid}`,value:'wait → consume',sub:`flat_idx ${x.consumer_flat_idx}`,kind:'wait'},
        {title:w('output'),value:`${tensor}_out [R${x.source_rank}, B${x.block_id}, G${x.head_group}]`,sub:`${w('offset')} ${x.output_offset}`,kind:'output_store'}
      ])
    ];
    $t('trace-path').innerHTML=`<div class="trace-path-summary"><span>${h(label(x))}</span><strong>${w(route?'direct':'peer_store')}</strong></div><div class="trace-path-steps">${path.map((step,i)=>`<div class="trace-path-node ${done(`${x.id}:${step.kind}`)?'done':''}"><span class="trace-path-number">${i+1}</span><div><span class="trace-path-title">${h(step.title)}</span><strong>${h(step.value)}</strong><small>${h(step.sub)}</small></div></div>`).join('<span class="trace-path-arrow">↓</span>')}</div>`;
  }
  function renderPlayer() {
    const s=scenario, current=task();
    $t('trace-percent').textContent=`${Math.round(s.events.filter(e=>e.level<tick).length/s.events.length*100)}%`;
    $t('trace-tick').textContent=`${tick} / ${s.total_ticks}`;
    $t('trace-scrubber').max=s.total_ticks; $t('trace-scrubber').value=tick;
    $t('trace-waves').innerHTML=s.waves.map(wave=>`<button type="button" data-wave="${wave.index}" class="${current.wave===wave.index?'selected':''}">Wave ${wave.index+1}<small>B${wave.block_start}–B${wave.block_end-1}</small></button>`).join('');
    const ids=[...current.event_ids]; if(current.fence_id) ids.push(current.fence_id);
    const events=ids.map(ev).filter(Boolean).sort((a,b)=>a.level-b.level);
    $t('trace-event-strip').innerHTML=events.map(e=>`<button type="button" data-event="${h(e.id)}" class="${e.level<tick?'done':e.level===tick?'active':''}"><b>${h(w(e.kind))}</b><small>${e.level+1} / ${s.total_ticks}</small></button>`).join('');
  }
  function renderDetail() {
    const x=task(), p=scenario.params, rows=[
      [w('source'),`R${x.source_rank}`],[w('target'),`R${x.target_rank}`],[w('block'),`B${x.block_id} · [${x.block_id*p.block_s}, ${x.block_id*p.block_s+x.valid_tokens})`],
      [w('group'),`G${x.head_group} · head [${x.head_start}, ${x.head_end})`],[w('tokens'),`${x.valid_tokens}/${p.block_s}`],
      [w('producer'),`P${String(x.producer_logical).padStart(2,'0')} · pid ${x.producer_pid} · flat_idx ${x.flat_idx}`],
      [w('consumer'),x.route==='local'?w('direct'):`C${String(x.consumer_logical).padStart(2,'0')} · pid ${x.consumer_pid} · flat_idx ${x.consumer_flat_idx}`],
      [w('route'),w(x.route==='local'?'direct':'peer_store')],[w('slot'),x.route==='local'?w('none'):`${x.signal_slot} × ${p.signal_stride} = ${x.signal_offset}`],
      ['input strides',`(${p.stride_a_s}, ${p.stride_a_h}, ${p.stride_a_d})`],
      ['output strides',`(${p.stride_out_s}, ${p.stride_out_h}, ${p.stride_out_d})`]
    ];
    $t('trace-detail-title').textContent=label(x);
    const sample=Array.from({length:p.block_s},(_,i)=>`<span class="${i<x.valid_tokens?'':'masked'}" title="token ${x.block_id*p.block_s+i}: ${i<x.valid_tokens?w('valid'):w('masked')}"></span>`).join('');
    const formulas=scenario.formulas;
    const env={...p,block_id:x.block_id,source_rank:x.source_rank,target_rank:x.target_rank,
      head_group:x.head_group,head_start:x.head_start,buffer_num:scenario.num_blocks,
      head_groups:scenario.head_groups,peer_head_size:p.block_s*p.block_d,
      fused_heads:3*p.heads_per_rank,block_slot_size:p.rank_size*3*p.heads_per_rank*p.block_s*p.block_d,
      output_row:x.output_row,peer_q:x.peer_offsets?.q,signal_slot:x.signal_slot};
    function substitutions(expr) { return [...new Set(expr.match(/[a-zA-Z_][a-zA-Z_0-9]*/g)||[])].filter(key=>env[key]!==undefined).map(key=>`${key}=${env[key]}`).join(' · '); }
    const offsetCards=[
      ['input_offset',x.input_offset],['peer_q',x.peer_offsets?.q],['peer_k',x.peer_offsets?.k],['peer_v',x.peer_offsets?.v],['signal_slot',x.signal_slot],['signal_offset',x.signal_offset],['output_offset',x.output_offset]
    ].filter(([,v])=>v!==null && v!==undefined).map(([key,value])=>`<div class="formula-card"><div class="formula-title">${h(key)} · ${w('offset')} ${value}</div><div class="formula-expr">${h(formulas[key])}</div><div class="formula-result">${h(substitutions(formulas[key]))} → <strong>${value}</strong></div><button class="formula-source" data-ref="${h(scenario.formula_refs[key])}">${w('jump')} ↗</button></div>`).join('');
    $t('trace-detail-body').innerHTML=`<div class="trace-detail-grid"><div><div class="detail-table">${rows.map(([k,v])=>`<span class="key">${h(k)}</span><span class="value">${h(v)}</span>`).join('')}</div><h3>${w('tokens')} · ${x.valid_tokens}/${p.block_s}</h3><div class="trace-token-grid">${sample}</div></div><div class="trace-formulas">${offsetCards}</div></div>`;
  }
  function renderLegend() {
    $t('trace-note').textContent=`${model.note_zh && lang()==='zh'?model.note_zh:model.note} · ${w('example')}: R=${scenario.params.rank_size}, H=${scenario.params.heads_per_rank}, AIV=${scenario.params.total_cores}, S=${scenario.params.sequence_length}.`;
    $t('trace-legend').innerHTML=`${Array.from({length:scenario.head_groups},(_,i)=>`<span class="trace-group-key g${i}">G${i}</span>`).join('')}<span>◯ ${w('pending')}</span><span>● ${w('active')}</span><span>✓ ${w('done')}</span><span>◩ ${w('masked')}</span><span>↘ ${w('direct')}</span>`;
  }
  function render() {
    if(!scenario) return;
    rankPickers(); renderLegend(); renderProducer(); renderClaims(); renderPath(); renderConsumer(); renderPlayer(); renderDetail();
    document.querySelectorAll('[data-trace-tab]').forEach(node=>node.setAttribute('aria-selected',String(node.dataset.traceTab===tab)));
    document.querySelectorAll('[data-trace-view]').forEach(node=>node.classList.toggle('trace-tab-hidden',node.dataset.traceView!==tab));
  }
  function init(m,s,jump,language) {
    pause(); model=m; scenario=s; sourceJump=jump; lang=language;
    eventMap=new Map(s.events.map(e=>[e.id,e])); taskMap=new Map(s.tasks.map(x=>[x.id,x]));
    sourceRank=0; targetRank=0; tensor='Q'; selected=s.tasks[0]?.id; tick=0; render();
  }
  function scenarioChange(s) { pause(); scenario=s; eventMap=new Map(s.events.map(e=>[e.id,e])); taskMap=new Map(s.tasks.map(x=>[x.id,x])); selected=s.tasks[0]?.id; sourceRank=0;targetRank=0;tick=0;render(); }
  document.addEventListener('click', e=>{
    const taskButton=e.target.closest('#trace-visualization [data-task]'); if(taskButton) return select(taskButton.dataset.task);
    const tabButton=e.target.closest('[data-trace-tab]'); if(tabButton){tab=tabButton.dataset.traceTab;render();return;}
    const waveButton=e.target.closest('[data-wave]'); if(waveButton && scenario){const wave=Number(waveButton.dataset.wave); const item=scenario.tasks.find(x=>x.source_rank===sourceRank && x.target_rank===targetRank && x.wave===wave); if(item) selected=item.id; const first=Math.min(...scenario.events.filter(x=>x.wave===wave).map(x=>x.level)); pause();seek(first);return;}
    const eventButton=e.target.closest('[data-event]'); if(eventButton){pause();seek(ev(eventButton.dataset.event).level+1);sourceJump(ev(eventButton.dataset.event).source_ref,true);return;}
    const refButton=e.target.closest('#trace-detail-body [data-ref]'); if(refButton){sourceJump(refButton.dataset.ref,true);return;}
  });
  for(const id of ['trace-source-rank','trace-target-rank','trace-tensor']) $t(id).addEventListener('change',e=>{
    if(id==='trace-source-rank') sourceRank=Number(e.target.value);
    if(id==='trace-target-rank') targetRank=Number(e.target.value);
    if(id==='trace-tensor') tensor=e.target.value;
    if(id!=='trace-tensor') selected=`s${sourceRank}-t${targetRank}-b0-g0`;
    render();
  });
  $t('trace-play').addEventListener('click',play);
  $t('trace-reset').addEventListener('click',()=>{pause();seek(0);});
  $t('trace-prev').addEventListener('click',()=>{pause();seek(tick-1);});
  $t('trace-next').addEventListener('click',()=>{pause();seek(tick+1);});
  $t('trace-scrubber').addEventListener('input',e=>{pause();seek(e.target.value);});
  return {init,scenarioChange,render,pause,currentRef};
})();
