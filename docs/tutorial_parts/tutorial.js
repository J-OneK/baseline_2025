'use strict';
(() => {
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const DATA = JSON.parse($('#tutorial-data').textContent);
  const chapters = $$('.chapter');
  const storageKey = 'taac-masterclass-' + DATA.commit.slice(0, 7);
  let completed = new Set();
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (value, digits = 5) => Number(value).toFixed(digits);
  const notify = value => {
    $('#status').textContent = value;
    $('#status').classList.add('show');
    clearTimeout(notify.timer);
    notify.timer = setTimeout(() => $('#status').classList.remove('show'), 2000);
  };
  try {
    const state = JSON.parse(localStorage.getItem(storageKey) || '{}');
    completed = new Set(state.completed || []);
    if (state.dark) document.body.classList.add('dark');
  } catch (_) { /* Reading remains available when file storage is disabled. */ }
  const save = () => {
    try { localStorage.setItem(storageKey, JSON.stringify({completed:[...completed], dark:document.body.classList.contains('dark')})); } catch (_) {}
  };
  const updateProgress = () => {
    for (const chapter of chapters) {
      const done = completed.has(chapter.id);
      const nav = $(`.toc a[href="#${chapter.id}"]`);
      if (nav) nav.classList.toggle('done', done);
      const button = $('.mark-complete', chapter);
      if (button) {
        button.classList.toggle('completed', done);
        button.setAttribute('aria-pressed', String(done));
        button.textContent = done ? '✓ 已理解本章' : '标记本章已理解';
      }
    }
    const count = chapters.filter(chapter => completed.has(chapter.id)).length;
    $('#progress-text').textContent = `${count} / ${chapters.length} 章已理解`;
    $('#progress-fill').style.width = `${count / chapters.length * 100}%`;
  };
  updateProgress();
  $$('.mark-complete').forEach(button => button.addEventListener('click', () => {
    const id = button.closest('.chapter').id;
    completed.has(id) ? completed.delete(id) : completed.add(id);
    updateProgress(); save();
  }));
  $('#theme-toggle').addEventListener('click', () => {document.body.classList.toggle('dark');save();});
  $('#source-toggle').addEventListener('click', () => {
    const snippets = $$('.source-snippet');
    const open = snippets.filter(item => item.open).length < snippets.length / 2;
    snippets.forEach(item => item.open = open);
    $('#source-toggle').textContent = open ? '折叠源码' : '展开源码';
  });
  $('#print').addEventListener('click', () => window.print());
  $('#menu').addEventListener('click', () => {
    const rail = $('.rail'); rail.classList.toggle('open');
    $('#menu').setAttribute('aria-expanded', String(rail.classList.contains('open')));
  });
  const searchable = chapters.map(chapter => ({id:chapter.id,text:chapter.textContent.toLowerCase()}));
  $('#search').addEventListener('input', event => {
    const q = event.target.value.trim().toLowerCase();
    let found = 0;
    searchable.forEach(item => {
      const match = !q || item.text.includes(q);
      $(`.toc a[href="#${item.id}"]`).hidden = !match;
      if (match) found++;
    });
    $('#search-status').textContent = q ? `${found} 个章节包含此关键词` : '搜索内容、函数名或公式 · Ctrl / ⌘ K';
  });
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {event.preventDefault();$('#search').focus();$('.rail').classList.add('open');}
    if (event.key === 'Escape') {$('.rail').classList.remove('open');$('#menu').setAttribute('aria-expanded','false');}
  });
  document.addEventListener('click', event => {
    const jump = event.target.closest('a[href^="#"]');
    if (!jump) return;
    const id = decodeURIComponent(jump.getAttribute('href').slice(1));
    const target = document.getElementById(id);
    if (!target) return;
    let ancestor = target;
    while (ancestor) {if (ancestor.tagName === 'DETAILS') ancestor.open = true; ancestor = ancestor.parentElement;}
    if (jump.closest('.toc')) {$('.rail').classList.remove('open');$('#menu').setAttribute('aria-expanded','false');}
  });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting);
      if (visible.length) {
        $$('.toc a').forEach(a => a.classList.remove('active'));
        const a = $(`.toc a[href="#${visible[0].target.id}"]`); if (a) a.classList.add('active');
      }
    }, {rootMargin:'-5% 0px -75% 0px',threshold:0});
    chapters.forEach(chapter => observer.observe(chapter));
  }
  $$('.copy-btn').forEach(button => button.addEventListener('click', async event => {
    event.preventDefault(); event.stopPropagation();
    const code = document.getElementById(button.dataset.copy);
    const clone = code.cloneNode(true); $$('.ln', clone).forEach(node => node.remove());
    const value = clone.textContent;
    try {await navigator.clipboard.writeText(value);}
    catch (_) {
      const input = document.createElement('textarea'); input.value = value; input.style.position = 'fixed'; input.style.left = '-9999px';
      document.body.appendChild(input); input.select(); const ok = document.execCommand('copy'); input.remove();
      if (!ok) {notify('浏览器未允许复制，请选中代码后复制');return;}
    }
    notify('源码已复制（不含行号）');
  }));

  // These small browser calculations mirror the displayed formulas. Python
  // source and recorded checkpoint results remain embedded separately.
  const cleanSplit = (events, minHistory, maxHistory = 10) => {
    const clean = events.filter(e => e && e.item_id != null && e.item_id > 0 && e.timestamp != null)
      .map((e,i) => ({...e,order:i})).sort((a,b) => a.timestamp - b.timestamp || a.order - b.order).slice(-maxHistory);
    const seen = new Set(), eligible = [];
    clean.forEach((e,pos) => {
      const cut = clean.findIndex(other => other.timestamp >= e.timestamp);
      if (e.action_type === 1 && !seen.has(e.item_id) && cut >= minHistory) eligible.push([pos,cut]);
      seen.add(e.item_id);
    });
    if (eligible.length < 2) return null;
    const test = eligible[eligible.length-1];
    const earlier = eligible.slice(0,-1).filter(pair => clean[pair[0]].timestamp < clean[test[0]].timestamp);
    if (!earlier.length) return null;
    const val = earlier[earlier.length-1];
    return {items:clean.map(e=>e.item_id),actions:clean.map(e=>e.action_type===0?1:e.action_type===1?2:3),
      times:clean.map(e=>e.timestamp),train_cut:val[1],val_cut:val[1],test_cut:test[1],val_pos:val[0],test_pos:test[0]};
  };
  const renderSplit = () => {
    const minimum = Number($('#split-min').value); $('#split-min-value').textContent = minimum;
    const events = DATA.demo.raw.first_user.seq.map(e => ({...e}));
    if ($('#split-mode').value === 'ties') events[7].timestamp = 90;
    if ($('#split-mode').value === 'seen') events[3].item_id = 6;
    const result = cleanSplit(events, minimum);
    $('#split-output').dataset.valid = String(Boolean(result));
    if (!result) {$('#split-output').innerHTML = '<strong>返回 None</strong><p>当前条件找不到两个时间严格先后的合格首次出现点击；调用方会跳过此用户。可以降低最少历史事件数，观察早期点击是否成为验证目标。</p>';return;}
    const target = split => result.items[result[split+'_pos']];
    const history = split => result.items.slice(0,result[split+'_cut']);
    $('#split-output').innerHTML = `<div class="table-wrap"><table><thead><tr><th>阶段</th><th>可见历史</th><th>目标 / 位置</th></tr></thead><tbody>
      <tr><td>训练</td><td>[${history('train')}]</td><td>在前缀内构造移位监督</td></tr>
      <tr><td>验证</td><td>[${history('val')}]</td><td>广告 ${target('val')} · pos=${result.val_pos}, cut=${result.val_cut}</td></tr>
      <tr><td>测试</td><td>[${history('test')}]</td><td>广告 ${target('test')} · pos=${result.test_pos}, cut=${result.test_cut}</td></tr></tbody></table></div>
      <p>内部动作编码：[${result.actions}]。只有严格早于目标的事件进入当前输入。</p>`;
    $('#split-output').dataset.valCut = result.val_cut;
    $('#split-output').dataset.testCut = result.test_cut;
  };
  ['#split-min','#split-mode'].forEach(s => $(s).addEventListener('input',renderSplit));renderSplit();

  const renderAttention = () => {
    const split = $('#attention-split').value, sample = DATA.demo.trace.samples[split];
    const kinds = sample.kinds, length = kinds.length;
    const query = Math.min(Number($('#attention-query').value), length-1);
    $('#attention-query-value').textContent = query;
    const labels = kinds.map((kind,i)=>kind===2?'U':kind===1?String(sample.items[i]):'·');
    const isAllowed = (i,j) => (j<=i && kinds[j]!==0) || (kinds[i]===0 && i===j);
    $('#attention-tokens').innerHTML = labels.map((label,i)=>`<span class="token ${kinds[i]===0?'pad':kinds[i]===2?'user':''} ${i===query?'target':''}">${label}<small>位置 ${i}</small></span>`).join('');
    let html = '<table class="heatmap" aria-label="因果注意力允许矩阵，行是query列是key"><thead><tr><th>Q↓K→</th>'+labels.map((label,i)=>`<th>${i}<br>${label}</th>`).join('')+'</tr></thead><tbody>';
    kinds.forEach((_,i) => {html += `<tr class="${i===query?'selected':''}"><th>${i} · ${labels[i]}</th>`+kinds.map((__,j)=>`<td class="${isAllowed(i,j)?'allow':'block'}" title="query ${i} → key ${j}">${isAllowed(i,j)?'1':'·'}</td>`).join('')+'</tr>';});
    html+='</tbody></table>';
    const readable=kinds.map((_,j)=>isAllowed(query,j)?`${j}:${labels[j]}`:null).filter(Boolean);
    html+=`<p>位置 ${query} 可读取：${readable.join('、')}。${kinds[query]===0?'这是 PAD query，只允许关注自身以避免全屏蔽 softmax，输出随后乘零。':'每个有效位置可读自己及更早的有效输入。'}</p>`;
    if(split==='train') html+='<p>主角唯一监督在输入位置 7：这里是广告 1，目标为点击广告 2。query=7 不得读取位置 8 的广告 2。</p>';
    $('#attention-output').innerHTML=html;
  };
  ['#attention-split','#attention-query'].forEach(s=>$(s).addEventListener('input',renderAttention));renderAttention();

  const renderLoss = () => {
    const positive=Number($('#loss-positive').value),negative=Number($('#loss-negative').value),k=Number($('#loss-k').value),tau=Number($('#loss-tau').value);
    $('#loss-positive-value').textContent=fmt(positive,1);$('#loss-negative-value').textContent=fmt(negative,1);$('#loss-tau-value').textContent=fmt(tau,1);
    const p=1/(1+k*Math.exp((negative-positive)/tau)), loss=-Math.log(p), singleNegative=(1-p)/k;
    $('#loss-output').innerHTML=`<div class="cards"><div class="card"><strong>${fmt(p,6)}</strong><small>正例 softmax 概率</small></div><div class="card"><strong>${fmt(loss,6)}</strong><small>−log(p)，单位置损失</small></div><div class="card"><strong>${fmt((p-1)/tau,6)}</strong><small>∂L/∂s⁺，梯度下降会提高正例分数</small></div></div>
      <div class="mini-bars"><div class="mini-bar positive"><span>正例概率</span><i style="width:${Math.max(.2,p*100)}%"></i><span>${fmt(p*100,2)}%</span></div><div class="mini-bar"><span>负例总概率</span><i style="width:${(1-p)*100}%"></i><span>${fmt((1-p)*100,2)}%</span></div></div>
      <p>单个负例的概率为 ${fmt(singleNegative,6)}，分数梯度为 ${fmt(singleNegative/tau,6)}。当正负分数全相等时，损失为 ln(${k+1})=${fmt(Math.log(k+1),6)}。</p>
      <p>本控件让所有负例使用同一分数以方便手算；真实模型的 64 个负例通常各有不同分数。这一 softmax 是采样候选内的相对分布。</p>`;
    $('#loss-output').dataset.probability=p;$('#loss-output').dataset.loss=loss;
  };
  ['#loss-positive','#loss-negative','#loss-k','#loss-tau'].forEach(s=>$(s).addEventListener('input',renderLoss));renderLoss();

  const renderMetric = () => {
    const rank=Math.max(1,Math.min(100,Number($('#metric-rank').value)||1)),k=Number($('#metric-k').value);
    $('#metric-k-value').textContent=k;
    $('#metric-overlap').max=k;
    const overlap=Math.min(k,Number($('#metric-overlap').value));$('#metric-overlap').value=overlap;$('#metric-overlap-value').textContent=overlap;
    const exact=Array.from({length:k},(_,i)=>i+1),approx=exact.slice(0,overlap).concat(Array.from({length:k-overlap},(_,i)=>1001+i));
    const hit=rank<=k?1:0,ndcg=hit?1/Math.log2(rank+1):0,annHit=approx.includes(rank)?1:0;
    $('#metric-output').innerHTML=`<p>目标广告 ID=${rank}。精确列表：[${exact}]；近似列表：[${approx}]。</p><div class="cards"><div class="card"><strong>${hit}</strong><small>精确检索 HR@${k} / 单目标 Recall@${k}</small></div><div class="card"><strong>${fmt(ndcg,6)}</strong><small>精确检索 NDCG@${k}</small></div><div class="card"><strong>${fmt(overlap/k*100,1)}%</strong><small>ANN 与精确列表重合率</small></div></div><p>Precision@${k}=${fmt(hit/k,4)}；近似列表对真实目标的 HR@${k}=${annHit}。重合率高与真实目标命中是两件需要分别测量的事；当前主项目固定 K=10。</p>`;
    $('#metric-output').dataset.ndcg=ndcg;$('#metric-output').dataset.hit=hit;
  };
  ['#metric-rank','#metric-k','#metric-overlap'].forEach(s=>$(s).addEventListener('input',renderMetric));renderMetric();

  const renderMips = () => {
    const q=[Number($('#mips-qx').value)||0,Number($('#mips-qy').value)||0],r2=10;
    const vectors=[[1,1],[3,0],[0,1]];
    const rows=vectors.map((v,i)=>{
      const norm=v[0]**2+v[1]**2, extra=Math.sqrt(r2-norm),score=q[0]*v[0]+q[1]*v[1];
      const distance=(q[0]-v[0])**2+(q[1]-v[1])**2+extra**2;
      const recovered=(q[0]**2+q[1]**2+r2-distance)/2;
      return {i,v,extra,score,distance,recovered};
    });
    const sorted=rows.slice().sort((a,b)=>b.score-a.score).map(r=>String.fromCharCode(65+r.i));
    $('#mips-output').innerHTML=`<div class="table-wrap"><table><thead><tr><th>广告</th><th>v</th><th>补的第3维</th><th>q·v</th><th>升维平方距离</th><th>还原内积</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${String.fromCharCode(65+r.i)}</td><td>(${r.v})</td><td>${fmt(r.extra,4)}</td><td>${fmt(r.score,4)}</td><td>${fmt(r.distance,4)}</td><td>${fmt(r.recovered,4)}</td></tr>`).join('')}</tbody></table></div><p>固定 R²=10；查询 q′=(${q},0)。内积从大到小与距离从小到大的顺序相同：${sorted.join(' → ')}。同分项的具体先后不由这条恒等式规定。</p>`;
    $('#mips-output').dataset.error=Math.max(...rows.map(r=>Math.abs(r.score-r.recovered)));
  };
  ['#mips-qx','#mips-qy'].forEach(s=>$(s).addEventListener('input',renderMips));renderMips();

  // Useful for checking that browser teaching widgets agree with Python traces.
  window.TAAC_TUTORIAL = {cleanSplit, data:DATA, renderSplit, renderAttention, renderLoss, renderMetric, renderMips};
})();
