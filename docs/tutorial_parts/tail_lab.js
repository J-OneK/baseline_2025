'use strict';
(() => {
  const main = Object.freeze([10, 20, 30, 40, 50, 60, 70, 80]);
  const frequency = Object.freeze({10:80, 20:70, 30:0, 40:3, 50:100, 60:60, 70:20, 80:9, 90:1, 100:4, 110:2});
  const trainingClicks = Object.freeze({10:20, 20:10, 30:0, 40:1, 50:20, 60:10, 70:5, 80:2, 90:0, 100:0, 110:0});
  const rawProposals = Object.freeze([60, 90, 100, 90, 110]);
  const seen = Object.freeze([100]);
  const prefix = 2;

  // Deliberately has no target argument: targets are evaluation-only labels.
  const fuseTailToy = quota => {
    if (!Number.isInteger(quota) || quota < 0 || quota > 3) throw new RangeError('配额必须为0～3的整数');
    const used = new Set([...main, ...seen]), proposals = [];
    for (const item of rawProposals) {
      if (frequency[item] >= 1 && frequency[item] <= 5 && trainingClicks[item] === 0 && !used.has(item)) {
        proposals.push(item); used.add(item);
      }
    }
    const protectedIds = main.filter((item, index) => index < prefix || frequency[item] <= 5);
    const removable = main.map((item, index) => ({item, index})).reverse()
      .filter(({item, index}) => index >= prefix && frequency[item] > 5);
    const applied = Math.min(quota, proposals.length, removable.length);
    const removed = removable.slice(0, applied).map(entry => entry.item);
    const removedSet = new Set(removed);
    const fused = main.filter(item => !removedSet.has(item)).concat(proposals.slice(0, applied));
    return {main:[...main], proposals, removed, fused, applied, protectedIds};
  };

  const root = document.getElementById('tail-lab');
  const render = () => {
    if (!root) return null;
    const quota = Number(document.getElementById('tail-quota').value);
    const target = Number(document.getElementById('tail-target').value);
    const result = fuseTailToy(quota);
    const before = Number(result.main.includes(target)), after = Number(result.fused.includes(target));
    const unchanged = result.main.slice(0, prefix).every((item, index) => item === result.fused[index]);
    document.getElementById('tail-quota-value').textContent = quota;
    const row = (label, items) => `<tr><th scope="row">${label}</th><td><code>[${items.join(', ')}]</code></td></tr>`;
    const output = document.getElementById('tail-output');
    output.innerHTML = `<div class="table-wrap"><table><thead><tr><th>步骤</th><th>广告local ID，按顺序排列</th></tr></thead><tbody>
      ${row('主通道8候选', result.main)}${row('必须保留', result.protectedIds)}
      ${row('过滤后的新提议', result.proposals)}${row('从主列表末尾移除', result.removed)}
      ${row('融合后仍为8候选', result.fused)}</tbody></table></div>
      <p>配额上限=${quota}，实际替换=${result.applied}。${quota > result.applied ? '合法新提议不足，配额不是必须填满的任务。' : '从末尾等量替换未保护候选，其余原顺序保留。'}</p>
      <div class="cards"><div class="card"><strong>${before} → ${after}</strong><small>事后目标=${target}：单目标Recall@8</small></div>
      <div class="card"><strong>${result.fused.length}</strong><small>最终候选数，固定预算</small></div>
      <div class="card"><strong>${unchanged ? '保持' : '变化'}</strong><small>原Top-2身份与顺序</small></div></div>
      <p>${after > before ? '目标长尾被补回，候选覆盖增加。' : after < before ? '目标原本在主列表中，替换后丢失：长尾新增可能伴随其他目标损失。' : result.protectedIds.includes(target) ? '该目标受保护，因此保持；这是融合规则的保证。' : '这次配额下目标覆盖未变化。'}目标只用于这段事后解释，不改变上方候选列表。</p>`;
    output.dataset.fused = JSON.stringify(result.fused);
    output.dataset.applied = result.applied;
    output.dataset.beforeHit = before;
    output.dataset.afterHit = after;
    output.dataset.protectedUnchanged = String(unchanged);
    return result;
  };
  window.TAAC_TAIL_LAB = {fuseTailToy, render, frequency, trainingClicks};
  if (!root) return;
  ['tail-quota', 'tail-target'].forEach(id => document.getElementById(id).addEventListener('input', render));
  render();
})();
