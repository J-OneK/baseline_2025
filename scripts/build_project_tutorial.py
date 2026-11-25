"""Build the offline, single-file Chinese project masterclass from real sources.

No web requests or model training are performed by this builder. The recorded
synthetic trace is generated separately by scripts/run_tutorial_demo.py.
"""

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
PARTS = REPO / 'docs/tutorial_parts'
TAIL_SNAPSHOT = PARTS / 'tail_recall_snapshot'
SOURCE_FILES = [
    'configs/main.json', 'scripts/run_main.py',
    'taac_retrieval/prepare.py', 'taac_retrieval/cohorts.py',
    'taac_retrieval/data.py', 'taac_retrieval/model.py', 'model.py',
    'taac_retrieval/objectives.py', 'taac_retrieval/train.py',
    'taac_retrieval/metrics.py', 'taac_retrieval/evaluate_checkpoint.py',
    'taac_retrieval/export.py', 'taac_retrieval/ann.py',
    'taac_retrieval/benchmark_ann.py', 'taac_retrieval/recommend.py',
    'taac_retrieval/study.py', 'taac_retrieval/objective_report.py',
    'scripts/run_objective_study.py',
    'taac_retrieval/coverage.py', 'tests/test_retrieval.py',
    'tests/test_objectives.py', 'tests/test_ann.py',
    'scripts/run_tutorial_demo.py', 'scripts/build_project_tutorial.py',
    'scripts/recommend_channel_rank.py', 'taac_retrieval/channel_ranker.py',
    'taac_retrieval/channel_features.py', 'taac_retrieval/group_recall.py',
    'scripts/recommend_sequence_rank.py', 'taac_retrieval/sequence_ranker.py',
    'scripts/build_channel_features.py', 'scripts/run_sequence_rank.py', 'configs/sequence_rank.json',
]
NAV = [
    ('00', 'overview', '整体架构与学习地图'),
    ('入口', 'workflow', 'run_main 与实际执行入口'),
    ('01', 'data', '原始数据与四层编号'),
    ('02', 'prepare', '清洗、切分与缓存'),
    ('03', 'dataset', '样本、移位与负例'),
    ('04', 'model', '模型、注意力与张量'),
    ('05', 'training', 'InfoNCE 与训练过程'),
    ('06', 'evaluation', '精确评估与指标'),
    ('07', 'retrieval', 'Faiss 与推荐出口'),
    ('08', 'experiments', '实验结果与证据范围'),
    ('08A', 'tail-recall', '新增：零点击长尾辅助召回'),
    ('08B', 'channel-ranking', '双通道召回与 DCN V2 排序'),
    ('08C', 'sequence-ranking', '联合排序与逐阶段数据'),
    ('09', 'demo', '真实执行：12用户完整轨迹'),
    ('10', 'debugging', '排查问题与理解验收'),
    ('11', 'glossary', '符号与术语速查'),
    ('12', 'sources', '嵌入源码与来源摘要'),
]

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_for_filename
except ImportError:
    highlight = None


def esc(value):
    return html.escape(str(value))


def source_id(path):
    return 'src-' + re.sub(r'[^a-zA-Z0-9_-]', '-', path)


def table(headers, rows):
    return ('<div class="table-wrap"><table><thead><tr>'
            + ''.join(f'<th>{esc(v)}</th>' for v in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def pre(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return '<pre><code>' + esc(value) + '</code></pre>'


class Builder:
    def __init__(self):
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        self.sources = {path: (REPO / path).read_text() for path in SOURCE_FILES}
        self.hashes = {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in SOURCE_FILES}
        self.research = json.loads((TAIL_SNAPSHOT / 'manifest.json').read_text())
        if (self.research['namespace'] != 'research/tail-recall/'
                or not re.fullmatch(r'[a-f0-9]{40}', self.research['commit'])):
            raise ValueError('Invalid research snapshot identity')
        prefix = self.research['namespace']
        for path, expected in self.research['files'].items():
            if Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('Invalid research snapshot path')
            content = (TAIL_SNAPSHOT / path).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Research snapshot digest mismatch: ' + path)
            self.sources[prefix + path] = content.decode('utf-8')
            self.hashes[prefix + path] = expected
        self.tail_metrics = json.loads(self.sources[prefix + 'reports/supplemental_tail_study/metrics.json'])
        self.tail_examples = json.loads(self.sources[prefix + 'reports/supplemental_tail_study/examples.json'])
        self.tail_verification = json.loads(self.sources[prefix + 'reports/supplemental_tail_study/verification.json'])
        if self.tail_metrics['status'] != 'completed_confirmation':
            raise ValueError('The tail teaching chapter requires completed confirmation evidence')
        for path, expected in self.tail_metrics['source_manifest'].items():
            if path in self.research['files'] and self.research['files'][path] != expected:
                raise ValueError('Snapshot disagrees with the experiment source manifest: ' + path)
        self.snippets = []

    def tail_tables(self):
        metrics = self.tail_metrics
        groups = metrics['confirmation']['groups']
        intervals = metrics['paired_intervals']
        labels = {'all': '全体', 'tail_1_5': '全部长尾', 'zero_click_tail': '零点击长尾', 'cold_0': '冷启动'}
        results = table(['新确认分组', '目标用户数', '主Recall@500', '融合Recall@500', '平均新增 / 丢失'], [
            [labels[key], str(intervals[key]['users']), f'{groups[key]["baseline_recall"]*100:.4f}%',
             f'{groups[key]["recall"]*100:.4f}%', f'{groups[key]["gained"]:.2f} / {groups[key]["lost"]:.2f}']
            for key in labels])
        example = self.tail_examples['seed_user_examples'][0]
        case = table(['实际记录字段', '来自确认产物的值'], [
            ['主模型seed / 内部用户行', f'{example["seed"]} / {example["user_row"]}'],
            ['目标广告local ID / RID', f'{example["target_local_id"]} / {example["target_rid"]}'],
            ['训练事件次数 / 点击次数', f'{example["training_events"]} / {example["training_clicks"]}'],
            ['原主通道排名', '前500以外'],
            ['辅助新候选中的排名', str(example['auxiliary_novel_rank'])],
            ['融合后排名', str(example['fused_rank'])],
            ['主Top-10是否保持', '是' if example['top10_unchanged'] else '否']])
        development = table(['开发方案', '整体Recall@500', '零点击尾Recall@500', '有新增的模型数', '通过门槛'], [
            [('<strong>' + esc(row['key']) + '（锁定）</strong>' if row['key'] == metrics['selected']['key'] else esc(row['key'])),
             f'{row["groups"]["all"]["recall"]*100:.4f}%',
             f'{row["groups"]["zero_click_tail"]["recall"]*100:.4f}%',
             f'{row["positive_seed_models"]} / 3', '是' if row['passed'] else '否']
            for row in metrics['development']])
        return dict(tail_results=results, tail_case=case, tail_protocol_results=development)

    def code(self, path, start, end, full=False):
        if path not in self.sources:
            raise ValueError(f'Source span is missing from the embedded source manifest: {path}')
        source = self.sources[path]
        lines = source.splitlines()
        end = len(lines) if end == '*' else int(end)
        start = int(start)
        if not (1 <= start <= end <= len(lines)):
            raise ValueError(f'Invalid source span: {path}:{start}:{end}, total {len(lines)}')
        index = len(self.snippets) + 1
        code_id = source_id(path) + '-code' if full else f'snippet-{index}-code'
        rendered = []
        lexer = get_lexer_for_filename(path) if highlight else None
        for number in range(start, end + 1):
            line = lines[number - 1]
            formatted = (highlight(line, lexer, HtmlFormatter(nowrap=True)).rstrip('\n')
                         if highlight and line else esc(line))
            anchor = f' id="{source_id(path)}-L{number}"' if full else ''
            rendered.append(f'<span class="source-line"{anchor}><span class="ln" aria-hidden="true">{number}</span>'
                            f'<span class="line-code">{formatted}</span>\n</span>')
        source_link = f'#{source_id(path)}-L{start}'
        block = (f'<summary>{esc(path)} · L{start}–{end}'
                 f'<button type="button" class="copy-btn" data-copy="{code_id}" aria-label="复制{esc(path)}源码">复制</button>'
                 + ('' if full else f' <a class="source-reference" href="{source_link}">完整源码 ↗</a>')
                 + f'</summary><pre class="source-code"><code id="{code_id}">'
                 + ''.join(rendered) + '</code></pre>')
        if full:
            return f'<details class="source-file" id="{source_id(path)}">{block}</details>'
        self.snippets.append(dict(path=path, start=start, end=end))
        return f'<details class="source-snippet" open>{block}</details>'

    def replace_code(self, text):
        return re.sub(r'\{\{code:([^:}]+):(\d+):(\d+|\*)\}\}',
                      lambda m: self.code(m[1], m[2], m[3]), text)


def diagram():
    boxes = [
        (28, 34, '原始Parquet', '序列 · 属性 · 内容'),
        (257, 34, 'prepare / Dataset', '合法历史 → 移位样本'),
        (486, 34, '共享编码与因果序列', '广告向量 v · 用户状态 q'),
        (28, 173, 'InfoNCE训练', '点击正例 + 64负例'),
        (257, 173, '验证选型 / 测试评估', '完整候选 · HR · NDCG'),
        (486, 173, '导出与Faiss', '历史过滤 → Top-10 OID'),
    ]
    content = ['<svg class="route-diagram" viewBox="0 0 720 291" role="img" aria-labelledby="flow-title">',
               '<title id="flow-title">原始数据、样本、模型、训练、评估、导出索引与推荐的流程图</title>',
               '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#93a8c6"/></marker></defs>',
               '<path d="M224 78 H250 M453 78 H479 M584 119 V150 H126 V166 M224 217 H250 M453 217 H479" fill="none" stroke="#93a8c6" stroke-width="1.5" marker-end="url(#arrow)"/>']
    for x, y, title, subtitle in boxes:
        content.append(f'<rect x="{x}" y="{y}" width="196" height="84" rx="9" fill="white" stroke="#d7e0ed"/>'
                       f'<text x="{x+16}" y="{y+31}" font-size="14" font-weight="600" fill="#203652">{title}</text>'
                       f'<text x="{x+16}" y="{y+56}" font-size="11" fill="#677c96">{subtitle}</text>')
    content.append('</svg>')
    return ''.join(content)


def laboratories():
    return {
        'prepare': '''<div class="lab" id="split-lab"><h3>动手：改变切分条件，先预测输出</h3><p>使用同一条十事件合成序列。默认应得到train_cut=5、val_pos=5、test_pos=8。试着给测试点击前一事件同样的时间戳，再观察pos与cut。</p><div class="controls"><label>最少历史 <input id="split-min" type="range" min="0" max="9" step="1" value="5"><output class="value" id="split-min-value">5</output></label><label>序列条件 <select id="split-mode"><option value="normal">原始十事件</option><option value="ties">广告7与测试点击同时间</option><option value="seen">广告6更早已曝光</option></select></label></div><div class="lab-output" id="split-output" aria-live="polite"></div><p class="source-note">浏览器复算clean_and_split的当前逻辑；Python源码与实际输出仍以上面的嵌入快照为准。</p></div>''',
        'model': '''<div class="lab" id="attention-lab"><h3>动手：看见一个位置真正允许读取哪些输入</h3><p>蓝色1表示允许，灰色点表示禁止。选择训练位置7：它读取广告1及更早输入，预测的广告2位于后面的输入位置8，必须被屏蔽。</p><div class="controls"><label>样本 <select id="attention-split"><option value="train">训练</option><option value="val">验证</option><option value="test">测试</option></select></label><label>Query位置 <input type="range" id="attention-query" min="0" max="10" value="7"><output class="value" id="attention-query-value">7</output></label></div><div class="tokens" id="attention-tokens"></div><div class="lab-output" id="attention-output" aria-live="polite"></div></div>''',
        'training': '''<div class="lab" id="loss-lab"><h3>动手：把InfoNCE从公式变成数字</h3><p>固定所有负例分数相同，改变正例、负例、负例数量和温度。先预测：正例分数提高时，损失和正例梯度分别怎么变？</p><div class="controls"><label>正例s⁺ <input type="range" id="loss-positive" min="-2" max="8" step=".1" value="2"><output class="value" id="loss-positive-value">2</output></label><label>负例s⁻ <input type="range" id="loss-negative" min="-2" max="5" step=".1" value="0"><output class="value" id="loss-negative-value">0</output></label><label>负例数 <select id="loss-k"><option>1</option><option>4</option><option>16</option><option selected>64</option></select></label><label>τ <input type="range" id="loss-tau" min=".2" max="2" step=".1" value="1"><output class="value" id="loss-tau-value">1</output></label></div><div class="lab-output" id="loss-output" aria-live="polite"></div></div>''',
        'evaluation': '''<div class="lab" id="metric-lab"><h3>动手：目标排名与ANN重合率，各自改变什么？</h3><p>这里用广告ID等于精确排名的人工列表。把目标排到第11名，再把ANN重合拉满，观察HR是否会因此变好。</p><div class="controls"><label>目标精确排名 <input type="number" id="metric-rank" min="1" max="100" value="3"></label><label>K <input type="range" id="metric-k" min="1" max="10" value="10"><output class="value" id="metric-k-value">10</output></label><label>重合项数 <input type="range" id="metric-overlap" min="0" max="10" value="9"><output class="value" id="metric-overlap-value">9</output></label></div><div class="lab-output" id="metric-output" aria-live="polite"></div></div>''',
        'retrieval': '''<div class="lab" id="mips-lab"><h3>动手：验证升维距离始终还原同一个内积分数</h3><p>教学数据库固定为A=(1,1)、B=(3,0)、C=(0,1)，R²=10。这里用2→3维便于手算；真实模型32→33维。</p><div class="controls"><label>q₁ <input type="number" id="mips-qx" min="-10" max="10" step=".1" value="1"></label><label>q₂ <input type="number" id="mips-qy" min="-10" max="10" step=".1" value="0"></label></div><div class="lab-output" id="mips-output" aria-live="polite"></div></div>''',
    }


def demo_chapter(demo):
    t = demo['trace']
    metadata = demo['prepared_metadata']
    sample_rows = [[esc(mode), esc(str(t['samples'][mode]['items'])),
                    esc(str(t['samples'][mode].get('positive', t['samples'][mode].get('target'))))]
                   for mode in ['train', 'val', 'test']]
    training_rows = [[str(row['epoch']), f"{row['train_loss']:.6f}",
                     f"{row['validation']['HR@10']:.6f}", f"{row['validation']['NDCG@10']:.6f}",
                     str(row['supervised_positions'])] for row in demo['history']]
    shape_rows = [[esc(key), esc(str(value))] for key, value in t['shapes'].items()]
    grad_rows = [[f'<code>{esc(key)}</code>', 'None：此路径未参与损失' if value is None else f'{value:.6f}']
                 for key, value in t['gradient_norms'].items()]
    oid_to_local = {int(oid): local for local, oid in enumerate(t['catalog_oids'])}
    rank_rows = [[str(i + 1), str(oid_to_local[int(item['item_oid'])]),
                 str(item['item_oid']), f"{item['score']:.6f}"]
                for i, item in enumerate(demo['recommendation']['recommendations'])]
    return '''<section class="chapter" id="demo"><div class="chapter-kicker">09 · 真实执行轨迹</div>
      <h2>把同一个用户送进真实代码，直到拿回十个OID</h2>
      <p class="lead">下面不是把公式填上想象的数字：本轮已经生成合成Parquet，实际调用当前prepare、cohorts、train、evaluate_checkpoint、export、benchmark_ann和recommend。所有中间数值来自该次CPU运行，完整记录嵌入本页。</p>
      <div class="callout warn"><strong>用途与尺度：</strong>这是一条用于学习和验证接线的小型合成链路，不能作为TencentGR-1M的效果结果。主架构、点击监督、64负例和五轮训练保留；用户数、历史长度、batch、参考抽样种子和索引预算按教学尺度调整。</div>''' + f'''
      <div class="cards"><div class="card"><strong>{metadata['users']}</strong><small>合成用户；五轮CPU训练</small></div><div class="card"><strong>{metadata['catalog_items']}</strong><small>非padding候选；向量矩阵107×32</small></div><div class="card"><strong>{demo['training']['best_epoch']}</strong><small>验证选择的最佳epoch，不强行取第5轮</small></div></div>
      <h3>9.1 生成与真实模块同格式的原始输入</h3>
      <p>主角用户RID=1使用前文十事件；另外11个用户提供可用负例。第2个用户的训练前缀包含广告6和8，使主角的留出目标在全局属于warm。原始广告OID人为设为20,000,000,000+RID，只为清楚显示映射；这不是公开真实广告的业务编码规则。</p>
      <p>本次有效训练事件共 {metadata['training_events']} 条，全局训练可见广告 {metadata['train_seen_items']} 个。主角的广告7和9未在全局训练前缀出现，因此ID lookup为0；广告6和8通过背景用户获得独立训练ID。</p>''' + table(['local ID', '训练出现次数', 'ID Embedding行号', 'OID'], [
          [str(i), str(t['first_train_counts'][i]), str(t['item_id_lookup_first10'][i]), str(t['first_catalog_oids'][i])]
          for i in range(1, 10)]) + '''
      <h3>9.2 冻结队列，再核对同一个用户的三种样本</h3>''' + f'''
      <p>先用参考抽样seed=2027准备6用户/2评估用户，再用主抽样seed=2026准备12用户，最后以seed=3107冻结6人的新队列。排除此前2名评估用户，重合数为 {demo['cohort']['overlap_with_previous_test_users']}；这复用了主项目的排除机制，但规模和参考抽样种子不同。主角处于新队列第 {t['cohort_position']} 个位置。</p>''' + table(['模式', '完整items，长度11', 'positive或target'], sample_rows) + '''
      <details><summary>展开实际切分结果和完整样本字段</summary>''' + pre(dict(split=demo['clean_and_split'], samples=t['samples'])) + '''</details>
      <h3>9.3 五轮真实训练，用验证指标选checkpoint</h3>''' + table(['epoch', '平均训练loss', '验证HR@10', '验证NDCG@10', '监督位置数'], training_rows) + f'''
      <p>该次验证最佳epoch为 {demo['training']['best_epoch']}。表里的训练loss是各批有效监督位置加权后的均值；后面单独追踪主角的一个位置，loss当然可以不同。示例没有为了制造命中而继续调模型或挑一个看起来更好的用户。</p>
      <h3>9.4 固定该checkpoint，展开一个监督位置的真实数值</h3>''' + table(['张量', '实际形状'], shape_rows) + f'''
      <p>主角唯一有效点击监督在数组位置 {t['supervised_positions'][0]}，对应“读广告1，预测点击广告2”。选出的state为[1,32]，64个采样负例为[1,64]。</p>
      <div class="formula"><p>实际s⁺ = {t['positive_scores'][0]:.9f}</p><p>实际p⁺ = {t['positive_softmax_probability'][0]:.9f}</p><p>实际L = −ln(p⁺) = {t['loss']:.9f}</p><small>负例分数并不相同，完整64项和负例ID在下面的折叠记录中。</small></div>
      <p>这次孤立的数值/梯度追踪将模型置于eval，关闭dropout以便复算，再执行一次反向传播；它不做optimizer.step，也不修改保存的checkpoint。下表是未裁剪的该次梯度；真正的五轮训练使用model.train、启用dropout，并在更新参数之前执行全局梯度范数裁剪。</p>''' + table(['参数', '实际梯度范数'], grad_rows) + '''
      <details><summary>展开正例、全部64负例分数/ID、softmax概率和梯度</summary>''' + pre({key:t[key] for key in ['supervised_positions', 'negative_ids', 'positive_scores', 'negative_scores', 'positive_softmax_probability', 'loss', 'gradient_norms']}) + '''</details>
      <h3>9.5 用真实用户向量，对完整106候选打分</h3>''' + f'''
      <p>验证历史是[1,2,3,4,5]，验证目标local ID=6。实际用户查询q有32维；过滤历史后，该目标分数为 <strong>{t['target_score']:.6f}</strong>，严格大于该分数的广告数量加一给出的排名为 <strong>{t['target_rank']}</strong>。</p>
      <div class="callout"><strong>这个样例的真实结果：</strong>目标没有进入精确Top-10，主角的HR@10=0、NDCG@10=0。索引准确复现模型排名也不能让一个原本排在前十之外的目标自动命中。教程保留这个结果，方便你检查模型能力与检索近似误差的区别。</div>
      <details><summary>展开真实32维查询与目标广告向量</summary>''' + pre({key:t[key] for key in ['query_vector', 'candidate_target_vector', 'target_score', 'target_rank']}) + '''</details>
      <h3>9.6 导出、建立索引、验证门槛，真正返回十条推荐</h3>''' + f'''
      <p>本次小索引使用M=16，测试efSearch候选[64,128,2048]，以验证ANN重合率≥98%且p95最小选择预算。该次选出efSearch={demo['index']['selected']['efSearch']}，验证重合率={demo['index']['selected']['ANNRecall@10']:.2%}。小索引的时间、预算选择与大规模主索引不是同一结果，不用于宣称加速倍数。</p>''' + table(['名次', 'local ID', '最终OID', '推荐原始内积分数'], rank_rows) + '''
      <p class="source-note">上表顺序和分数来自recommend实际返回值。对应精确结果的local ID也保存在完整trace中；若今后改参数使ANN重合率不再100%，两种列表应分别查看，不能假设它们始终一致。</p>
      <h3>9.7 自己重新跑一遍</h3>
      <pre><code>cd /home/jyk/project/baseline_2025
.project_venv/bin/python scripts/run_tutorial_demo.py \
  --root artifacts/tutorial_demo_replay_01

# trace.json、execution.log、合成Parquet、缓存、权重和Faiss索引都在该root下
# 每次复现使用一个不存在的新root，避免覆盖历史输出</code></pre>
      <p>该脚本直接复用主模块，没有绕过主配置去假装跑了300k流程。main.json的严格入口用于固定大规模主方案，合成教学入口则把缩小后的设置明确记录在trace中。浮点结果和索引计时仍依赖实际软件/硬件环境。</p>
      {{code:scripts/run_tutorial_demo.py:1:*}}
      <details><summary>本次执行环境、设置与来源摘要</summary>''' + pre({key:demo[key] for key in ['purpose', 'executed_at', 'base_commit', 'settings', 'environment', 'source_sha256']}) + '''</details>
      <details><summary>本次测试评估和最终推荐JSON</summary>''' + pre(dict(evaluation=demo['evaluation'], recommendation=demo['recommendation'])) + '''</details>
      <details><summary>自测：为什么完整链路能运行，仍不能说明模型已经做好？</summary><p>接线正确解决“是否按预期计算”，效果实验解决“是否比对照更好”。小数据的人为分布与规模不代表真实公开数据，更不代表线上流量。这个例子目标排名靠后，恰好说明要继续用固定数据、合理对照和分桶指标评估模型，而不是只检查程序返回了十个广告。</p></details>
      </section>'''


def source_chapter(builder):
    manifest = table(['文件', 'SHA256前16位', '行数'], [
        [f'<a href="#{source_id(path)}"><code>{esc(path)}</code></a>',
         f'<code>{builder.hashes[path][:16]}</code>', str(len(builder.sources[path].splitlines()))]
        for path in builder.sources])
    files = ''.join(builder.code(path, 1, '*', full=True) for path in builder.sources)
    return ('<section class="chapter" id="sources"><div class="chapter-kicker">12 · 可以离线核对的来源</div>'
            '<h2>核心源码、真实实验文件与构建说明</h2>'
            f'<p>Git 构建基准 <code>{builder.commit}</code>。正文片段从构建时的文件自动截取。每份嵌入源码以本页的文件摘要为准。</p>'
            f'<p>新增08A章单独使用<code>{esc(builder.research["branch"])}</code>的固定提交<code>{builder.research["commit"]}</code>。带<code>research/tail-recall/</code>前缀的文件来自本教程保存的研究快照，构建时核对每个文件SHA256；它们没有替换主分支实现，不能把研究指标与主模型历史成绩混为同一次实验。</p>'
            '<p>要定位正文某段代码，可点击“完整源码”；它会自动展开对应文件并跳到该行。真正运行代码仍需Python环境和相应的数据/产物，本页面本身可完全离线阅读。</p>'
            + manifest + '<details><summary>展开完整来源摘要</summary>' + pre(builder.hashes) + '</details>'
            '<h3>主要证据文件</h3><ul>'
            '<li><code>reports/objective_study/model_selection.json</code>：开发阶段按验证指标锁定方案。</li>'
            '<li><code>reports/objective_study/metrics.json</code>：三种子、配对区间、分桶和ANN实测。</li>'
            '<li><code>reports/objective_study/ann_initial_failure.json</code>、<code>ann_amendment.json</code>：失败尝试与验证后修正。</li>'
            '<li><code>docs/release.md</code>：主方案固定范围、seed2026单模型、历史测量口径。</li>'
            '<li><code>docs/tutorial_parts/demo_trace.json</code>：本教程新执行的合成教学记录。</li>'
            '<li><code>docs/tutorial_parts/tail_recall_snapshot/manifest.json</code>：研究分支固定提交与源码、指标、真实补回案例、独立核验的快照摘要。</li>'
            '</ul><h3>重新生成本HTML</h3>'
            '<pre><code>.project_venv/bin/python scripts/build_project_tutorial.py\n'
            '# 默认输出 docs/project_masterclass.html；不运行训练、不访问外网\n'
            '# 源章节、样式、交互脚本和演示记录在 docs/tutorial_parts/</code></pre>'
            '<p>如果修改了核心源码，构建器会检查演示来源摘要并提示不一致；应重新运行合成演示取得匹配的trace，再生成新版教程。历史大规模指标仍属于原始报告，不会因重建HTML自动变成新代码的实测成绩。</p>'
            '<h3>完整相关源码（按需展开）</h3>' + files + '</section>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/project_masterclass.html')
    parser.add_argument('--trace', default='docs/tutorial_parts/demo_trace.json')
    args = parser.parse_args()
    builder = Builder()
    demo = json.loads((REPO / args.trace).read_text())
    mismatches = [path for path, digest in demo['source_sha256'].items()
                  if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise ValueError('Teaching trace source mismatch; rerun run_tutorial_demo.py: ' + ', '.join(mismatches))
    pieces = [(PARTS / name).read_text() for name in ['overview.html', 'data.html', 'model.html', 'evaluation.html']]
    body = ''.join(pieces).replace('{{flowdiagram}}', diagram())
    for chapter_id, laboratory in laboratories().items():
        pattern = rf'(<section\b[^>]*id="{chapter_id}"[^>]*>.*?)(</section>)'
        body, count = re.subn(pattern, lambda m: m[1] + laboratory + m[2], body, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'Missing laboratory chapter {chapter_id}')
    tail_chapter = (PARTS / 'tail_recall.html').read_text()
    for key, value in builder.tail_tables().items():
        tail_chapter = tail_chapter.replace('{{' + key + '}}', value)
    tail_chapter = tail_chapter.replace('{{tail_lab}}', (PARTS / 'tail_lab.html').read_text())
    body += tail_chapter + (PARTS / 'channel_ranking.html').read_text()
    rank_trace = json.loads((REPO / 'reports/sequence_rank/demo_rid311428.json').read_text())
    sequence_chapter = (PARTS / 'sequence_ranking.html').read_text()
    body += sequence_chapter.replace('{{sequence_trace}}', pre(rank_trace))
    body += demo_chapter(demo) + (PARTS / 'appendix.html').read_text()
    body = builder.replace_code(body)
    if '{{code:' in body or '{{flowdiagram}}' in body or '{{tail_' in body or '{{sequence_trace}}' in body:
        raise ValueError('Unresolved template placeholder')
    body += source_chapter(builder)
    for _, chapter_id, _ in NAV:
        footer = (f'<div class="chapter-footer"><button type="button" class="mark-complete" aria-pressed="false">标记本章已理解</button>'
                  '<span>能独立回答本章问题后，再标记完成。</span></div>')
        body, count = re.subn(rf'(<section\b[^>]*id="{chapter_id}"[^>]*>.*?)(</section>)',
                             lambda m: m[1] + footer + m[2], body, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'Missing or repeated chapter {chapter_id}')
    css = (PARTS / 'tutorial.css').read_text()
    js = (PARTS / 'tutorial.js').read_text() + '\n' + (PARTS / 'tail_lab.js').read_text()
    data = json.dumps(dict(commit=builder.commit, demo=demo, source_sha256=builder.hashes,
                           snippets=builder.snippets, research=dict(snapshot=builder.research,
                           metrics=builder.tail_metrics, examples=builder.tail_examples,
                           verification=builder.tail_verification)), ensure_ascii=False).replace('<', '\\u003c')
    toc = ''.join(f'<a href="#{key}"><span>{no}</span><span>{label}</span></a>' for no, key, label in NAV)
    built = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M %Z')
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="TAAC2025序列推荐项目的完整离线代码教程，含真实源码、公式、贯穿样例、CPU实际执行轨迹和交互实验。">
<meta name="color-scheme" content="light dark"><title>吃透 TAAC2025 序列推荐项目 · 源码与实验证据</title>
<style>{css}\n[hidden]{{display:none!important}}</style></head><body>
<a class="skip" href="#main-content">跳到正文</a>
<aside class="rail" aria-label="学习目录"><div class="brand"><div class="brand-icon">R.</div><div><strong>序列推荐 · 代码讲义</strong><small>TAAC2025 / TENCENTGR-1M</small></div></div>
<label for="search">查找章节与源码关键词</label><input id="search" type="search" placeholder="例如：positive、NDCG、冷启动" autocomplete="off"><div id="search-status">搜索内容、函数名或公式 · Ctrl / ⌘ K</div>
<nav class="toc" aria-label="章节">{toc}</nav><div class="rail-foot"><div id="progress-text">0 / {len(NAV)} 章已理解</div><div class="meter" aria-hidden="true"><div id="progress-fill"></div></div>
<div class="tool-row"><button class="small-btn" id="source-toggle">折叠源码</button><button class="small-btn" id="theme-toggle">明 / 暗</button><button class="small-btn" id="print">打印 / PDF</button></div><p>单文件 · 无外部依赖<br>源码基准 {builder.commit[:7]}<br>本地保存阅读进度</p></div></aside>
<div class="mobile-bar"><button id="menu" aria-label="打开学习目录" aria-expanded="false">☰ 目录</button><span>TAAC2025 · 代码与公式</span><a href="#overview">首页 ↑</a></div>
<main class="main" id="main-content"><div class="article"><div class="topline"><span>RECOMMENDATION ENGINEERING / LEARNING EDITION</span><span class="version">{builder.commit[:7]} · OFFLINE HTML</span></div>
<header class="hero"><div class="eyebrow">从数据到模型，再到可验证的推荐结果</div><h1>从一条行为，<br>到十条推荐。</h1><p class="subtitle">一份把 TAAC2025 主项目讲透的代码讲义：沿同一个用户追踪完整流程，把每个数组、每条公式和每项实验结论对应起来。08C 章用真实用户展示 Parquet、候选、特征、两个残差和最终名次。</p>
<div class="hero-tags"><span class="pill">实际源码与行号</span><span class="pill">6组互动实验</span><span class="pill">双通道召回与列表排序</span><span class="pill">12用户真实CPU执行轨迹</span><span class="pill">离线可读 · 可打印</span></div>
<div class="hero-stats"><div><strong>32维</strong><small>广告与用户共享检索空间</small></div><div><strong>1层 / 1头</strong><small>当前固定主模型</small></div><div><strong>64负例</strong><small>每个有效点击监督位置</small></div><div><strong>270万+</strong><small>历史主实验的完整候选广告</small></div></div></header>
<noscript><div class="callout">浏览器关闭了JavaScript。正文、源码、公式、手算例子和实际运行结果仍可阅读；互动滑块与本地进度需要启用脚本。</div></noscript>
{body}<footer class="end-note">构建于 {built} · 原始大规模实验保持原记录，教学演示单独注明。<br>阅读时以本页嵌入的源码快照、来源摘要和标明的评估口径为准。</footer></div></main>
<div class="live-status" id="status" role="status" aria-live="polite"></div>
<script id="tutorial-data" type="application/json">{data}</script><script>{js}</script></body></html>'''
    output = REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size,
                          chapters=len(NAV), source_snippets=len(builder.snippets),
                          full_sources=len(builder.sources), trace_source_match=True,
                          research_commit=builder.research['commit'], research_snapshot_match=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
