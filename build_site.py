"""Build the dependency-free Pages report from the published score tables."""
import csv
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / 'docs'
TEST_ROWS = [1000, 1500, 1000, 1000, 1000, 1000, 114, 30]

def load_data():
    data = {}
    for panel in ('raw', 'adjusted'):
        path = ROOT / 'published_results' / f'{panel}_balanced_accuracy.csv'
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            models = reader.fieldnames[1:]
            rows = []
            for i, row in enumerate(reader):
                scores = {}
                for model in models:
                    match = re.fullmatch(r'(\d+\.\d) ± (\d+\.\d) \(n=(\d+)\)', row[model])
                    assert match, row[model]
                    mean, sd, n = match.groups()
                    scores[model] = dict(mean=float(mean), sd=float(sd), n=int(n), display=row[model])
                classical = models[:11]
                best = max(scores[m]['mean'] for m in classical)
                rows.append(dict(dataset=row['dataset'], kind='text' if i < 4 else 'tabular',
                    testRows=TEST_ROWS[i], scores=scores, best=best,
                    bestModels=[m for m in classical if scores[m]['mean'] == best]))
        data[panel] = dict(models=models, rows=rows)
    return data

def chart_rows(rows):
    result = []
    for row in rows:
        bars = []
        for key, value, label in [('zero', row['scores']['Jev zero-shot']['mean'], 'Jev zero-shot prompts'),
                                  ('few', row['scores']['Jev few-shot']['mean'], 'Jev few-shot prompts'),
                                  ('classic', row['best'], 'Best classical pipeline')]:
            bars.append(f'<div class="bar-line" style="--value:{value}%" aria-label="{label}: {value:.1f}%"><span class="bar {key}"></span><span class="value">{value:.1f}</span></div>')
        result.append(f'<div class="chart-row"><div class="chart-label"><strong>{html.escape(row["dataset"])}</strong><small>{row["kind"].title()} · {row["testRows"]:,} test rows</small></div><div class="bars">{"".join(bars)}</div></div>')
    return ''.join(result) + '<div class="axis" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>'

def full_table(panel):
    models = panel['models']
    out = ['<caption class="sr-only">Raw balanced accuracy: mean ± sample standard deviation, three seeds.</caption><thead><tr><th scope="col">Dataset</th>']
    out.extend(f'<th scope="col">{html.escape(m)}</th>' for m in models)
    out.append('</tr></thead><tbody>')
    for row in panel['rows']:
        best = max(s['mean'] for s in row['scores'].values())
        out.append(f'<tr><th scope="row">{html.escape(row["dataset"])}</th>')
        for model in models:
            score = row['scores'][model]
            cls = ('jev-cell ' if model.startswith('Jev ') else '') + ('best' if score['mean'] == best else '')
            out.append(f'<td class="{cls}">{score["mean"]:.1f} ± {score["sd"]:.1f}</td>')
        out.append('</tr>')
    return ''.join(out) + '</tbody>'

def build():
    data = load_data()
    (DOCS / 'assets/data.js').write_text('window.BENCHMARK = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
    template = (DOCS / 'report.template.html').read_text(encoding='utf-8-sig')
    page = template.replace('__CHART__', chart_rows(data['raw']['rows'])).replace('__TABLE__', full_table(data['raw']))
    assert '__CHART__' not in page and '__TABLE__' not in page
    (DOCS / 'index.html').write_text(page, encoding='utf-8')
    for panel in data:
        name = f'{panel}_balanced_accuracy.csv'
        shutil.copyfile(ROOT / 'published_results' / name, DOCS / 'data' / name)
    print('Built docs/index.html and data assets from both published result panels.')

if __name__ == '__main__':
    build()
