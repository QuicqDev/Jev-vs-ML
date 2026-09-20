"""Export saved final tables without executing notebook cells or API calls."""
import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        elif tag in ('td', 'th'): self.cell = ''
    def handle_data(self, data):
        if self.cell is not None: self.cell += data
    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None

def saved_tables(notebook):
    final = next(c for c in reversed(notebook['cells']) if '### Raw decision rules' in ''.join(c.get('source', [])))
    tables = [o['data']['text/html'] for o in final['outputs'] if '<table' in ''.join(o.get('data', {}).get('text/html', []))]
    assert len(tables) == 2
    for panel, table in zip(('raw', 'adjusted'), tables):
        parser = TableParser()
        parser.feed(''.join(table))
        header = ['dataset'] + parser.rows[0][1:]
        rows = [r for r in parser.rows[1:] if len(r) == len(header) and r[0] != 'dataset']
        assert len(rows) == 8 and len(header) == 15
        assert all('(n=3)' in v and 'pending' not in v for r in rows for v in r[1:])
        yield panel, [header] + rows

def export():
    path = ROOT / 'jev_benchmark_v3.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    out = ROOT / 'published_results'
    out.mkdir(exist_ok=True)
    for panel, rows in saved_tables(notebook):
        with (out / f'{panel}_balanced_accuracy.csv').open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
    manifest = {
        'source_notebook': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'protocol': '3.0.1', 'requested_jev_model': 'jev-1.13.0',
        'training_seeds': [2027, 2028, 2029], 'holdout_seed': 20260920,
        'provenance': 'Final saved notebook outputs; no cells rerun. CSV values retain displayed rounding.',
        'uncertainty': 'Mean and sample SD across training seeds on the same test cases; not confidence intervals.',
        'unavailable': ['per-example predictions', 'paired_bootstrap_intervals.csv', 'run_diagnostics.csv', 'dataset snapshots', 'full Kaggle result archive'],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Exported both saved result panels and provenance manifest.')

if __name__ == '__main__':
    export()
