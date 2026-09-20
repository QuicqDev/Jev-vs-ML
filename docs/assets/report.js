(() => {
  'use strict';
  const data = window.BENCHMARK;
  const chart = document.getElementById('comparison-chart');
  const table = document.getElementById('full-table');
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function render() {
    const panel = document.querySelector('input[name="panel"]:checked').value;
    const kind = document.getElementById('dataset-filter').value;
    const rows = data[panel].rows.filter(row => kind === 'all' || row.kind === kind);
    const adjusted = panel === 'adjusted';
    const title = adjusted ? 'Threshold-adjusted' : 'Raw decisions';
    document.getElementById('panel-note').textContent = adjusted
      ? 'Adjusted: binary thresholds learned from separate labeled policy data. Jev “zero-shot” describes its prompts only; these binary results are not zero-shot end to end. Multiclass results are unchanged.'
      : 'Raw: default decision rules after classical model selection. Jev zero-shot uses no task examples or policy threshold fitting.';
    chart.innerHTML = rows.map(row => {
      const series = [ ['zero', row.scores['Jev zero-shot'].mean, 'Jev zero-shot prompts'], ['few', row.scores['Jev few-shot'].mean, 'Jev few-shot prompts'], ['classic', row.best, 'Best classical: ' + row.bestModels.join(', ')] ];
      return `<div class="chart-row"><div class="chart-label"><strong>${escape(row.dataset)}</strong><small>${row.kind === 'text' ? 'Text' : 'Tabular'} · ${row.testRows.toLocaleString('en-US')} test rows</small></div><div class="bars">${series.map(([key,value,label]) => `<div class="bar-line" style="--value:${value}%" aria-label="${escape(label)}: ${value.toFixed(1)}%" title="${escape(label)}: ${value.toFixed(1)}%"><span class="bar ${key}"></span><span class="value">${value.toFixed(1)}</span></div>`).join('')}</div></div>`;
    }).join('') + '<div class="axis" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>';
    const wins = rows.filter(row => row.scores['Jev zero-shot'].mean > row.best);
    const lead = wins.length ? wins.reduce((a,b) => a.scores['Jev zero-shot'].mean-a.best > b.scores['Jev zero-shot'].mean-b.best ? a : b) : null;
    document.getElementById('result-takeaway').textContent = `${adjusted ? 'Adjusted zero-shot-prompt' : 'Raw zero-shot'} Jev leads the best classical mean on ${wins.length ? wins.map(r => r.dataset).join(' and ') : 'none of the selected datasets'}.${lead ? ` The largest lead is ${lead.dataset}: +${(lead.scores['Jev zero-shot'].mean-lead.best).toFixed(1)} percentage points.` : ''} These are descriptive differences, not statistical-significance claims.`;
    const models = ['Jev zero-shot', 'Jev few-shot', ...data[panel].models.filter(m => !m.startsWith('Jev '))];
    table.innerHTML = `<caption class="sr-only">${title} balanced accuracy: mean ± sample standard deviation, three seeds.</caption><thead><tr><th scope="col">Dataset</th>${models.map(m=>`<th scope="col">${escape(m)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => {
      const best = Math.max(...Object.values(row.scores).map(s => s.mean));
      return `<tr><th scope="row">${escape(row.dataset)}</th>${models.map(m => { const s = row.scores[m]; return `<td class="${s.mean === best ? 'best' : ''}" title="${escape(m)}: ${escape(s.display)}">${s.mean === best ? '<strong>' : ''}${s.mean.toFixed(1)} ± ${s.sd.toFixed(1)}${s.mean === best ? '</strong>' : ''}</td>`; }).join('')}</tr>`;
    }).join('')}</tbody>`;
    document.getElementById('table-panel').textContent = `${title} · ${rows.length} datasets · 14 model columns`;
    const download = document.getElementById('csv-download');
    download.href = `data/${panel}_balanced_accuracy.csv`;
    download.textContent = 'Download all eight datasets ↓';
  }
  document.querySelectorAll('input[name="panel"]').forEach(input => input.addEventListener('change',render));
  document.getElementById('dataset-filter').addEventListener('change',render);
  render();
})();
