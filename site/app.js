'use strict';

const comparisons = {
  matched: {
    title: 'Same model size. Same observation access.',
    note: 'Official Qwen3-VL-8B versus ChronoReplay. OVO-S uses Uniform-128; OST and UCS follow their fixed causal evaluation protocols. Higher is better.',
    caption: 'Matched causal-access evaluation, scores in percent.',
    columns: ['Model', 'OVO-S ↑', 'OST ↑', 'UCS ↑'],
    rows: [['Official Qwen3-VL-8B', 48.98, 50.52, 45.33], ['ChronoReplay', 53.89, 63.93, 52.77]],
    charts: ['OVO-S-Bench', 'OST-Bench', 'UCS-Bench']
  },
  controls: {
    title: 'Complementary supervision supports broader transfer.',
    note: 'All variants start independently from Qwen3-VL-8B and use matched observation access. The categorical-only diagnostic arm is distinct from the full preservation pool, which also contains numerical targets.',
    caption: 'Single-family supervision controls from the paper, scores in percent.',
    columns: ['Training', 'OVO-S ↑', 'OST ↑', 'UCS ↑', 'Mean ↑'],
    rows: [['Official base', 48.98, 50.52, 45.33, 48.28], ['Categorical-only', 58.03, 47.48, 46.72, 50.74], ['Trajectory-only', 52.03, 51.03, 49.22, 50.76], ['ChronoReplay', 53.89, 63.93, 52.77, 56.86]],
    charts: ['OVO-S-Bench', 'OST-Bench', 'UCS-Bench']
  },
  access: {
    title: 'Change evidence access. Keep the checkpoint fixed.',
    note: 'These rows use the same ChronoReplay checkpoint. Uniform-81 matches the approximate average frame cost of adaptive access; the adaptive-access gain is an inference-time system effect.',
    caption: 'OVO-S observation-access comparison with fixed trained parameters.',
    columns: ['Observation policy', 'OVO-S ↑', 'Mean frames ↓'],
    rows: [['Uniform-81', 52.40, 81.00], ['Uniform-128', 53.89, 128.00], ['Adaptive access', 57.98, 80.50]],
    charts: ['OVO-S-Bench']
  }
};

function renderResults(key) {
  const data = comparisons[key];
  document.querySelector('#results-title').textContent = data.title;
  document.querySelector('#results-note').textContent = data.note;
  const table = document.querySelector('#results-table');
  table.replaceChildren();
  const caption = table.createCaption();
  caption.textContent = data.caption;
  const headRow = table.createTHead().insertRow();
  data.columns.forEach(label => {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    headRow.append(cell);
  });
  const body = table.createTBody();
  data.rows.forEach((values, index) => {
    const row = body.insertRow();
    if (index === data.rows.length - 1) row.classList.add('highlight');
    values.forEach((value, column) => {
      const cell = document.createElement(column ? 'td' : 'th');
      if (!column) cell.scope = 'row';
      cell.textContent = typeof value === 'number' ? value.toFixed(2) : value;
      row.append(cell);
    });
  });
  const chart = document.querySelector('#result-chart');
  chart.replaceChildren();
  data.charts.forEach((label, column) => {
    const group = document.createElement('div');
    group.className = 'chart-column';
    const title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = label + ' · score (%)';
    group.append(title);
    data.rows.forEach((row, index) => {
      const barRow = document.createElement('div');
      barRow.className = 'bar-row';
      const barLabel = document.createElement('div');
      barLabel.className = 'bar-label';
      const name = document.createElement('span');
      name.textContent = row[0];
      const value = document.createElement('strong');
      value.textContent = row[column + 1].toFixed(2);
      barLabel.append(name, value);
      const track = document.createElement('div');
      track.className = 'bar-track';
      const fill = document.createElement('div');
      fill.className = 'bar-fill' + (index === data.rows.length - 1 ? ' ours' : '');
      fill.style.width = row[column + 1] + '%';
      track.append(fill);
      barRow.append(barLabel, track);
      group.append(barRow);
    });
    chart.append(group);
  });
  document.querySelectorAll('[data-result]').forEach(button => {
    const active = button.dataset.result === key;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}
document.querySelectorAll('[data-result]').forEach(button => button.addEventListener('click', () => renderResults(button.dataset.result)));
renderResults('matched');

document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = 'Copied!';
      document.querySelector('#copy-status').textContent = 'Copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Text selected';
      document.querySelector('#copy-status').textContent = 'Text selected. Press Control+C or Command+C to copy.';
    }
    setTimeout(() => { button.textContent = original; }, 2200);
  });
});

const viewer = document.querySelector('#image-viewer');
document.querySelectorAll('[data-zoom]').forEach(link => {
  link.addEventListener('click', event => {
    if (typeof viewer.showModal !== 'function') return;
    event.preventDefault();
    const image = link.querySelector('img');
    const expanded = document.querySelector('#expanded-image');
    expanded.src = link.href;
    expanded.alt = image.alt;
    document.querySelector('#expanded-caption').textContent = image.alt;
    viewer.showModal();
  });
});
document.querySelector('.close-viewer').addEventListener('click', () => viewer.close());
viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
