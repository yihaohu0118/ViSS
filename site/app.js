'use strict';

const comparisons = {
  "matched": {
    "title": "Separate what the model learns from what it sees.",
    "note": "One Qwen3-VL-8B backbone. The first contrast fixes recent-4 access; the second freezes the ViSS checkpoint and enables TVS. Higher is better.",
    "caption": "Full-split results: StreamingBench accuracy and OVO-Bench official group-macro accuracy (%).",
    "columns": [
      "Model / view policy",
      "StreamingBench ↑",
      "OVO-Bench ↑"
    ],
    "rows": [
      [
        "Qwen3-VL-8B · recent-4",
        80.52,
        58.97
      ],
      [
        "ViSS · recent-4",
        84.77,
        65.21
      ],
      [
        "ViSS + TVS",
        86.33,
        69.77
      ]
    ],
    "charts": [
      "StreamingBench",
      "OVO-Bench"
    ]
  },
  "controls": {
    "title": "What does visible-state supervision add?",
    "note": "Recent-4 access is fixed. The strict-matched generic control uses the same training instances, visual inputs, weighting, ordering, optimization, initialization, and seed.",
    "caption": "Controlled state-supervision comparison (%), Table 3(a) in the manuscript.",
    "columns": [
      "Supervision",
      "StreamingBench ↑",
      "OVO-Bench ↑"
    ],
    "rows": [
      [
        "Base",
        80.52,
        58.97
      ],
      [
        "Strict-matched generic",
        84.97,
        63.48
      ],
      [
        "ViSS without sufficiency",
        84.89,
        63.52
      ],
      [
        "ViSS",
        84.77,
        65.21
      ]
    ],
    "charts": [
      "StreamingBench",
      "OVO-Bench"
    ]
  },
  "access": {
    "title": "Acquire more evidence with the same checkpoint.",
    "note": "All rows freeze the ViSS checkpoint. Frame counts include repeated frames across calls. The oracle uses reference answers and is diagnostic, not a deployable policy.",
    "caption": "Accuracy (%) and mean processed frames per query from the manuscript's access-cost comparison.",
    "columns": [
      "View policy",
      "SB accuracy ↑",
      "OVO accuracy ↑",
      "SB frames",
      "OVO frames"
    ],
    "rows": [
      [
        "Recent-4",
        84.77,
        65.21,
        4.0,
        4.0
      ],
      [
        "Full history",
        86.45,
        68.84,
        29.97,
        30.54
      ],
      [
        "Exhaustive max-confidence",
        86.77,
        68.43,
        53.48,
        54.33
      ],
      [
        "Three-view oracle (diagnostic)",
        91.42,
        78.52,
        53.48,
        54.33
      ],
      [
        "TVS",
        86.33,
        69.77,
        17.08,
        30.33
      ]
    ],
    "charts": [
      "StreamingBench",
      "OVO-Bench"
    ]
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
