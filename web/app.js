const byId = (id) => document.getElementById(id);
const form = byId('analysis-form');
const fileInput = byId('csv-file');
const source = byId('source');
const run = byId('run');
const count = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 });
let current = null;

function displayTable(id, rows) {
  const body = byId(id);
  body.replaceChildren();
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) row.insertCell().textContent = value;
  }
}

function render(data, label, budget) {
  const { holdout, selected, split } = data;
  byId('result-source').textContent = label;
  byId('result-context').textContent = `${selected} was selected by validation average precision. ${count.format(holdout.captured)} of ${count.format(holdout.contacted)} customers on the contact list churned.`;
  byId('auc').textContent = holdout.auc.toFixed(3);
  byId('precision').textContent = percent.format(holdout.precision);
  byId('recall').textContent = percent.format(holdout.recall);
  byId('lift').textContent = `${holdout.lift.toFixed(2)}×`;
  byId('split').textContent = `${count.format(split[0])} training · ${count.format(split[1])} validation · ${count.format(split[2])} test`;
  displayTable('model-rows', Object.entries(data.comparison).map(([name, metrics]) => [
    name === selected ? `${name} (selected)` : name, metrics.auc.toFixed(3), metrics.ap.toFixed(3),
  ]));
  const preview = data.preview.slice(0, holdout.contacted);
  byId('contact-detail').textContent = `Showing ${count.format(preview.length)} of ${count.format(holdout.contacted)} contacts at a ${budget}% budget. The CSV includes the full test set.`;
  displayTable('contact-rows', preview.map((row, index) => [index + 1, row.id, row.score.toFixed(3), row.churned ? 'Churned' : 'Stayed']));
  byId('placeholder').hidden = true;
  byId('report').hidden = false;
  current = data;
}

function download(value, name, type) {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function syncDataset() {
  const upload = source.value === 'csv';
  byId('upload').hidden = !upload;
  fileInput.required = upload;
  fileInput.disabled = !upload;
}

async function analyze() {
  const demo = source.value === 'demo';
  const file = fileInput.files[0];
  const budget = byId('budget').value;
  run.disabled = true;
  byId('status').textContent = 'Training both models…';
  byId('error').textContent = '';
  byId('result').setAttribute('aria-busy', 'true');
  try {
    if (!demo && (!file?.size || file.size > 2 * 1024 * 1024)) throw new Error('Choose a non-empty CSV file of 2 MB or less.');
    const response = await fetch(`api/analyze?contact=${encodeURIComponent(budget)}`, {
      method: 'POST', headers: { 'Content-Type': demo ? 'application/json' : 'text/csv' },
      body: demo ? JSON.stringify({ demo: true }) : file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The analysis could not finish.');
    render(data, demo ? 'Synthetic demo data' : file.name, budget);
    byId('status').textContent = 'Analysis complete.';
  } catch (error) {
    byId('status').textContent = current ? 'Showing your last completed analysis.' : '';
    byId('error').textContent = error instanceof TypeError ? 'The local server could not be reached. Restart Churn Predictor and try again.' : error.message;
    byId('placeholder-text').textContent = 'Run an analysis to view results.';
  } finally {
    byId('result').setAttribute('aria-busy', 'false');
    run.disabled = false;
  }
}

source.addEventListener('change', syncDataset);
form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!run.disabled) analyze();
});
byId('csv-download').addEventListener('click', () => download(current.rankingCsv, 'holdout_ranking.csv', 'text/csv;charset=utf-8'));
byId('report-download').addEventListener('click', () => download(current.report, 'churn_report.md', 'text/markdown;charset=utf-8'));
syncDataset();
analyze();
