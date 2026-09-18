// Publish the verified replay counts, preserving all existing PRC columns.
// ARTIFACT_WORKSPACE must contain a node_modules symlink to the bundled runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const workspace = process.env.ARTIFACT_WORKSPACE;
assert(workspace, 'Set ARTIFACT_WORKSPACE to the temporary bundled-dependency workspace');
const require = createRequire(path.join(workspace, 'loader.cjs'));
const { Workbook } = await import(require.resolve('@oai/artifact-tool'));
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(repo, 'outputs/comparison_redetect');
const setup = path.join(output, 'textseal_setup/direct_prefix');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const before = await fs.readFile(path.join(setup, 'before_textseal.csv'), 'utf8');
const oldProvenance = JSON.parse(await fs.readFile(path.join(setup, 'before_textseal.provenance.json'), 'utf8'));
assert.equal(hash(before), oldProvenance.csv_sha256);
const workbook = await Workbook.fromCSV(before, { sheetName: 'Comparisons' });
const sheet = workbook.worksheets.getItem('Comparisons');
const initial = sheet.getUsedRange().values.map(row => row.map(v => v ?? ''));
const oldHeaders = initial[0];
assert.equal(initial.length, 7);
assert.equal(oldHeaders.length, 22);

if (process.argv.includes('--preview-only')) {
  // CSV has no stored styles. These sizing changes apply only to the preview.
  sheet.getRange('K1:R7').format.columnWidth = 24;
  sheet.getRange('K1:R1').format.wrapText = true;
  sheet.getRange('K1:R1').format.rowHeight = 36;
  workbook.recalculate();
  const preview = await workbook.render({ sheetName: sheet.name, range: 'K1:R7', scale: 1, format: 'png' });
  await fs.writeFile(path.join(workspace, 'before.png'), new Uint8Array(await preview.arrayBuffer()));
  console.log((await workbook.inspect({ kind: 'table', range: 'Comparisons!K1:R7', include: 'values', tableMaxRows: 7, tableMaxCols: 8 })).ndjson);
  process.exit(0);
}

const summaryText = await fs.readFile(path.join(setup, 'full_summary.json'), 'utf8');
const summary = JSON.parse(summaryText);
assert.equal(summary.checks.record_checksums_and_inputs_verified, 1000);
assert.equal(summary.checks.prefix_results_verified, 6000);
assert.equal(summary.checks.shared_nulls_match_prc, true);
assert.equal(summary.nominal_fpr, .001);
assert.equal(summary.score_field, 'p_value_weighted');
const additions = ['Method', 'alpha', 'Score', 'Old TPR', 'TPR', 'FPR', 'TPR change (pp)'];
assert(additions.every(name => !oldHeaders.includes(name)));
const headers = [...oldHeaders, ...additions];
const objects = initial.slice(1).map(row => Object.fromEntries(oldHeaders.map((k,i) => [k,row[i]])));
for (const row of objects) {
  assert.equal(row['PRC Construction'], 'online_causal_prc_v1');
  // Posterior is the historical primary PRC test. Its entropy-aware results
  // remain in their existing, explicitly named columns.
  Object.assign(row, { Method: 'online_prc', alpha: '', Score: 'posterior',
    'Old TPR': row['Old Posterior TPR'], TPR: row['Posterior TPR'],
    FPR: row['Posterior FPR'], 'TPR change (pp)': row['Posterior TPR change (pp)'] });
}
const countText = (count, total) => `${count}/${total} (${(100*count/total).toFixed(1)}%)`;
const textsealRows = [];
for (const n of [128,256,400,512,768,1024]) {
  const wm = summary.counts[n].textseal, nil = summary.counts[n].null;
  assert.equal(wm.count, 500); assert.equal(nil.count, 500);
  for (const x of [wm,nil]) {
    assert.equal(x.detected, x.old_detected+x.gained-x.lost);
    assert(x.detected >= 0 && x.detected <= 500 && x.abstained >= 0 && x.abstained <= 500);
  }
  const row = Object.fromEntries(headers.map(k => [k, '']));
  Object.assign(row, { Method: 'textseal', alpha: .1, Score: 'p_value_weighted', T:n, n,
    'Target FPR': .001, 'Entropy Model': 'Qwen3-8B-Base', 'Generation Model': 'Qwen3-8B-Base',
    'Entropy Trace Source': summary.protocol,
    'Old TPR': countText(wm.old_detected,wm.count), TPR:countText(wm.detected,wm.count),
    FPR:countText(nil.detected,nil.count), 'TPR change (pp)': ((wm.detected-wm.old_detected)/5).toFixed(1),
    Notes: `Original upstream TextSeal ${summary.upstream_commit}; ngram=3; alpha=0.1; keys=42/12387; v2; BF16; completion-only direct detection at each length; weighted p < 0.001; shared null cohort=T13088; one-shot per prefix; repeat handling unchanged; old TPR=historical prompt-conditioned generation entropy; abstentions WM=${wm.abstained}/500, null=${nil.abstained}/500; report=${summary.modal_root}/full.json` });
  textsealRows.push(row);
}
const matrix = [headers,...objects,...textsealRows].map((row,i) => i===0 ? row : headers.map(k => row[k]));
sheet.getRangeByIndexes(0,0,matrix.length,headers.length).values = matrix;
// Formatting affects the human preview only; the deliverable remains CSV.
sheet.getRange('W1:AC13').format.columnWidth = 25;
sheet.getRange('W1:AC1').format.wrapText = true;
sheet.getRange('W1:AC1').format.rowHeight = 36;
workbook.recalculate();
const actual = sheet.getRangeByIndexes(0,0,matrix.length,headers.length).values.map(row => row.map(v => v ?? ''));
assert.deepEqual(actual, matrix);
for (let i=0; i<initial.length; i++) assert.deepEqual(actual[i].slice(0,oldHeaders.length),initial[i]);
console.log((await workbook.inspect({ kind:'table', range:'Comparisons!W1:AC13', include:'values',tableMaxRows:13,tableMaxCols:7 })).ndjson);
const preview = await workbook.render({ sheetName:sheet.name, range:'W1:AC13', scale:1, format:'png' });
await fs.writeFile(path.join(workspace,'after.png'),new Uint8Array(await preview.arrayBuffer()));
// The documented API provides CSV import and range values. Serialize those
// verified values as RFC 4180 fields; do not create an unrequested XLSX.
const cell = v => /[",\r\n]/.test(String(v)) ? `"${String(v).replaceAll('"','""')}"` : String(v);
const csv = actual.map(row => row.map(cell).join(',')).join('\n')+'\n';
const roundTrip = await Workbook.fromCSV(csv,{sheetName:'Check'});
assert.deepEqual(roundTrip.worksheets.getItem('Check').getUsedRange().values.map(row => row.map(v => String(v??''))),actual.map(row => row.map(String)));
const csvPath = path.join(output,'baseline_comparisons.csv');
const current = await fs.readFile(csvPath,'utf8');
assert(current===before || current===csv, 'Comparison changed during replay; preserve concurrent edits');
const provenance = { ...oldProvenance, csv_sha256:hash(csv),
  comparison_schema: { version:2, primary_columns:additions, prc_primary:'posterior',
    note:'Original PRC columns and values are unchanged. Generic columns show primary tests; PRC entropy-aware results remain in their existing columns. TextSeal-specific PRC fields are blank.' },
  textseal:{ ...summary, full_summary_sha256:hash(summaryText),
    publisher_sha256:hash(await fs.readFile(fileURLToPath(import.meta.url))) } };
const check = { passed:true, original_prc_rows:6, all_original_prc_cells_unchanged:true,
  textseal_rows:6, target_fpr:.001, score_field:'p_value_weighted',
  before_csv_sha256:hash(before), csv_sha256:hash(csv),
  before_provenance_sha256:hash(await fs.readFile(path.join(setup,'before_textseal.provenance.json'))),
  full_summary_sha256:hash(summaryText), csv_roundtrip_values_exact:true };
await fs.writeFile(csvPath+'.partial',csv);
await fs.writeFile(path.join(output,'baseline_comparisons.provenance.json.partial'),JSON.stringify(provenance,null,2)+'\n');
await fs.rename(csvPath+'.partial',csvPath);
await fs.rename(path.join(output,'baseline_comparisons.provenance.json.partial'),path.join(output,'baseline_comparisons.provenance.json'));
await fs.writeFile(path.join(setup,'comparison_verification.json'),JSON.stringify(check,null,2)+'\n');
console.log(JSON.stringify(check,null,2));
