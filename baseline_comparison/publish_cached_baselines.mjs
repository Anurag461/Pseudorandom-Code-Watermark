// Append the locally verified, unchanged SynthID and Gumbel detection counts.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

assert(process.env.ARTIFACT_WORKSPACE, 'Set ARTIFACT_WORKSPACE to the bundled-dependency workspace');
const require = createRequire(path.join(process.env.ARTIFACT_WORKSPACE,'loader.cjs'));
const {Workbook} = await import(require.resolve('@oai/artifact-tool'));
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output = path.join(repo,'outputs/comparison_redetect');
const setup = path.join(output,'token_baseline_reuse');
const sha = data => createHash('sha256').update(data).digest('hex');
const before = await fs.readFile(path.join(setup,'before.csv'),'utf8');
const beforeProvenance = JSON.parse(await fs.readFile(path.join(setup,'before.provenance.json'),'utf8'));
assert.equal(sha(before),beforeProvenance.csv_sha256);
const summaryText = await fs.readFile(path.join(setup,'summary.json'),'utf8');
const summary = JSON.parse(summaryText);
assert.equal(summary.checks.cached_records_verified,12000);
assert.equal(summary.checks.shared_nulls_match_textseal_and_prc,500);
assert.equal(summary.checks.detector_calls,0);
assert.equal(summary.checks.model_forwards,0);
assert.equal(summary.checks.remote_calls,0);
assert.equal(summary.shared_null_source,beforeProvenance.null_source);
const workbook = await Workbook.fromCSV(before,{sheetName:'Comparisons'});
const sheet = workbook.worksheets.getItem('Comparisons');
const original = sheet.getUsedRange().values.map(row => row.map(v=>v??''));
assert.equal(original.length,13); assert.equal(original[0].length,29);
const headers = original[0];
assert.equal(headers.at(-7),'Method');
assert.deepEqual([...new Set(original.slice(1).map(row=>row[headers.indexOf('Method')]))].sort(),['online_prc','textseal']);
const additions = [];
const display = c => `${c.detected}/${c.count} (${(100*c.detected/c.count).toFixed(1)}%)`;
for (const method of ['synthid_text','gumbel_max']) {
  for (const n of [128,256,400,512,768,1024]) {
    const counts = summary.counts[method][n];
    assert.equal(counts.watermarked.count,500); assert.equal(counts.null.count,500);
    const row = Object.fromEntries(headers.map(k=>[k,'']));
    const calibration = method==='synthid_text'
      ? 'weighted normal approximation; official Google g-values; depth=10; fixed layer weights'
      : 'exact Gamma test; original TextSeal comparison uniform PRF; key=42';
    Object.assign(row,{Method:method,T:n,n,'Target FPR':.001,'Entropy Model':'not used',
      'Generation Model':'Qwen3-8B-Base','Entropy Trace Source':'cached_completion_token_scores',
      Score:'p_value','Old TPR':display(counts.watermarked),TPR:display(counts.watermarked),
      FPR:display(counts.null),'TPR change (pp)':'0.0',
      Notes:`Existing cached detection reused unchanged; no redetection or model replay; ${calibration}; context=3; v2 context-token deduplication; p < 0.001; shared null cohort=T13088; separate one-shot tests per prefix; scores use completion tokens and keys only; prompt and entropy are not detector inputs; source=${summary.source_directory}/controlled_baseline_full_prompt_level.jsonl; upstream=${summary.source_commits[method]}`});
    additions.push(headers.map(k=>row[k]));
  }
}
const matrix = [...original,...additions];
sheet.getRangeByIndexes(0,0,matrix.length,headers.length).values=matrix;
// CSV stores no formatting; sizing below is only for verification previews.
sheet.getRange('W1:AC25').format.columnWidth=25;
sheet.getRange('W1:AC1').format.wrapText=true;
sheet.getRange('W1:AC1').format.rowHeight=36;
workbook.recalculate();
const actual = sheet.getUsedRange().values.map(row=>row.map(v=>v??''));
assert.deepEqual(actual,matrix);
assert.deepEqual(actual.slice(0,original.length),original);
console.log((await workbook.inspect({kind:'table',range:'Comparisons!W14:AC25',include:'values',tableMaxRows:12,tableMaxCols:7})).ndjson);
const preview = await workbook.render({sheetName:sheet.name,range:'W1:AC25',scale:1,format:'png'});
await fs.writeFile(path.join(process.env.ARTIFACT_WORKSPACE,'all_baselines.png'),new Uint8Array(await preview.arrayBuffer()));
const field = v => /[",\r\n]/.test(String(v)) ? `"${String(v).replaceAll('"','""')}"` : String(v);
const csv = actual.map(row=>row.map(field).join(',')).join('\n')+'\n';
const checkWorkbook = await Workbook.fromCSV(csv,{sheetName:'Readback'});
assert.deepEqual(checkWorkbook.worksheets.getItem('Readback').getUsedRange().values.map(row=>row.map(v=>String(v??''))),actual.map(row=>row.map(String)));
const csvPath = path.join(output,'baseline_comparisons.csv');
const current = await fs.readFile(csvPath,'utf8');
assert(current===before || current===csv,'Concurrent comparison changes must be preserved');
const provenance = {...beforeProvenance,csv_sha256:sha(csv),cached_token_baselines:{...summary,
  summary_sha256:sha(summaryText),publisher_sha256:sha(await fs.readFile(fileURLToPath(import.meta.url)))}};
const checks = {passed:true,original_rows_preserved:12,original_cells_preserved:12*headers.length,
  added_rows:12,total_rows:24,methods:['online_prc','textseal','synthid_text','gumbel_max'],
  lengths:summary.prefix_lengths,nominal_fpr:.001,shared_null_source:summary.shared_null_source,
  before_csv_sha256:sha(before),csv_sha256:sha(csv),summary_sha256:sha(summaryText),
  csv_roundtrip_exact:true,detector_calls:0,model_forwards:0,remote_calls:0};
await fs.writeFile(csvPath+'.partial',csv);
await fs.writeFile(path.join(output,'baseline_comparisons.provenance.json.partial'),JSON.stringify(provenance,null,2)+'\n');
await fs.rename(csvPath+'.partial',csvPath);
await fs.rename(path.join(output,'baseline_comparisons.provenance.json.partial'),path.join(output,'baseline_comparisons.provenance.json'));
await fs.writeFile(path.join(setup,'verification.json'),JSON.stringify(checks,null,2)+'\n');
console.log(JSON.stringify(checks,null,2));
