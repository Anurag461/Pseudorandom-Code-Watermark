"""Recompute historical token repetition metrics under matched fallback policies.

Local CPU only. Run from the repository root with PYTHONPATH=.
The dispatched generation sources and all completed experiment records stay frozen.
"""
import json
from pathlib import Path

import numpy as np

from baseline_comparison.scoring import distinct_n, ngram_repetition_rate
from self_bleu.config import digest
from self_bleu.pilot import paired_interval
from self_bleu.repeat import ROOT, validate
from self_bleu.validation import save, sha


OUTPUT = Path(__file__).resolve().parent
REPEAT = ROOT/'outputs/self_bleu_repeat/setup_v4'
PILOT = ROOT/'outputs/self_bleu_pilot/stage_a_v2'
METRICS = ('repeated_4gram_fraction', 'distinct_3')
NATIVE = {'online_prc': 'prc', 'null': 'null', 'textseal': 'textseal_off',
          'synthid_text': 'synthid_on', 'gumbel_max': 'gumbel_off'}


def main():
    sources = {}

    def read(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Input hash changed: {path}')
        sources[str(path.relative_to(ROOT))] = actual
        return json.loads(path.read_text())

    manifest = read(REPEAT/'manifest.json')
    validate(manifest)
    refs = manifest['reference_files']
    inputs = read(PILOT/'inputs.json', refs[str((PILOT/'inputs.json').relative_to(ROOT))])
    historical = read(PILOT/'diversity.json', refs[str((PILOT/'diversity.json').relative_to(ROOT))])
    cohorts = {setting: [r for r in inputs if r['method'] == method] for method, setting in NATIVE.items()}
    for arm, stage in [('textseal_on', 'other_generators'), ('gumbel_on', 'other_generators'), ('synthid_off', 'synthid')]:
        report = read(REPEAT/f'{stage}_report.json')
        assert report['passed'] and report['manifest_id'] == manifest['id']
        cohorts[arm] = []
        for response in (0, 1):
            relative = f'batches/{arm}_r{response}.json'
            batch = read(REPEAT/'raw'/stage/relative, report['files'][relative])
            assert batch['manifest']['setting'] == manifest['arms'][arm]
            assert batch['manifest']['sampling_seed'] == manifest['seeds'][response]
            cohorts[arm].extend(batch['responses'])
    draws = np.random.default_rng(manifest['analysis']['bootstrap_seed']).integers(0, 50, (2000, 50))
    response_rows, summaries, arrays = [], [], {}
    for setting, responses in cohorts.items():
        responses.sort(key=lambda r: (r['prompt_index'], r['response_index']))
        assert [(r['prompt_index'], r['response_index']) for r in responses] == [(i, j) for i in range(50) for j in (0, 1)]
        assert all(len(r['token_ids']) == 1024 and digest(r['token_ids']) == r['completion_sha256'] for r in responses)
        for length in manifest['primary_lengths']:
            values = []
            for row in responses:
                ids = row['token_ids'][:length]
                metrics = dict(repeated_4gram_fraction=ngram_repetition_rate(ids, 4), distinct_3=distinct_n(ids, 3))
                assert all(0 <= value <= 1 for value in metrics.values())
                values.append([metrics[k] for k in METRICS])
                response_rows.append(dict(setting=setting, length=length, prompt_index=row['prompt_index'],
                    response_index=row['response_index'], response_id=row['response_id'],
                    completion_sha256=row['completion_sha256'], prefix_sha256=digest(ids), **metrics))
            response_values = np.asarray(values)
            prompt_values = response_values.reshape(50, 2, 2).mean(axis=1)
            arrays[setting, length] = prompt_values
            summaries.append(dict(setting=setting, length=length, responses=100, prompt_clusters=50,
                metrics={name: {**paired_interval(prompt_values[:, k], draws),
                                'median_response': float(np.median(response_values[:, k]))}
                         for k, name in enumerate(METRICS)}))
    parity_checks = 0
    for row in historical['rows']:
        values = arrays[NATIVE[row['method']], row['length']][row['prompt_index']]
        for got, key in zip(values, ('repetition_rate', 'distinct_3')):
            assert abs(got-row[key]) < 1e-12
            parity_checks += 1
    contrasts = []
    pairs = [(f'{method}_on', f'{method}_off', 'on_minus_off') for method in ('textseal', 'gumbel', 'synthid')]
    pairs += [('prc', other, 'prc_minus_comparator') for other in ('null', 'textseal_on', 'gumbel_on', 'synthid_on')]
    for left, right, kind in pairs:
        for length in manifest['primary_lengths']:
            delta = arrays[left, length]-arrays[right, length]
            contrasts.append(dict(kind=kind, left=left, right=right, length=length,
                metrics={name: paired_interval(delta[:, k], draws) for k, name in enumerate(METRICS)}))
    save(OUTPUT/'response_metrics.json', response_rows)
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    sources['baseline_comparison/scoring.py'] = sha(ROOT/'baseline_comparison/scoring.py')
    sources['self_bleu/pilot.py'] = sha(ROOT/'self_bleu/pilot.py')
    result = dict(manifest_id=manifest['id'], generation_attempts=0, incremental_modal_cost_usd=0,
        definitions={'repeated_4gram_fraction': '1 - unique token 4-grams / (T - 3); lower is better',
                     'distinct_3': 'unique token trigrams / (T - 2); higher is better',
                     'token_scope': 'Generated completion IDs only; no prompt, decoding, special-token removal or detector mask',
                     'aggregation': 'Mean within each two-response prompt pair, then mean over 50 prompts; equal to mean of 100 response slots',
                     'median_response': 'Supplemental median across the 100 response slots'},
        bootstrap=dict(resamples=2000, seed=manifest['analysis']['bootstrap_seed'], unit='paired prompt cluster',
                       draws_sha256=digest(draws.tolist()), intervals='Exploratory marginal 95% percentile intervals; no multiplicity correction'),
        policy_groups={'fallback_on': ['prc', 'synthid_on', 'textseal_on', 'gumbel_on', 'null'],
                       'fallback_off': ['prc', 'synthid_off', 'textseal_off', 'gumbel_off', 'null']},
        policy_limitations=['PRC is position based and ordinary sampling is already unwatermarked; their responses are reused in both groups.',
                            'Contextual methods share within-response repeat fallback, but native context initialization, RNGs and detectors are retained.',
                            'Same 50 prompts, two seeds, one key and fixed default parameters per method; not a parameter-frontier or quality evaluation.'],
        historical_metric_values_verified=parity_checks, sources=sources,
        response_metrics_sha256=sha(OUTPUT/'response_metrics.json'), results=summaries, contrasts=contrasts)
    save(OUTPUT/'summary.json', result)
    print(json.dumps({'results': summaries, 'contrasts': contrasts, 'historical_metric_values_verified': parity_checks}, indent=2))


if __name__ == '__main__':
    main()
