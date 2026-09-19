"""Consolidate saved paired Self-BLEU, repetition, detection and null evidence.

Run locally from the repository root with PYTHONPATH=.; no GPU or Modal calls.
Bootstrap the actual per-prompt PRC-minus-comparator vectors, preserving pairing.
"""
import csv
import json
from pathlib import Path

import numpy as np

from self_bleu.config import digest
from self_bleu.pilot import paired_interval, score_rows
from self_bleu.repeat import ROOT, validate
from self_bleu.validation import save, sha


OUTPUT = Path(__file__).resolve().parent
PILOT = ROOT/'outputs/self_bleu_pilot/stage_a_v2'
REPEAT = ROOT/'outputs/self_bleu_repeat/setup_v4'
REPETITION = ROOT/'outputs/self_bleu_repeat/matched_repetition'
METHODS = {'prc': 'online_prc', 'null': 'null', 'textseal_off': 'textseal',
           'textseal_on': 'textseal', 'synthid_on': 'synthid_text',
           'gumbel_off': 'gumbel_max', 'gumbel_on': 'gumbel_max'}
NATIVE = {'online_prc': 'prc', 'null': 'null', 'textseal': 'textseal_off',
          'synthid_text': 'synthid_on', 'gumbel_max': 'gumbel_off'}
VIEWS = {'native': ['prc', 'textseal_off', 'synthid_on', 'gumbel_off', 'null'],
         'fallback_on': ['prc', 'textseal_on', 'synthid_on', 'gumbel_on', 'null']}
NAMES = {'textseal': 'TextSeal', 'synthid_text': 'SynthID', 'gumbel_max': 'Gumbel-Max'}
METRICS = ('self_bleu', 'repeated_4gram_fraction', 'distinct_3', 'tpr')


def close(actual, expected):
    assert np.allclose(actual, expected, atol=1e-12, rtol=0), (actual, expected)


def check_interval(actual, expected):
    close(actual['mean'], expected['mean'])
    close(actual['ci95'], expected['ci95'])


def main():
    sources = {}

    def read(path, expected=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError(f'Input hash differs: {path}')
        sources[str(path.relative_to(ROOT))] = value
        return json.loads(path.read_text())

    manifest = read(REPEAT/'manifest.json')
    validate(manifest)
    refs = manifest['reference_files']
    pilot = read(PILOT/'summary.json', refs[str((PILOT/'summary.json').relative_to(ROOT))])
    for name, expected in pilot['sources'].items():
        assert sha(PILOT/name) == expected, name
        sources[str((PILOT/name).relative_to(ROOT))] = expected
    original = read(PILOT/'diversity.json', pilot['sources']['diversity.json'])['rows']
    index = read(REPEAT/'artifact_index.json')
    ablation = read(REPEAT/'all_analysis.json', index['local_record_sha256']['all_analysis.json'])
    modified = read(REPEAT/'raw/all_metric_rows.json')
    repetition = read(REPETITION/'summary.json')
    for name, expected in repetition['sources'].items():
        assert sha(ROOT/name) == expected, name
    rep_rows = read(REPETITION/'response_metrics.json', repetition['response_metrics_sha256'])
    draws = np.random.default_rng(manifest['analysis']['bootstrap_seed']).integers(0, 50, (2000, 50))
    draws_hash = digest(draws.tolist())
    assert draws_hash == pilot['bootstrap_draws_sha256'] == ablation['bootstrap_draws_sha256'] == repetition['bootstrap']['draws_sha256']
    lengths = manifest['primary_lengths']
    prior_results = {(r['method'], r['length']): r for r in pilot['results']}
    old_metric = {(r['method'], r['length'], r['prompt_index']): r for r in original}
    new_metric = {(r['arm'], r['length'], r['prompt_index']): r for r in modified}
    assert len(old_metric) == len(original) == 500 and len(new_metric) == len(modified) == 300
    rep_metric = {(r['setting'], r['length'], r['prompt_index'], r['response_index']): r for r in rep_rows}
    assert len(rep_metric) == len(rep_rows) == 1600
    scores = score_rows(PILOT)
    score = {(r['detector'], r['method'], r['prompt_index'], r['response_index'], r['length']): r['score'] for r in scores}
    assert len(score) == len(scores)

    # Independently reconstruct each modified prompt's saved Self-BLEU to verify
    # its pairing, rather than validating only an aggregate that could hide swaps.
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    assert sacrebleu.__version__ == '2.4.3'
    tokenizer_path = PILOT/'raw/tokenizer.json'
    assert sha(tokenizer_path) == manifest['model']['tokenizer_sha256']
    sources[str(tokenizer_path.relative_to(ROOT))] = sha(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bleu = BLEU(tokenize='13a', smooth_method='exp', effective_order=True, lowercase=False)
    modified_pair_checks = 0
    for arm in ('textseal_on', 'gumbel_on'):
        batches = []
        for response in (0, 1):
            name = f'raw/other_generators/batches/{arm}_r{response}.json'
            batch = read(REPEAT/name, ablation['verified_files'][name])
            assert batch['manifest']['setting'] == manifest['arms'][arm]
            assert batch['manifest']['sampling_seed'] == manifest['seeds'][response]
            rows = {r['prompt_index']: r for r in batch['responses']}
            assert len(rows) == len(batch['responses']) == 50 and set(rows) == set(range(50))
            batches.append(rows)
        for n in lengths:
            for i in range(50):
                texts = []
                for response in (0, 1):
                    r = batches[response][i]
                    assert r['response_index'] == response and r['sampling_seed'] == manifest['seeds'][response]
                    assert len(r['token_ids']) == 1024 and digest(r['token_ids']) == r['completion_sha256']
                    rm = rep_metric[arm, n, i, response]
                    assert rm['response_id'] == r['response_id'] and rm['completion_sha256'] == r['completion_sha256']
                    texts.append(tokenizer.decode(r['token_ids'][:n], skip_special_tokens=True))
                measured = (bleu.sentence_score(texts[0], [texts[1]]).score + bleu.sentence_score(texts[1], [texts[0]]).score)/200
                close(measured, new_metric[arm, n, i]['self_bleu'])
                modified_pair_checks += 1
    assert str(bleu.get_signature()) == ablation['bleu_signature'] == pilot['bleu_signature']

    prompt_rows, vectors, absolute = [], {}, []
    rep_results = {(r['setting'], r['length']): r for r in repetition['results']}
    for setting, method in METHODS.items():
        for n in lengths:
            records = []
            for i in range(50):
                pair = [rep_metric[setting, n, i, response] for response in (0, 1)]
                data = new_metric[setting, n, i] if setting in ('textseal_on', 'gumbel_on') else old_metric[method, n, i]
                decisions = (data['detected'] if setting in ('textseal_on', 'gumbel_on') else
                             [bool(score[method, method, i, response, n]['decision']) for response in (0, 1)]) if method != 'null' else None
                r = dict(setting=setting, method=method, length=n, prompt_index=i,
                         response_ids=[q['response_id'] for q in pair], completion_sha256=[q['completion_sha256'] for q in pair],
                         self_bleu=data['self_bleu'], repeated_4gram_fraction=float(np.mean([q['repeated_4gram_fraction'] for q in pair])),
                         distinct_3=float(np.mean([q['distinct_3'] for q in pair])),
                         detected=decisions, tpr=float(np.mean(decisions)) if decisions is not None else None)
                records.append(r)
            prompt_rows.extend(records)
            metrics = METRICS if method != 'null' else METRICS[:-1]
            vectors[setting, n] = {k: np.array([r[k] for r in records]) for k in metrics}
            value = dict(setting=setting, method=method, length=n, prompt_clusters=50, response_slots=100,
                         metrics={k: paired_interval(v, draws) for k, v in vectors[setting, n].items()})
            if method != 'null':
                value['detected'] = sum(sum(r['detected']) for r in records)
            absolute.append(value)
            reference = (next(r for r in ablation['results'] if r['arm'] == setting and r['length'] == n)
                         if setting in ('textseal_on', 'gumbel_on') else prior_results[method, n])
            check_interval(value['metrics']['self_bleu'], reference['self_bleu'])
            if method != 'null':
                check_interval(value['metrics']['tpr'], reference['tpr'])
                assert value['detected'] == reference['detected']
            for k in ('repeated_4gram_fraction', 'distinct_3'):
                check_interval(value['metrics'][k], rep_results[setting, n]['metrics'][k])

    contrasts = []
    prior_contrasts = {(r['comparison'], r['length']): r for r in pilot['contrasts']}
    for view, settings in VIEWS.items():
        for setting in settings[1:-1]:
            for n in lengths:
                # Direct paired differences; do not subtract marginal interval endpoints.
                contrast = dict(view=view, baseline=setting, method=METHODS[setting], length=n,
                    prc_minus_baseline={k: paired_interval(vectors['prc', n][k]-vectors[setting, n][k], draws) for k in METRICS})
                contrasts.append(contrast)
                if view == 'native':
                    previous = prior_contrasts[f'PRC minus {NAMES[METHODS[setting]]}', n]
                    check_interval(contrast['prc_minus_baseline']['self_bleu'], previous['self_bleu_difference'])
                    check_interval(contrast['prc_minus_baseline']['tpr'], previous['tpr_difference'])
                else:
                    previous = next(r for r in repetition['contrasts'] if r['kind'] == 'prc_minus_comparator' and r['right'] == setting and r['length'] == n)
                    for k in ('repeated_4gram_fraction', 'distinct_3'):
                        check_interval(contrast['prc_minus_baseline'][k], previous['metrics'][k])

    null_path = ROOT/'outputs/comparison_redetect/baseline_comparisons.csv'
    assert sha(null_path) == pilot['shared_fpr_source_sha256']
    sources[str(null_path.relative_to(ROOT))] = sha(null_path)
    null_csv = list(csv.DictReader(null_path.open()))
    nulls = []
    for method in ('online_prc', 'textseal', 'synthid_text', 'gumbel_max'):
        for n in lengths:
            fresh = [r for r in scores if r['detector'] == method and r['method'] == 'null' and r['length'] == n]
            assert len(fresh) == 100 and {(r['prompt_index'], r['response_index']) for r in fresh} == {(i, j) for i in range(50) for j in (0, 1)}
            fp = sum(r['score']['decision'] for r in fresh)
            assert fp == prior_results[method, n]['fresh_null_false_positives']
            if method in ('synthid_text', 'gumbel_max'):
                shared = [r for r in scores if r['detector'] == method and r['method'] == 'shared_null' and r['length'] == n]
                assert len(shared) == 500 and {r['prompt_index'] for r in shared} == set(range(500))
                shared_fp = sum(r['score']['decision'] for r in shared)
            else:
                cell = next(r for r in null_csv if r['Method'] == method and int(r['T']) == n)['FPR']
                shared_fp, total = map(int, cell.split()[0].split('/'))
                assert total == 500
            previous = next(r for r in pilot['historical_shared_null_fpr'] if r['method'] == method and r['length'] == n)
            assert shared_fp == previous['false_positives']
            nulls.append(dict(method=method, length=n, reused_for_views=list(VIEWS),
                fresh_pilot=dict(false_positives=int(fp), responses=100, prompt_clusters=50),
                historical_shared=dict(false_positives=int(shared_fp), responses=500, binomial_ci95=previous['binomial_ci95']),
                detector_mask='official context mask' if method == 'synthid_text' else 'unchanged method-native rule'))
    save(OUTPUT/'prompt_metrics.json', prompt_rows)
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    sources['self_bleu/pilot.py'] = sha(ROOT/'self_bleu/pilot.py')
    result = dict(manifest_id=manifest['id'], views=VIEWS, results=absolute, contrasts=contrasts, null_counts=nulls,
        bootstrap=dict(unit='paired prompt cluster', prompts=50, responses_per_prompt=2, resamples=2000,
                       seed=manifest['analysis']['bootstrap_seed'], draws_sha256=draws_hash,
                       estimator='Mean of the 50 per-prompt PRC-minus-baseline differences',
                       ci='2.5th and 97.5th percentiles of means resampled jointly by prompt'),
        definitions=dict(self_bleu='SacreBLEU 2.4.3 symmetric sentence BLEU, 0-1; negative PRC-minus-baseline favors PRC',
                         repeated_4gram_fraction='1 - unique token 4-grams / (T-3); negative difference favors PRC',
                         distinct_3='unique token trigrams / (T-2); positive difference favors PRC',
                         tpr='Mean of the two response decisions per prompt; positive difference favors PRC',
                         native='Native generation policies: TextSeal/Gumbel off, SynthID on; all use this study\'s completion-only detectors',
                         fallback_on='TextSeal/Gumbel/SynthID on, with PRC and ordinary-sampling references unchanged'),
        verification=dict(passed=True, modified_prompt_self_bleu_values_recomputed=modified_pair_checks,
                          native_self_bleu_contrasts_reproduced=6, absolute_metric_intervals_verified=True,
                          prior_fallback_on_repetition_contrasts_reproduced=True, fresh_null_counts_recounted=True,
                          historical_null_counts_verified=True, prompt_records=len(prompt_rows)),
        null_notes=['Fallback changes generation only, so detector-specific null counts are reused in both views.',
                    'Fresh 100-response nulls contain 50 prompt clusters; historical 500 nulls overlap those prompts. Do not pool as 600 independent observations.',
                    'Gumbel-Max historical shared nulls at 400 are 2/500, despite an earlier Stage A prose claim of all zeros; machine-readable counts and CSV agree.',
                    'SynthID counts use the official context-mask detector; native describes generation policy, not the legacy 500-prompt tuple-mask scoring variant.'],
        limitations=['Exploratory marginal intervals, not multiplicity-adjusted.',
                     'Nominal p<0.001 rules are not calibrated to matched empirical FPR.',
                     'Boundary bootstrap intervals do not establish perfect detection or zero population FPR.',
                     'One 50-prompt cohort and fixed default settings/keys; not a parameter-frontier comparison.',
                     'Common fallback retains method-native initialization and RNGs; PRC is position based.'],
        sources=sources, prompt_metrics_sha256=sha(OUTPUT/'prompt_metrics.json'),
        incremental_modal_cost_usd=0, generation_attempts=0, model_inference_calls=0)
    save(OUTPUT/'summary.json', result)
    print(json.dumps({'contrasts': contrasts, 'null_counts': nulls, 'verification': result['verification']}, indent=2))


if __name__ == '__main__':
    main()
