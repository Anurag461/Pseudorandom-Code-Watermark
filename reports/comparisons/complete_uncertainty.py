"""Small aggregation of saved metrics for the alternative uncertainty table.

No token scoring, models, detectors, cloud workers, or additional generations.
Uses the original 2,000 paired prompt-bootstrap draws, seed 20260918.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data/uncertainty_supplement.json"
sources = {}


def read(name):
    content = (ROOT / name).read_bytes()
    sources[name] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def pair_means(rows, metric):
    grouped = {}
    for row in rows:
        key = row['prompt_index'], row['response_index']
        assert key not in grouped
        grouped[key] = row[metric]
    assert set(grouped) == {(p, r) for p in range(50) for r in (0, 1)}
    return np.array([(grouped[p, 0] + grouped[p, 1]) / 2 for p in range(50)])


def prompt_values(rows, metric):
    assert sorted(r['prompt_index'] for r in rows) == list(range(50))
    return np.array([r[metric] if metric != 'tpr' else np.mean(r['detected'])
                     for r in sorted(rows, key=lambda r: r['prompt_index'])])


def main():
    depth = read('outputs/self_bleu_depth/depth2_30_v1/repetition_1024.json')
    responses = read('outputs/self_bleu_repeat/matched_repetition/response_metrics.json')
    prompts = read('outputs/self_bleu_repeat/paired_comparison/prompt_metrics.json')
    off = read('reports/comparisons/data/synthid_off_1024_prompt_metrics.json')
    saved = read('reports/comparisons/data/results.json')
    draws = np.random.default_rng(20260918).integers(0, 50, (2000, 50))
    draw_hash = hashlib.sha256(json.dumps(draws.tolist(), sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()
    assert draw_hash == '5ab115a4b5c632f81fd04e9cf02b5d4f67cf68dd60cf622759f99a04fb767c77'

    def interval(values):
        assert values.shape == (50,) and np.isfinite(values).all()
        return {'mean': float(values.mean()),
                'ci95': np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}

    vectors = {}
    for metric in ('distinct_3', 'repeated_4gram_fraction'):
        for setting in ('prc', 'synthid_off'):
            vectors[setting, metric] = pair_means(
                [r for r in responses if r['setting'] == setting and r['length'] == 1024], metric)
        vectors['synthid_depth30', metric] = pair_means(depth['response_metrics'], metric)
    for metric in ('self_bleu', 'tpr'):
        vectors['prc', metric] = prompt_values(
            [r for r in prompts if r['setting'] == 'prc' and r['length'] == 1024], metric)
        vectors['synthid_off', metric] = prompt_values(off, metric)

    # Check identity with the existing exported means and saved intervals.
    checks = 0
    for (setting, metric), values in vectors.items():
        reference = next(r for r in saved if r['regime'] == '8b_full'
                         and r['setting'] == setting and r['length'] == 1024)['metrics'][metric]
        calculated = interval(values)
        assert abs(calculated['mean'] - reference['mean']) < 1e-12
        if reference['ci95'] is not None:
            assert np.allclose(calculated['ci95'], reference['ci95'], atol=1e-12, rtol=0)
        checks += 1

    absolute = [{'setting': 'synthid_depth30', 'metric': metric,
                 **interval(vectors['synthid_depth30', metric])}
                for metric in ('distinct_3', 'repeated_4gram_fraction')]
    contrasts = []
    for setting, metrics in [('synthid_depth30', ('distinct_3', 'repeated_4gram_fraction')),
                             ('synthid_off', ('distinct_3', 'repeated_4gram_fraction', 'self_bleu', 'tpr'))]:
        for metric in metrics:
            contrasts.append({'left': 'prc', 'right': setting, 'metric': metric,
                              **interval(vectors['prc', metric] - vectors[setting, metric])})
    result = {'regime': '8b_full', 'length': 1024, 'prompts': 50, 'responses': 100,
              'bootstrap': {'resamples': 2000, 'seed': 20260918, 'draws_sha256': draw_hash,
                            'unit': 'paired prompt cluster', 'ci': 'marginal 95% percentile; unadjusted'},
              'sources': sources, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'existing_metric_checks': checks,
              'absolute': absolute, 'contrasts': contrasts,
              'new_generation': 0, 'new_token_or_detector_scoring': 0, 'cloud_cost_usd': 0}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'absolute': absolute, 'contrasts': contrasts, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
