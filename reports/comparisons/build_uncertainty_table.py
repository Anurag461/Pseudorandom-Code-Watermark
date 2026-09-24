"""Format a second table from saved intervals and paired comparisons."""
import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ORDER = ['null', 'prc', 'synthid_depth2', 'synthid_depth10', 'synthid_depth30',
         'textseal_on', 'gumbel_on', 'synthid_off', 'textseal_off', 'gumbel_off']
LABELS = ['Unwatermarked', r'\textbf{PRC (ours)}', 'SynthID depth 2',
          'SynthID depth 10', 'SynthID depth 30', r'TextSeal $\alpha=.1$',
          'Gumbel-Max', 'SynthID depth 10', r'TextSeal $\alpha=.1$', 'Gumbel-Max']
METRICS = [('tpr', 100, 0), ('self_bleu', 1, 5),
           ('repeated_4gram_fraction', 100, 2), ('distinct_3', 1, 4)]


def main():
    levels = {}
    for r in csv.DictReader((BASE / 'data/absolute_results.csv').open()):
        if r['regime'] == '8b_full' and r['length'] == '1024':
            levels[r['setting'], r['metric']] = {
                'mean': float(r['mean']),
                'ci95': [float(r[k]) for k in ('ci95_low', 'ci95_high')] if r['ci95_low'] else None}
    paired = {}
    for r in csv.DictReader((BASE / 'data/paired_contrasts.csv').open()):
        if r['regime'] == '8b_full' and r['length'] == '1024' and r['left'] == 'prc':
            paired[r['right'], r['metric']] = [float(r[k]) for k in ('ci95_low', 'ci95_high')]
    supplement = json.loads((BASE / 'data/uncertainty_supplement.json').read_text())
    for r in supplement['absolute']:
        assert abs(levels[r['setting'], r['metric']]['mean'] - r['mean']) < 1e-12
        levels[r['setting'], r['metric']]['ci95'] = r['ci95']
    for r in supplement['contrasts']:
        assert r['left'] == 'prc'
        paired[r['right'], r['metric']] = r['ci95']

    template = (BASE / 'tables/repeat_fallback_1024.tex').read_text()
    caption_end = 'row rate $99/100$, and posterior-mean/Hoeffding detection.}'
    caption_extension = r'''row rate $99/100$, and posterior-mean/Hoeffding detection.
Small upper and lower scripts give the upper and lower limits of 95\%
confidence intervals from 2,000 bootstrap resamples of the 50 prompts,
keeping each response pair together. The PRC row is bold for reference;
other values are bold when the paired 95\% interval for their difference
from PRC excludes zero, without adjustment for multiple comparisons.
At 100\% TPR, bootstrap intervals collapse to 100--100; this does not
imply perfect population detection.}'''
    assert template.count(caption_end) == 1
    template = template.replace(caption_end, caption_extension)
    template = template.replace('tab:matched-selfbleu-levels', 'tab:matched-selfbleu-uncertainty')
    start = template.index('Unwatermarked\n')
    end = template.index('\\bottomrule', start)
    lines = []
    evidence = []
    for setting, label in zip(ORDER, LABELS):
        if setting in ('synthid_depth2', 'synthid_off'):
            mode = 'on' if setting == 'synthid_depth2' else 'off'
            lines += [r'\midrule', r'\multicolumn{5}{l}{\textbf{Repeat fallback ' + mode + r'}} \\']
        cells = []
        for metric, scale, digits in METRICS:
            if setting == 'null' and metric == 'tpr':
                cells.append('--')
                continue
            value = levels[setting, metric]
            assert value['ci95'] is not None
            difference = None if setting == 'prc' else paired[setting, metric]
            significant = difference is not None and (difference[0] > 0 or difference[1] < 0)
            bold = setting == 'prc' or significant
            args = ''.join('{' + f'{v * scale:.{digits}f}' + '}'
                           for v in (value['mean'], *value['ci95']))
            cells.append(('\\wmcibf' if bold else '\\wmci') + args)
            evidence.append({'setting': setting, 'metric': metric, **value,
                             'prc_minus_method_ci95': difference,
                             'significant_vs_prc': significant, 'bold': bold})
        lines += [label + '\n& ' + ' & '.join(cells) + r' \\', '']
    template = template[:start] + '\n'.join(lines) + '\n' + template[end:]
    macros = r'''% CI arguments: estimate, lower bound, upper bound.
\providecommand{\wmci}[3]{\ensuremath{\mbox{#1}\;{}^{\mbox{\scriptsize #3}}_{\mbox{\scriptsize #2}}}}
\providecommand{\wmcibf}[3]{\ensuremath{\mbox{\bfseries #1}\;{}^{\mbox{\scriptsize\bfseries #3}}_{\mbox{\scriptsize\bfseries #2}}}}
'''
    template = template.replace('\\begin{table*}', macros + '\\begin{table*}', 1)
    (BASE / 'tables/repeat_fallback_1024_uncertainty.tex').write_text(template)
    wrapper = (BASE / 'tables/repeat_fallback_1024_preview.tex').read_text()
    wrapper = wrapper.replace('repeat_fallback_1024.tex', 'repeat_fallback_1024_uncertainty.tex')
    (BASE / 'tables/repeat_fallback_1024_uncertainty_preview.tex').write_text(wrapper)
    sources = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in (
        'data/absolute_results.csv', 'data/paired_contrasts.csv', 'data/uncertainty_supplement.json',
        'tables/repeat_fallback_1024.tex')}
    (BASE / 'data/uncertainty_table_cells.json').write_text(
        json.dumps({'sources': sources, 'cells': evidence}, indent=2, sort_keys=True) + '\n')
    print('Significant values:', [(r['setting'], r['metric']) for r in evidence if r['significant_vs_prc']])


if __name__ == '__main__':
    main()
