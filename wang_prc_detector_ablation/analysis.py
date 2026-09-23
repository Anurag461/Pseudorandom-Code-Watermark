"""CPU-only scoring/reporting stage. Never imported by the local launch planner."""
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .config import DESIGN, TEMPERATURES, fingerprint, digest_json, sha256, seed
from .storage import load_arrays, write_json, save_arrays, exists
from .wang import token_bits, hard_count, hard_threshold, soft_evidence, soft_score, calibrate, decisions
from .hierarchy import bit_entropy

LABELS = {
    'wang_published': 'Wang hard detector (published threshold)',
    'posterior_standard': 'Posterior detector (standard threshold, FPR target 1e-3)',
    'hard_matched': 'Wang hard statistic @ matched FPR',
    'posterior_matched': 'Posterior statistic @ matched FPR',
}


def read_trace(root, sample):
    arrays, metadata = load_arrays(root / 'traces' / (sample['id'] + '.npz'),
                                  dict(sample=sample, fingerprint=fingerprint(), context='completion-only'))
    bits = token_bits(arrays['tokens'], DESIGN['bits'])
    if not np.array_equal(bits, arrays['observed_bits']):
        raise ValueError('Saved token IDs and hierarchical bits disagree')
    if sample['source'] == 'wm':
        if metadata['key_sha256'] != sha256(root / 'keys' / f"wm_{sample['group']:03d}.npz"):
            raise ValueError('Watermark trace/key binding changed')
        if metadata['codeword_sha256'] != sha256(root / metadata['codeword']):
            raise ValueError('Watermark trace/codeword binding changed')
    if arrays['replay_p1'].shape != bits.shape or not np.isnan(arrays['replay_p1'][0]).all():
        raise ValueError('First-token or trace shape invariant failed')
    evidence = soft_evidence(bits, arrays['replay_p1'])
    return arrays, metadata, bits.ravel(), evidence


def score_nulls(root, samples, split):
    """Score text against each independent key, bounded to 16 text rows in RAM."""
    path = root / 'scores' / f'null_{split}.npz'
    trace_hashes = {s['id']: sha256(root / 'traces' / (s['id'] + '.npz')) for s in samples}
    key_hashes = {f'{split}_{k:03d}': sha256(root / 'keys' / f'{split}_{k:03d}.npz') for k in range(256)}
    identity = dict(fingerprint=fingerprint(), ids=[s['id'] for s in samples], split=split,
                    trace_hashes=trace_hashes, key_hashes=key_hashes)
    if exists(path):
        return load_arrays(path, identity)[0], identity
    traces = [read_trace(root, s) for s in samples]
    bits = np.stack([x[2] for x in traces])
    evidence = np.stack([x[3] for x in traces])
    out = {k: np.empty((len(samples), 256), dtype=np.float64) for k in ('H', 'S', 'V', 'Z', 'tau')}
    out['no_evidence'] = np.empty((len(samples), 256), dtype=bool)
    out['standard'] = np.empty((len(samples), 256), dtype=bool)
    for k in range(256):
        key, _ = load_arrays(root / 'keys' / f'{split}_{k:03d}.npz',
                             dict(domain=split, group=k, fingerprint=fingerprint()))
        for begin in range(0, len(samples), 16):
            sl = slice(begin, begin + 16)
            out['H'][sl, k] = hard_count(bits[sl], key)
            soft = soft_score(evidence[sl], key)
            for name in soft:
                out[name][sl, k] = soft[name]
    save_arrays(path, out, identity)
    return out, identity


def bootstrap_weights(groups, prompts=16, replicates=2000, domain='wm'):
    rng = np.random.default_rng(seed('bootstrap', domain))
    g = rng.multinomial(groups, np.ones(groups) / groups, replicates)
    p = rng.multinomial(prompts, np.ones(prompts) / prompts, replicates)
    return (g[:, :, None] * p[:, None, :]).reshape(replicates, -1) / (groups * prompts)


def interval(values, weights):
    return np.quantile(weights @ np.asarray(values), [.025, .975]).tolist()


def roc(watermarked, null):
    """Equal WM text weight; each null text's weight split across its keys."""
    wm, neg = np.asarray(watermarked).ravel(), np.asarray(null).ravel()
    scores = np.concatenate((wm, neg))
    order = np.argsort(-scores, kind='stable')
    positives = np.concatenate((np.ones(len(wm)) / len(wm), np.zeros(len(neg))))[order]
    negatives = np.concatenate((np.zeros(len(wm)), np.ones(len(neg)) / len(neg)))[order]
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)-1]
    tpr, fpr = np.r_[0, np.cumsum(positives)[ends]], np.r_[0, np.cumsum(negatives)[ends]]
    # trapezoids assign half credit to ties, including no-evidence (-inf).
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows for {path}')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_record(root, s):
    arrays, _, bits, evidence = read_trace(root, s)
    agreement = None
    if s['source'] == 'wm':
        word = load_arrays(root / 'codewords' / f"g{s['group']:02d}_p{s['prompt']:02d}.npz")[0]['codeword']
        agreement = float(np.mean(word == bits))
    return dict(sample_id=s['id'], source=s['source'], group=s['group'], prompt=s['prompt'],
                temperature=s['temperature'], split=s['split'],
                generation_seed=s['seed'], config_fingerprint=fingerprint(),
                trace_sha256=sha256(root / 'traces' / (s['id'] + '.npz')),
                average_full_vocab_entropy=float(np.nanmean(arrays['replay_entropy'])),
                mean_hierarchical_bit_entropy=float(np.nanmean(bit_entropy(arrays['replay_p1']))),
                generation_full_vocab_entropy=float(np.mean(arrays['generation_entropy'])),
                generation_hierarchical_bit_entropy=float(np.mean(bit_entropy(arrays['generation_p1']))),
                entropy_units='full_vocab=nats;binary=bits', bit_agreement=agreement)


def scored_record(base, key_id, pairing, hard, soft, dec, thresholds):
    finite_z = float(soft['Z']) if np.isfinite(soft['Z']) else None
    return {**base, 'pairing_id': base['sample_id'] + ':' + key_id, 'scoring_key': key_id,
            'pairing': pairing, 'hard_violation_count': int(hard),
            'hard_score': (DESIGN['r'] - 2 * float(hard)) / np.sqrt(DESIGN['r']),
            'S': float(soft['S']), 'V': float(soft['V']), 'Z': finite_z,
            'no_evidence': bool(soft['no_evidence']), 'posterior_S_threshold': float(soft['tau']),
            'wang_published_threshold': hard_threshold(DESIGN['r']),
            'posterior_standard_Z_threshold': float(np.sqrt(2 * np.log(1000))),
            'hard_matched_threshold': thresholds['hard']['cutoff'],
            'posterior_matched_threshold': thresholds['posterior']['cutoff'],
            **{k: bool(v) for k, v in dec.items()}}


def run(root, provenance, oracle=False):
    root = Path(root)
    begun = time.monotonic()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['fingerprint'] != fingerprint() or manifest['smoke']:
        raise ValueError('Production-only scoring requires exact prepared design')
    samples = manifest['inventory']
    if len(samples) != 1600 or len({s['id'] for s in samples}) != 1600:
        raise ValueError('Production inventory incomplete')
    calibration = [s for s in samples if s['split'] == 'calibration']
    evaluation = [s for s in samples if s['split'] == 'evaluation']
    wm = [s for s in samples if s['source'] == 'wm']
    null_cal, cal_identity = score_nulls(root, calibration, 'calibration')
    thresholds = calibrate(null_cal['H'], null_cal['Z'])
    # This file is durably written BEFORE scoring any WM output or evaluation null.
    frozen_path = root / 'threshold_calibration.json'
    frozen_content = dict(thresholds=thresholds, identity=cal_identity,
                          design_fingerprint=fingerprint(), pooled_temperatures=list(TEMPERATURES))
    frozen_hash = digest_json(frozen_content)
    if frozen_path.exists():
        old = json.loads(frozen_path.read_text())
        if old['sha256'] != frozen_hash or digest_json(old['content']) != frozen_hash:
            raise ValueError('Refusing to retune an already frozen calibration')
    else:
        write_json(frozen_path, dict(content=frozen_content, sha256=frozen_hash,
                                    frozen_at=datetime.now(timezone.utc).isoformat()))
    null_eval, _ = score_nulls(root, evaluation, 'evaluation')
    wm_scores, wm_rows, oracle_rows = [], [], []
    for s in wm:
        arrays, _, bits, evidence = read_trace(root, s)
        key_id = f"wm_{s['group']:03d}"
        key = load_arrays(root / 'keys' / (key_id + '.npz'))[0]
        h, soft = int(hard_count(bits, key)), soft_score(evidence, key)
        dec = decisions(h, soft, DESIGN['r'], thresholds)
        base = base_record(root, s)
        wm_rows.append(scored_record(base, key_id, 'true-watermark-key', h, soft, dec, thresholds))
        wm_scores.append(dict(H=h, **soft, **dec))
        if oracle:
            evidence_oracle = soft_evidence(bits.reshape(-1, DESIGN['bits']), arrays['generation_p1'], False)
            oscore = soft_score(evidence_oracle, key)
            oracle_rows.append({**base, 'label': 'oracle-context diagnostic',
                               'first_token_included': True, 'S': float(oscore['S']),
                               'V': float(oscore['V']),
                               'Z': float(oscore['Z']) if np.isfinite(oscore['Z']) else None,
                               'standard_decision': bool(oscore['standard'])})
    wm_all = {k: np.asarray([v[k] for v in wm_scores]) for k in wm_scores[0]}
    eval_dec = decisions(null_eval['H'], null_eval, DESIGN['r'], thresholds)
    cal_dec = decisions(null_cal['H'], null_cal, DESIGN['r'], thresholds)
    summary, curves, paired = [], {}, []
    weights_wm, weights_null = bootstrap_weights(10), bootstrap_weights(5, domain='null')
    for temp in TEMPERATURES:
        wi = np.array([s['temperature'] == temp for s in wm])
        ni = np.array([s['temperature'] == temp for s in evaluation])
        # Inventory is group-major, then prompt-major within each temperature.
        for name in LABELS:
            ishard = name in ('wang_published', 'hard_matched')
            wscore = -wm_all['H'][wi] if ishard else wm_all['Z'][wi]
            nscore = -null_eval['H'][ni] if ishard else null_eval['Z'][ni]
            fpr_curve, tpr_curve, auc = roc(wscore, nscore)
            curves[(temp, 'hard' if ishard else 'posterior')] = (fpr_curve, tpr_curve)
            values = wm_all[name][wi].astype(float)
            null_values = eval_dec[name][ni].mean(axis=1)
            tlo, thi = interval(values, weights_wm)
            flo, fhi = interval(null_values, weights_null)
            threshold = (hard_threshold(DESIGN['r']) if name == 'wang_published' else
                         float(np.sqrt(2 * np.log(1000))) if name == 'posterior_standard' else
                         thresholds['hard']['cutoff'] if name == 'hard_matched' else
                         thresholds['posterior']['cutoff'])
            summary.append(dict(temperature=temp, detector=name, label=LABELS[name],
                role='primary' if name in ('wang_published', 'posterior_standard') else 'matched-FPR',
                N=int(wi.sum()), TPR=float(values.mean()), TPR_CI_low=tlo, TPR_CI_high=thi,
                null_N=int(ni.sum()), null_key_pairings=int(ni.sum())*256,
                realized_FPR=float(null_values.mean()), FPR_CI_low=flo, FPR_CI_high=fhi,
                threshold=threshold, rule='H <= cutoff' if ishard else 'V > 0 and Z >= cutoff',
                AUC=auc, threshold_freeze_sha256=frozen_hash,
                CI='crossed group x prompt; frozen thresholds; null conditional on fixed evaluation-key pool'))
        for role, a, b in [('primary', 'wang_published', 'posterior_standard'),
                           ('matched-FPR', 'hard_matched', 'posterior_matched')]:
            delta = wm_all[b][wi].astype(float) - wm_all[a][wi].astype(float)
            lo, hi = interval(delta, weights_wm)
            paired.append(dict(temperature=temp, comparison=role, posterior_minus_hard=float(delta.mean()),
                               CI_low=lo, CI_high=hi, N=int(wi.sum())))
    csv_write(root / 'results_summary.csv', summary)
    csv_write(root / 'paired_differences.csv', paired)
    mechanism = []
    for temp in TEMPERATURES:
        rows = [r for r in wm_rows if r['temperature'] == temp]
        row = dict(temperature=temp, N=len(rows))
        for name in ('average_full_vocab_entropy', 'mean_hierarchical_bit_entropy',
                     'generation_full_vocab_entropy', 'generation_hierarchical_bit_entropy',
                     'bit_agreement', 'hard_score', 'Z'):
            values = np.array([r[name] for r in rows if r[name] is not None])
            row[name+'_mean'] = float(values.mean()) if len(values) else None
            row[name+'_q05'] = float(np.quantile(values,.05)) if len(values) else None
            row[name+'_q95'] = float(np.quantile(values,.95)) if len(values) else None
        row['posterior_no_evidence_N'] = sum(r['no_evidence'] for r in rows)
        mechanism.append(row)
    csv_write(root / 'mechanism_summary.csv', mechanism)
    discussion = ['# Low-temperature detector comparison', '',
        'Paired differences below are posterior minus hard, in percentage points. ',
        'Intervals resample key groups and prompts; matched thresholds stay frozen. ',
        'The primary comparison measures actual methods at different operating points. ',
        'Matched-FPR results and held-out ROC/AUC assess scoring quality separately.', '']
    for p in paired:
        if p['temperature'] > 1.4:
            continue
        discussion.append(f"- T={p['temperature']:.1f}, {p['comparison']}: "
                          f"{100*p['posterior_minus_hard']:+.2f} pp "
                          f"(95% CI {100*p['CI_low']:+.2f} to {100*p['CI_high']:+.2f}).")
    discussion += ['', 'Consult both realized FPR columns and AUC in results_summary.csv before interpreting a TPR gain. ',
        'A gain confined to published thresholds does not establish superior posterior information; ',
        'a matched-FPR gain supported by ROC is stronger evidence. Oracle results cannot support the headline.']
    (root / 'interpretation.md').write_text('\n'.join(discussion)+'\n')
    for role, filename in [('primary', 'primary_methods_summary.csv'), ('matched-FPR', 'matched_fpr_summary.csv')]:
        wide = []
        for temp in TEMPERATURES:
            row = dict(temperature=temp, N=160)
            for s in summary:
                if s['role'] == role and s['temperature'] == temp:
                    for column in ('TPR', 'TPR_CI_low', 'TPR_CI_high', 'realized_FPR', 'FPR_CI_low',
                                   'FPR_CI_high', 'threshold', 'AUC'):
                        row[s['detector'] + '_' + column] = s[column]
            wide.append(row)
        csv_write(root / filename, wide)

    def records():
        yield from wm_rows
        for split, ss, scores, dec in [('calibration', calibration, null_cal, cal_dec),
                                      ('evaluation', evaluation, null_eval, eval_dec)]:
            for i, sample in enumerate(ss):
                base = base_record(root, sample)
                for k in range(256):
                    soft = {name: value[i, k] for name, value in scores.items() if name != 'H'}
                    yield scored_record(base, f'{split}_{k:03d}', 'independent-null-key', scores['H'][i, k],
                                        soft, {name: value[i, k] for name, value in dec.items()}, thresholds)
    # Stream the cross products to disk; no list of 205,600 dictionaries in RAM.
    stream = iter(records())
    first = next(stream)
    with (root / 'per_sample_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(stream)
    if oracle:
        csv_write(root / 'oracle_diagnostic.csv', oracle_rows)
    plots(root, summary, wm_rows, curves)
    timing = dict(seconds=time.monotonic()-begun, provenance=provenance, threshold_freeze_sha256=frozen_hash,
                  wm_N=800, calibration_null_N=400, evaluation_null_N=400,
                  calibration_pairs=102400, evaluation_pairs=102400)
    write_json(root / 'scoring_timing.json', timing)
    (root / 'README.md').write_text(
        '# Wang-channel Qwen3-8B-Base detector ablation\n\n'
        'Primary: published Wang versus standard prompt-free posterior. Secondary: matched-FPR statistics.\n'
        'See results_summary.csv, paired_differences.csv and threshold_calibration.json.\n'
        'N counts unique texts, never text/key pairs. Null intervals condition on the frozen held-out key pool.\n'
        'No Figure 5 reproduction claim is made for this different model. Oracle rows are diagnostic only.\n\n'
        f'Config `{fingerprint()}`; scoring commit `{provenance["git_commit"]}`.\n'
        'Exact launch commands/approvals: run_ledger/; GPU timings: timing_*.json; CPU timing: scoring_timing.json.\n'
        'Cost ledger gives metered-runtime estimates; provider billing must be attached separately.\n')
    return dict(summary=summary, paired=paired, timing=timing)


def plots(root, summary, wm_rows, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for role, filename in [('primary', 'tpr_vs_temperature'), ('matched-FPR', 'tpr_matched_fpr_vs_temperature')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in LABELS:
            rows = [s for s in summary if s['detector'] == name and s['role'] == role]
            if not rows:
                continue
            y = np.array([s['TPR'] for s in rows])
            ax.errorbar([s['temperature'] for s in rows], y,
                        yerr=[y-np.array([s['TPR_CI_low'] for s in rows]),
                              np.array([s['TPR_CI_high'] for s in rows])-y],
                        marker='o', capsize=3, label=LABELS[name])
        ax.set(xlabel='Generation temperature', ylabel='True positive rate', ylim=(-.02, 1.02))
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(root / f'{filename}.{ext}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, temp in zip(axes, TEMPERATURES[:3]):
        for name in ('hard', 'posterior'):
            x, y = curves[(temp, name)]
            ax.plot(x, y, label=name)
        ax.set(xscale='symlog', xlim=(0, 1), xlabel='Held-out false positive rate', title=f'T={temp:.1f}')
        ax.axvline(.001, color='gray', linestyle=':')
        ax.legend()
    axes[0].set_ylabel('True positive rate')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'roc_low_temperature.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ('hard_score', 'Z')):
        for temp in TEMPERATURES:
            rows = [r for r in wm_rows if r['temperature'] == temp and r[name] is not None]
            ax.scatter([r['mean_hierarchical_bit_entropy'] for r in rows],
                       [r[name] for r in rows], s=10, alpha=.45, label=f'T={temp:.1f}')
        ax.set(xlabel='Mean completion-only hierarchical bit entropy (bits)', ylabel=name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(root / f'score_vs_entropy.{ext}', dpi=180)
    plt.close(fig)
