"""Independent CPU postprocessing of frozen TextSeal/Gumbel ablation artifacts.

Run from the repository root with PYTHONPATH=. after downloading other_generators.
Kept with the experiment record so the dispatched, hash-pinned worker is unchanged.
No generation, detector calls, or Modal dispatch occur here.
"""
import json
from pathlib import Path

import numpy as np

from self_bleu.config import digest
from self_bleu.pilot import paired_interval
from self_bleu.repeat import ROOT, trajectory_summary, validate
from self_bleu.validation import save, sha


SETUP = Path(__file__).resolve().parent


def repeat_mask(tokens, prompt):
    """Native three-token contexts: prompt suffix initially; empty history."""
    history, mask = set(), []
    context = tuple(prompt[-3:])
    assert len(context) == 3
    for token in tokens:
        mask.append(context in history)
        history.add(context)
        context = (*context[1:], token)
    return mask


def describe(row, mask, enabled, divergence):
    positions = [i for i, repeated in enumerate(mask) if repeated]
    fallbacks = positions if enabled else []
    return dict(response_id=row['response_id'], completion_sha256=digest(row['token_ids']),
                fallback_enabled=enabled, repeat_positions=positions, fallback_positions=fallbacks,
                repeat_count=len(positions), fallback_count=len(fallbacks),
                first_repeat_position=next(iter(positions), None),
                first_fallback_position=next(iter(fallbacks), None), first_token_divergence=divergence)


def main():
    read = lambda path: json.loads(path.read_text())
    manifest = read(SETUP/'manifest.json')
    validate(manifest)
    report = read(SETUP/'other_generators_report.json')
    assert report['passed'] and report['manifest_id'] == manifest['id']
    assert report['stage'] == 'other_generators'
    source_hashes = {name: sha(SETUP/name) for name in ('manifest.json', 'other_generators_report.json')}
    source_hashes['check_followup_trajectories.py'] = sha(Path(__file__))
    for name, expected in report['files'].items():
        relative = 'raw/other_generators/'+name
        assert sha(SETUP/relative) == expected, relative
        source_hashes[relative] = expected
    inputs_path = ROOT/'outputs/self_bleu_pilot/stage_a_v2/inputs.json'
    assert sha(inputs_path) == manifest['reference_files'][str(inputs_path.relative_to(ROOT))]
    original = {(r['method'], r['prompt_index'], r['response_index']): r for r in read(inputs_path)}
    prompts = [json.loads(line)['prompt_tokens'] for line in (ROOT/'prompts.jsonl').read_text().splitlines()]
    draws = np.random.default_rng(manifest['analysis']['bootstrap_seed']).integers(0, 50, (2000, 50))
    summaries, compact = {}, []
    for arm, method in [('textseal_on', 'textseal'), ('gumbel_on', 'gumbel_max')]:
        pairs, control_checks = [], 0
        for response in (0, 1):
            directory = SETUP/'raw/other_generators'
            batch = read(directory/'batches'/f'{arm}_r{response}.json')
            controls = read(directory/'controls'/f'{arm}_r{response}.json')['responses']
            assert batch['manifest']['setting'] == manifest['arms'][arm]
            assert batch['manifest']['sampling_seed'] == manifest['seeds'][response]
            assert [r['prompt_index'] for r in batch['responses']] == manifest['prompt_indices']
            assert [r['prompt_index'] for r in controls] == manifest['prompt_indices']
            for new, control in zip(batch['responses'], controls):
                i = new['prompt_index']
                old = original[(method, i, response)]
                assert new['response_index'] == response and new['sampling_seed'] == manifest['seeds'][response]
                assert len(old['token_ids']) == len(new['token_ids']) == manifest['length']
                assert digest(new['token_ids']) == new['completion_sha256']
                assert digest(old['token_ids']) == manifest['references'][f'{method}/{response}']['completion_sha256'][i]
                old_mask, new_mask = [repeat_mask(r['token_ids'], prompts[i]) for r in (old, new)]
                n = manifest['control_tokens']
                assert control['token_ids'] == old['token_ids'][:n]
                trace = control['generation_diagnostics']
                assert trace['repeated_context'] == old_mask[:n]
                assert trace['fallback_applied'] == [False]*n and trace['first_fallback_position'] is None
                control_checks += 1
                divergence = next((j for j, (a, b) in enumerate(zip(old['token_ids'], new['token_ids'])) if a != b), None)
                before, after = describe(old, old_mask, False, divergence), describe(new, new_mask, True, divergence)
                first = before['first_repeat_position']
                through = manifest['length'] if divergence is None else divergence+1
                recorded = new['generation_diagnostics']
                checks = {
                    'no_divergence_before_first_repeat': divergence is None or (first is not None and divergence >= first),
                    'no_repeat_reference_stays_identical': first is not None or divergence is None,
                    'first_repeat_positions_agree': first == after['first_repeat_position'],
                    'repeat_history_agrees_through_divergence': old_mask[:through] == new_mask[:through],
                    'modified_repeat_trace_matches_reconstruction': recorded['repeated_context'] == new_mask,
                    'modified_fallback_trace_matches_repeats': recorded['fallback_applied'] == new_mask
                        and recorded['first_fallback_position'] == after['first_repeat_position'],
                }
                pairs.append(dict(arm=arm, prompt_index=i, response_index=response,
                                  sampling_seed=manifest['seeds'][response], first_token_divergence=divergence,
                                  original=before, modified=after, checks=checks, passed=all(checks.values())))
        pairs.sort(key=lambda r: (r['prompt_index'], r['response_index']))
        assert len(pairs) == control_checks == 100
        summary = dict(passed=all(r['passed'] for r in pairs), response_pairs=len(pairs),
                       native_prefix_traces_verified=control_checks,
                       checks_passed={k: sum(r['checks'][k] for r in pairs) for k in pairs[0]['checks']},
                       summaries=trajectory_summary(pairs, manifest['primary_lengths']),
                       failed_pairs=[{k: r[k] for k in ('prompt_index', 'response_index', 'checks')} for r in pairs if not r['passed']])
        for prefix in summary['summaries']:
            n = prefix['length']
            delta = [sum(p < n for p in r['modified']['repeat_positions'])-sum(p < n for p in r['original']['repeat_positions']) for r in pairs]
            prefix['descriptive_repeat_count_delta'] = paired_interval(np.asarray(delta).reshape(50, 2).mean(axis=1), draws)
        for pair in pairs:
            row = {k: pair[k] for k in ('arm', 'prompt_index', 'response_index', 'sampling_seed', 'first_token_divergence', 'passed')}
            for policy in ('original', 'modified'):
                values = pair[policy]
                row[policy] = {k: v for k, v in values.items() if k not in ('repeat_positions', 'fallback_positions')}
                row[policy]['counts_by_prefix'] = {str(n): {event+'_count': sum(p < n for p in values[event+'_positions'])
                                                        for event in ('repeat', 'fallback')} for n in manifest['primary_lengths']}
            compact.append(row)
        save(SETUP/'raw'/f'{arm}_trajectory_pairs.json', pairs)
        summaries[arm] = summary
    output = dict(manifest_id=manifest['id'], passed=all(s['passed'] for s in summaries.values()),
                  position_convention='zero-based generated-token positions; null if event never occurs',
                  original_trace_provenance='Reconstructed from saved tokens and prompt suffix with empty per-response context history; verified against native GPU prefixes.',
                  modified_trace_provenance='Recorded full generation traces independently checked against token-based reconstruction.',
                  detector_inputs='Prompts used only to reconstruct generation contexts here; unchanged detectors receive completion tokens only.',
                  repeat_count_intervals='Additional descriptive diagnostics, not predeclared primary endpoints.',
                  source_hashes=source_hashes, arms=summaries)
    save(SETUP/'followup_response_diagnostics.json', dict(manifest_id=manifest['id'], position_convention=output['position_convention'], rows=compact))
    save(SETUP/'followup_trajectory.json', output)
    print(json.dumps(output, indent=2))
    if not output['passed']:
        raise ValueError('Full-trajectory validation failed; inspect diagnostics before interpretation')


if __name__ == '__main__':
    main()
