"""Local metadata-only gate. Modal is imported only AFTER explicit run approval.

Never call cloud functions directly: this entry point records one-use approval,
verifies the committed/pushed source, then starts the explicitly approved stage.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from .config import DESIGN, HERE, fingerprint, relative_root, digest_json
from .storage import write_json

# Base rates verified 2026-09-24. Refresh from modal.com/pricing before approval.
RATES = dict(H100=0.001097, core=0.0000131, GiB=0.00000222)
STAGES = {
    'experiment': dict(cpu=4, memory_GiB=16, cpu_seconds=900, gpu_workers=5,
                   gpu_seconds=3600, gpu_batch=80, estimate=[12.0, 25.0],
                   wall_time_hours=[1, 2],
                   included_calls=[
                       dict(stage='prepare', workers=1, cpu=4, memory_GiB=16, timeout_seconds=900,
                            estimate_usd=[.10,.40]),
                       dict(stage='production', workers=5, gpu='H100', cpu=4, memory_GiB=64,
                            batch=80, timeout_seconds=3600, estimate_usd=[9,22]),
                       dict(stage='score', workers=1, cpu=8, memory_GiB=16, timeout_seconds=3600,
                            estimate_usd=[.50,2])],
                   workload='10 WM keys, 160 shared codewords, 512 independent null keys; 800 WM +800 null x1024 tokens across five temperatures, prompt-free replay, primary/matched-FPR tables, 2000 crossed bootstraps, ROC and figures; no extra smoke, oracle, retry or follow-up run'),
    'numerical-diagnostic': dict(cpu=4, memory_GiB=64, cpu_seconds=0, gpu_workers=1,
                   gpu_seconds=420, gpu_batch=1, estimate=[0.25, 0.75],
                   workload='Reuse the saved failing T=1.0 null prefix; static/concat/uncached at lengths 1,4,8,16 in BF16 and FP32; 122 teacher-forced positions, no generation, no source recheck'),
    'sanity': dict(cpu=4, memory_GiB=16, cpu_seconds=600, gpu_workers=1,
                   gpu_seconds=900, gpu_batch=2, estimate=[0.50, 1.50],
                   workload='Complete T=1.8 source check (10x16); one full key; 4 WM + 4 null x64 tokens; 504 prompt-free replay positions; bounded prefix checks'),
    'prepare': dict(cpu=4, memory_GiB=16, cpu_seconds=900, gpu_workers=0,
                    gpu_seconds=0, gpu_batch=0, estimate=[0.10, 0.40],
                    workload='10 watermark keys, 160 codewords, 256 calibration + 256 held-out independent null keys; frozen 1600-text inventory'),
    'production': dict(cpu=4, memory_GiB=64, cpu_seconds=0, gpu_workers=5,
                       gpu_seconds=3600, gpu_batch=80, estimate=[9.0, 22.0],
                       workload='5 temperatures, each 160 WM +160 null x1024 tokens and completion-only replay; four generation and four replay batches per temperature'),
    'score': dict(cpu=8, memory_GiB=16, cpu_seconds=3600, gpu_workers=0,
                  gpu_seconds=0, gpu_batch=0, estimate=[0.50, 2.0],
                  workload='102400 calibration +102400 evaluation null/key pairings, 800 true-key WM scores, frozen thresholds, 2000 crossed bootstraps, CSV/PDF/PNG; no LM'),
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=HERE.parent, text=True).strip()


def quote(stage):
    spec = STAGES[stage]
    cpu_max = spec['cpu_seconds'] * (spec['cpu'] * RATES['core'] + spec['memory_GiB'] * RATES['GiB'])
    gpu_max = spec['gpu_workers'] * spec['gpu_seconds'] * (RATES['H100'] + 4 * RATES['core'] + 64 * RATES['GiB'])
    if stage == 'experiment':
        cpu_max += 3600 * (8 * RATES['core'] + 16 * RATES['GiB'])
    result = dict(stage=stage, specification=spec, rates_per_second=RATES,
                estimated_usd=spec['estimate'], metered_function_timeout_envelope_usd=cpu_max+gpu_max,
                caveat='Timeout envelope excludes image builds, container startup/teardown and storage; estimate includes a planning allowance. Not a provider-enforced dollar cap.',
                remaining_budget_last_stated=35, remaining_budget_unreconciled=True,
                budget_after_estimate_if_35_available=[35-spec['estimate'][1],35-spec['estimate'][0]],
                fingerprint=fingerprint(), git_commit=git('rev-parse', 'HEAD'),
                volume='prc-data', root=relative_root(), profile='new-prc-watermark',
                retries=0, automatic_next_stage=False,
                sanity_skip_rule='Skip (not pass) initial package if concrete quote exceeds $5')
    if stage == 'experiment':
        from .validation import evidence
        result['validation'] = evidence()
        result['included_sequence'] = ['prepare', 'production x5', 'score']
        result['automatic_next_stage'] = 'Only the seven calls explicitly included in this package'
        # Historical billing is not a reusable budget. The quote above leaves the
        # balance unreconciled; every new approval must include a fresh review.
    return result


def validate_approval(approval, stage, current_quote):
    if approval.get('approved') is not True or not approval.get('user_approval_text'):
        raise ValueError('This particular paid run needs explicit user approval; a quote is not approval')
    for name in ('stage', 'fingerprint', 'git_commit', 'profile'):
        if approval.get(name) != current_quote[name]:
            raise ValueError(f'Approval does not match current {name}')
    if approval.get('quote_sha256') != digest_json(current_quote):
        raise ValueError('Approval quote changed')
    if not approval.get('run_id') or not approval.get('billing_review'):
        raise ValueError('Missing run identity or recent billing review')
    if approval.get('max_estimated_usd', 0) < current_quote['estimated_usd'][1]:
        raise ValueError('Quoted estimate exceeds this approval')
    if stage == 'sanity' and current_quote['estimated_usd'][1] > 5:
        raise ValueError('Initial sanity quote exceeds $5: SKIP it; do not launch')
    if stage == 'production' and approval.get('sanity_status') not in ('passed', 'skipped_cost_over_5'):
        raise ValueError('Production needs passed sanity or explicit documented >$5 cost skip')
    if stage == 'experiment':
        if approval.get('validation_policy') != current_quote['validation']['policy']:
            raise ValueError('Approve the disclosed completed-controls validation policy')
        if approval.get('included_sequence') != current_quote['included_sequence']:
            raise ValueError('Approval must explicitly cover preparation, all five GPU calls, and scoring')
        if approval.get('oracle_prompt_context', False):
            raise ValueError('Oracle diagnostic is outside this experiment package')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['quote', 'launch'])
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--oracle-prompt-context', action='store_true',
                        help='CPU score-stage diagnostic using saved generation probabilities only')
    args = parser.parse_args()
    q = quote(args.stage)
    if args.command == 'quote':
        print(json.dumps(q, indent=2))
        return
    if args.approval is None:
        parser.error('--approval is required before any Modal import/build')
    approval = json.loads(args.approval.read_text())
    validate_approval(approval, args.stage, q)
    if args.oracle_prompt_context != bool(approval.get('oracle_prompt_context', False)):
        raise ValueError('Oracle setting is not covered by this exact approval')
    if git('branch', '--show-current') != 'cryptoanalysis-redetection':
        raise ValueError('Wrong branch')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Commit setup changes before a paid launch')
    tracked = set(git('ls-files').splitlines())
    required = [*HERE.glob('*.py'), *HERE.glob('*.json'), *HERE.glob('vendor/*')]
    if any(str(p.relative_to(HERE.parent)) not in tracked for p in required if p.is_file()):
        raise ValueError('Untracked experiment setup file; commit/push first')
    subprocess.run(['git', 'merge-base', '--is-ancestor', q['git_commit'],
                    'origin/cryptoanalysis-redetection'], cwd=HERE.parent, check=True)
    if os.environ.get('MODAL_PROFILE') != q['profile']:
        raise ValueError('Set MODAL_PROFILE=new-prc-watermark explicitly')
    receipt_dir = HERE / 'local_runs'
    receipt_dir.mkdir(exist_ok=True)
    # Reserve before importing Modal. Any failure consumes this one-use approval.
    receipt = receipt_dir / (digest_json(approval) + '.json')
    with receipt.open('x') as f:
        json.dump(dict(status='reserved', quote=q, approval=approval,
                       command=sys.argv, reserved_at=datetime.now(timezone.utc).isoformat()), f, indent=2)
    from .cloud import dispatch
    result = dispatch(args.stage, q, approval, args.oracle_prompt_context)
    write_json(receipt, dict(status='completed', quote=q, approval=approval, result=result,
                            completed_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
