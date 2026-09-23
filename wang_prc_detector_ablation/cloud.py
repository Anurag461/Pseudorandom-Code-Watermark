"""Paid Modal functions. Use launch.py's approval gate, never direct invocation."""
import json
import time
from pathlib import Path
import modal
from .config import HERE, relative_root

app = modal.App('wang-prc-detector-ablation')
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.4.0', 'transformers==4.51.3', 'tokenizers==0.21.1',
                 'safetensors==0.4.5', 'huggingface_hub==0.30.2', 'scipy==1.14.1',
                 'galois==0.4.2', 'numba==0.59.1', 'numpy==1.26.0', 'matplotlib==3.9.2')
    .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
          'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': '/workspace',
          'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
          'OMP_NUM_THREADS': '4', 'MPLCONFIGDIR': '/tmp/matplotlib',
          'NUMBA_CACHE_DIR': '/tmp/numba'})
    .add_local_dir(HERE, '/workspace/wang_prc_detector_ablation',
                   ignore=['__pycache__', 'local_runs', 'approvals', '*.pyc', 'tests'])
    .add_local_file(HERE.parent / 'qwen.py', '/workspace/qwen.py')
    .add_local_file(HERE.parent / 'detectors.py', '/workspace/detectors.py')
    .add_local_file(HERE.parent / 'prc.py', '/workspace/prc.py'))
data = modal.Volume.from_name('prc-data', create_if_missing=False)
hf = modal.Volume.from_name('prc-hf-cache', create_if_missing=False)


def root(smoke=False):
    path = Path('/data') / relative_root()
    return path / 'sanity' if smoke else path


def record(stage, approval, result, seconds, rate):
    from .storage import write_json
    from .config import digest_json
    value = dict(stage=stage, approval=approval, result=result, wall_seconds=seconds,
                 metered_runtime_estimate_usd=seconds*rate,
                 provider_cost_usd=None, provider_cost_status='awaiting billing reconciliation',
                 billing_caveat='Runtime estimate excludes startup/build/teardown/storage',
                 gpu_hours=seconds/3600 if stage.startswith('gpu') else 0)
    write_json(root() / 'run_ledger' / f'{stage}_{digest_json(approval)[:16]}.json', value)
    data.commit()
    return value


@app.function(image=image, cpu=4, memory=16384, volumes={'/data': data}, timeout=600, retries=0,
              include_source=False, single_use_containers=True, max_containers=1)
def sanity_cpu(provenance, approval):
    from urllib.request import urlretrieve
    from .prepare import prepare, source_check
    from .config import DESIGN
    start = time.monotonic()
    archive = Path('/tmp/Deepseek_t_3_temp_1.8.zip')
    urlretrieve('https://raw.githubusercontent.com/1234wangtr/PRC_estimator/' +
                DESIGN['source']['artifact_commit'] + '/llm/data/Deepseek_t_3_temp_1.8.zip', archive)
    prepare(root(True), provenance, smoke=True)
    checked = source_check(archive, root(True) / 'source_check.json')
    return record('sanity_cpu', approval, checked, time.monotonic()-start, 4*.0000131+16*.00000222)


@app.function(image=image, cpu=4, memory=16384, volumes={'/data': data}, timeout=900, retries=0,
              include_source=False, single_use_containers=True, max_containers=1)
def prepare_cpu(provenance, approval):
    from .prepare import prepare
    start = time.monotonic()
    result = prepare(root(), provenance)
    return record('prepare', approval, dict(keys=len(result['keys']), samples=len(result['inventory'])),
                  time.monotonic()-start, 4*.0000131+16*.00000222)


@app.function(image=image, gpu='H100', cpu=4, memory=65536,
              volumes={'/data': data, '/cache': hf}, timeout=900, retries=0,
              include_source=False, single_use_containers=True, max_containers=1)
def sanity_gpu(provenance, approval):
    from .lm import run_temperature, load_model
    from .storage import write_json
    start = time.monotonic()
    data.reload()
    if json.loads((root(True) / 'source_check.json').read_text())['status'] != 'passed':
        raise ValueError('Source convention validation did not pass')
    bundle = load_model('/cache')
    results = [run_temperature(root(True), t, '/cache', provenance, smoke=True,
                               commit=data.commit, loaded=bundle) for t in (1.0, 1.8)]
    write_json(root(True) / 'status.json', dict(status='passed', results=results))
    return record('gpu_sanity', approval, results, time.monotonic()-start, .001097+4*.0000131+64*.00000222)


@app.function(image=image, gpu='H100', cpu=4, memory=65536,
              volumes={'/data': data, '/cache': hf}, timeout=3600, retries=0,
              include_source=False, single_use_containers=True, max_containers=5)
def production_gpu(temperature, provenance, approval):
    from .lm import run_temperature
    data.reload()
    if approval['stage'] == 'experiment':
        from .validation import evidence
        if evidence() != provenance['quote']['validation']:
            raise ValueError('Completed source/numerical evidence changed')
    elif approval['sanity_status'] == 'passed':
        if json.loads((root(True) / 'status.json').read_text())['status'] != 'passed':
            raise ValueError('Missing paid sanity pass')
    elif approval['sanity_status'] == 'skipped_cost_over_5':
        if approval.get('skipped_sanity_quote_usd', 0) <= 5:
            raise ValueError('Cost skip requires actual quote over $5')
    else:
        raise ValueError('Unapproved sanity status')
    start = time.monotonic()
    result = run_temperature(root(), temperature, '/cache', provenance, commit=data.commit)
    return record(f'gpu_production_t{temperature:.1f}', approval, result, time.monotonic()-start,
                  .001097+4*.0000131+64*.00000222)


@app.function(image=image, cpu=8, memory=16384, volumes={'/data': data}, timeout=3600, retries=0,
              include_source=False, single_use_containers=True, max_containers=1)
def score_cpu(provenance, approval, oracle=False):
    from .analysis import run
    data.reload()
    start = time.monotonic()
    result = run(root(), provenance, oracle=oracle)
    return record('score', approval, result, time.monotonic()-start, 8*.0000131+16*.00000222)


@app.function(image=image, gpu='H100', cpu=4, memory=65536,
              volumes={'/data': data, '/cache': hf}, timeout=420, retries=0,
              include_source=False, single_use_containers=True, max_containers=1)
def numerical_gpu(provenance, approval):
    from .numerical_diagnostic import run
    data.reload()
    start = time.monotonic()
    result = run('/cache', root() / 'numerical_diagnostic.json', provenance)
    return record('gpu_numerical_diagnostic', approval, result, time.monotonic()-start,
                  .001097+4*.0000131+64*.00000222)


def dispatch(stage, quote, approval, oracle):
    from .config import TEMPERATURES
    provenance = dict(git_commit=quote['git_commit'], fingerprint=quote['fingerprint'],
                      run_id=approval['run_id'], quote=quote)
    with app.run():
        if stage == 'experiment':
            prepared = prepare_cpu.remote(provenance, approval)
        if stage == 'numerical-diagnostic':
            return numerical_gpu.remote(provenance, approval)
        if stage == 'sanity':
            a = sanity_cpu.remote(provenance, approval)
            b = sanity_gpu.remote(provenance, approval)
            return [a, b]
        if stage == 'prepare':
            return prepare_cpu.remote(provenance, approval)
        if stage == 'score':
            return score_cpu.remote(provenance, approval, oracle)
        calls = []
        try:
            for t in TEMPERATURES:
                calls.append(production_gpu.spawn(t, provenance, approval))
            results = [call.get() for call in calls]
        except BaseException:
            # An error does not authorize siblings to continue unchecked or retry.
            for call in calls:
                call.cancel(terminate_containers=True)
            raise
        if stage == 'experiment':
            scored = score_cpu.remote(provenance, approval, False)
            return dict(preparation=prepared, production=results, scoring=scored)
        return results
