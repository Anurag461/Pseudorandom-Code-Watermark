"""Pure Python mocks: no native imports, model calls, or experiment scoring."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class SetupChecks(unittest.TestCase):
    def test_reviewed_scope_and_reject_expansion(self):
        verify = function(ROOT / 'online_8b_eta020_batch50.py', 'verify_plan',
                          {'RUN': 'online_8b_eta020_T14336_N50_v1'})
        plan = json.loads((OUT / 'setup.json').read_text())
        verify(plan)
        for key, value in [('N', 500), ('null_count', 50), ('automatic_retries', 1),
                           ('small_detector_records', 50), ('max_memory_fraction', 1.0)]:
            changed = copy.deepcopy(plan); changed[key] = value
            with self.assertRaises(ValueError):
                verify(changed)

    def memory_case(self, fraction):
        written, called = [], []
        tensor = types.SimpleNamespace(to=lambda device: 'fake-device-tensor')
        partition = types.SimpleNamespace()
        class Partition:
            def __getitem__(self, index): return tensor
        cuda = types.SimpleNamespace(empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 90, max_memory_reserved=lambda: 92,
            get_device_properties=lambda device: types.SimpleNamespace(total_memory=100))
        model = types.SimpleNamespace(parameters=lambda: iter([
            types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))]))
        def write(path, payload):
            path.parent.mkdir(parents=True, exist_ok=True); path.touch(); written.append(payload)
        def replay(*args): called.append(True); return 'mock-primary-trace'
        namespace = {'_redetect_inputs': lambda *args: {'tokens': tensor, 'partition': Partition()},
                     '_redetect_write': write, '_redetect_trace': lambda *args: written[-1]}
        modules = {'torch': types.SimpleNamespace(cuda=cuda),
                   'detectors': types.SimpleNamespace(tensor_sha256=lambda x: 'mock-hash'),
                   'qwen': types.SimpleNamespace(completion_only_partition_trace_batch=replay,
                                                make_kv_cache=lambda *args, **kwargs: None)}
        fn = function(ROOT / 'modal_run.py', '_recover_redetection_batch', namespace)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            kwargs = {} if fraction is None else {'max_memory_fraction': fraction}
            if fraction is None:
                with self.assertRaisesRegex(ValueError, 'completed trace saved'):
                    fn(model, {'root': 'batch', 'identity': {'cache': 'static'}}, directory, **kwargs)
            else:
                result = fn(model, {'root': 'batch', 'identity': {'cache': 'static'}}, directory, **kwargs)
                self.assertFalse(result['cached'])
            self.assertEqual(len(called), 1)
            self.assertEqual(len(written), 1)
            self.assertTrue((Path(directory) / 'batch/trace.pt').exists())
            self.assertEqual(written[0]['memory_limit_fraction'], .85 if fraction is None else fraction)

    def test_default_limit_retains_completed_trace(self): self.memory_case(None)
    def test_explicit_95_percent_limit(self): self.memory_case(.95)


if __name__ == '__main__': unittest.main()
