"""Tiny synthetic correctness checks only: no pretrained LM or dataset scoring."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from wang_prc_detector_ablation import config, hierarchy, wang
from wang_prc_detector_ablation.storage import save_arrays, load_arrays


def test_exact_prompt_strings_and_source_hashes():
    tree = ast.parse((config.HERE / 'vendor/main.py').read_text())
    chats = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'chats' for t in n.targets))
    assert config.PROMPTS == [p[0]['content'] for p in chats]
    for name, digest in config.DESIGN['source']['files'].items():
        assert config.sha256(config.HERE / 'vendor' / name) == digest


@pytest.mark.parametrize('vocab', [3, 8, 11, 17])
def test_hierarchy_direct_sums_and_fixed_uniforms(vocab):
    rng = np.random.default_rng(71)
    width = (vocab - 1).bit_length()
    p = rng.random((5, vocab))
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random((5, width))
    x = rng.integers(0, 2, (5, width))
    for code in (None, x):
        y, path = hierarchy.walk(torch.tensor(p), uniforms=torch.tensor(u),
                                 codeword=None if code is None else torch.tensor(code))
        for i in range(5):
            ref_y, ref_p = hierarchy.reference_sample(p[i], u[i], None if code is None else code[i])
            assert int(y[i]) == ref_y
            np.testing.assert_allclose(path[i], ref_p, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(path[i], hierarchy.reference_path(p[i], ref_y), atol=1e-12)
        _, observed_p = hierarchy.walk(torch.tensor(p), observed=y)
        np.testing.assert_array_equal(path, observed_p)


def test_small_tail_mass_not_erased_by_prefix_subtraction():
    p = np.array([[1 - 6e-30, 1e-30, 2e-30, 3e-30]], dtype=np.float64)
    _, got = hierarchy.walk(torch.tensor(p), observed=torch.tensor([3]))
    np.testing.assert_allclose(got[0], hierarchy.reference_path(p[0], 3), rtol=1e-12)
    assert got[0, 1] == .6


def test_endpoints_and_not_unconditional_marginals():
    p = torch.tensor([[.4, .1, .1, .4], [0., 0., 0., 1.]], dtype=torch.float64)
    _, got = hierarchy.walk(p, observed=torch.tensor([3, 3]))
    np.testing.assert_allclose(got, [[.5, .8], [1., 1.]])
    # Second coordinate is .8 conditional on the first bit, not marginal .5.
    np.testing.assert_array_equal(wang.token_bits([0, 5], 3), [[0,0,0], [1,0,1]])


def test_sampler_exact_categorical_distribution_enumerated():
    p = np.array([.12, .21, .27, .4])
    # Averaging the channel's two equiprobable latent-bit transitions gives p1.
    for y in range(4):
        ps = hierarchy.reference_path(p, y)
        bits = wang.token_bits(y, 2)
        probability = 1.
        for prob, bit in zip(ps, bits):
            q0 = 0 if prob <= .5 else 2*prob-1
            q1 = 2*prob if prob <= .5 else 1
            upper = (q0+q1)/2
            probability *= upper if bit else 1-upper
        assert probability == pytest.approx(p[y])


def test_wang_keygen_encode_against_vendored_reference(monkeypatch):
    spec = importlib.util.spec_from_file_location('official_test', config.HERE / 'vendor/llm_prc_api.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    class Recorded:
        def __init__(self):
            self.rng = np.random.default_rng(73)
            self.randoms, self.choices, self.permutations = [], [], []
        def integers(self, *args, **kwargs):
            value = self.rng.integers(*args, **kwargs)
            self.randoms.append(value.copy())
            return value
        def choice(self, *args, **kwargs):
            value = self.rng.choice(*args, **kwargs)
            self.choices.append(value.copy())
            return value
        def permutation(self, *args):
            value = self.rng.permutation(*args)
            self.permutations.append(value.copy())
            return value
    recorded = Recorded()
    key = wang.keygen(40, recorded, r=30)
    monkeypatch.setattr(official.GF, 'Random', lambda *a, **k: official.GF(recorded.randoms.pop(0)))
    monkeypatch.setattr(official.np.random, 'choice', lambda *a, **k: recorded.choices.pop(0))
    monkeypatch.setattr(official.np.random, 'permutation', lambda *a, **k: recorded.permutations.pop(0))
    enc, dec = official.KeyGen(40, r=30)
    np.testing.assert_array_equal(key['generator'], enc[0])
    np.testing.assert_array_equal(key['otp'], enc[1])
    np.testing.assert_array_equal(key['supports'], np.sort(dec[1].indices.reshape(30, 3), axis=1))
    assert not np.bitwise_xor.reduce(key['generator'][key['supports']], axis=1).any()
    encoded = wang.encode(key, np.random.default_rng(99))
    recorded.randoms.append(encoded['payload'])
    monkeypatch.setattr(official.np.random, 'binomial', lambda *a, **k: encoded['noise'])
    np.testing.assert_array_equal(encoded['codeword'], official.Encode(enc))
    expected = int(wang.hard_count(encoded['codeword'], key))
    assert official.Detect(dec, encoded['codeword']) == (expected <= wang.hard_threshold(30))


def test_first_token_zero_and_map_formula_unchanged():
    from detectors import map_soft_token
    bits = np.array([[0,1,0], [1,0,1]], dtype=np.uint8)
    p = np.array([[np.nan]*3, [.1,.7,.9]])
    soft = wang.soft_evidence(bits, p).reshape(2,3)
    np.testing.assert_array_equal(soft[0], 0)
    np.testing.assert_array_equal(soft[1], map_soft_token(bits[1], p[1]))
    key = dict(supports=np.array([[0, 1, 2]]), otp=np.zeros(6, dtype=np.uint8))
    h_before = wang.hard_count(bits.ravel(), key)
    bits[0,0] ^= 1
    assert wang.hard_count(bits.ravel(), key) != h_before  # Hard retains first token.
    score = wang.soft_score(soft.ravel(), key)
    assert score['V'] == 0 and score['no_evidence'] and not score['standard']
    assert score['Z'] == -np.inf


def test_soft_parity_and_exact_standard_normalization():
    key = dict(supports=np.array([[0,1,2],[1,2,3]]), otp=np.array([0,1,0,1], dtype=np.uint8))
    bits = np.array([1,1,0,0], dtype=np.uint8)
    s = wang.soft_score(1-2*bits.astype(float), key)
    assert s['S'] == 2 - 2*wang.hard_count(bits,key)
    assert s['V'] == 2
    small = wang.soft_score(np.array([1,2,3,4])*1e-8, key)
    assert small['V'] > 0
    assert small['Z'] == pytest.approx(small['S']/np.sqrt(small['V']))
    assert not np.isclose(small['Z'], small['S']/np.sqrt(small['V']+1e-12))


def test_whole_ties_gaps_and_inclusive_decisions():
    c = wang.calibrate([1,3,3,8,9], [5,4,4,2,1], alpha=.4)
    assert c['hard']['cutoff'] == 2 and c['hard']['accepted'] == 1
    assert c['posterior']['cutoff'] == np.nextafter(4., np.inf)
    assert c['posterior']['ties'] == 2 and c['posterior']['accepted'] == 1
    soft = dict(Z=np.array([c['posterior']['cutoff'], 4., -np.inf]),
                no_evidence=np.array([False,False,True]), standard=np.array([True,True,False]))
    d = wang.decisions(np.array([2,3,0]), soft, 16, c)
    np.testing.assert_array_equal(d['hard_matched'], [True,False,True])
    np.testing.assert_array_equal(d['posterior_matched'], [True,False,False])
    assert d['wang_published'][2]  # r=16: cutoff exactly zero, inclusive equality.
    zero = wang.calibrate([0,0], [1.,1.])
    assert zero['hard']['cutoff'] == -1 and zero['posterior']['accepted'] == 0
    absent = wang.calibrate([2,2], [-np.inf,-np.inf])
    assert absent['posterior']['reject_all'] and absent['posterior']['cutoff'] is None


def test_disjoint_rng_domains_and_paired_codewords():
    samples = list(config.inventory())
    assert len(samples) == 1600 and len({s['seed'] for s in samples}) == 1600
    assert sum(s['split']=='calibration' for s in samples) == 400
    assert sum(s['split']=='evaluation' for s in samples) == 400
    keys = {config.seed('key', domain, k) for domain, n in
            [('wm',10),('calibration',256),('evaluation',256)] for k in range(n)}
    assert len(keys) == 522
    s = samples[5]
    isolated = np.random.default_rng(s['seed']).random((3,18))
    scheduled = {v['id']: np.random.default_rng(v['seed']).random((3,18)) for v in samples[:6][::-1]}
    np.testing.assert_array_equal(isolated, scheduled[s['id']])


def test_cache_rejects_mismatch_and_corruption(tmp_path):
    p = tmp_path/'trace.npz'
    save_arrays(p, {'tokens':np.array([1,2])}, {'sample':'one'})
    load_arrays(p, {'sample':'one'})
    with pytest.raises(ValueError):
        load_arrays(p, {'sample':'two'})
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):
        load_arrays(p)


def test_completion_only_replay_input_boundary(monkeypatch):
    from wang_prc_detector_ablation.lm import replay
    monkeypatch.setitem(config.DESIGN, 'bits', 3)
    class CannedLogits:
        def parameters(self):
            yield torch.zeros(1)
        def __call__(self, ids, cache=None):
            cache.extend(ids[0].tolist())
            return torch.tensor([[[.1,.2,.3,.4,.5,.6]]]).repeat(len(ids),1,1)
    ids=np.array([[5,1,3,2]])
    capture=[]
    result=replay(CannedLogits(), ids, 1.4, capture=capture, cache_factory=lambda n:[])
    np.testing.assert_array_equal(np.concatenate(capture,axis=1), ids[:,:-1])
    assert np.isnan(result['replay_p1'][:,0]).all()
    assert np.isfinite(result['replay_p1'][:,1:]).all()


def test_bootstrap_cluster_weights_and_roc_ties():
    from wang_prc_detector_ablation.analysis import bootstrap_weights, roc
    w=bootstrap_weights(2,3,20)
    np.testing.assert_allclose(w.sum(axis=1),1)
    for row in w.reshape(20,2,3):
        assert np.linalg.matrix_rank(row) <= 1  # Crossed group and prompt counts.
    _,_,auc=roc([1,1], [[1,1],[1,1]])
    assert auc == .5
    assert roc([2],[[1,1]])[2] == 1.


def test_approval_gate_without_importing_modal():
    from wang_prc_detector_ablation.launch import validate_approval
    q=dict(stage='score',fingerprint='abc',git_commit='def',profile='profile',estimated_usd=[.5,2])
    approval=dict(**{k:q[k] for k in ('stage','fingerprint','git_commit','profile')},
        approved=True,user_approval_text='go ahead with this score run',run_id='one',
        billing_review='reviewed',max_estimated_usd=2,quote_sha256=config.digest_json(q))
    validate_approval(approval,'score',q)
    with pytest.raises(ValueError):
        validate_approval({**approval,'approved':False},'score',q)
    with pytest.raises(ValueError):
        validate_approval({**approval,'fingerprint':'changed'},'score',q)
