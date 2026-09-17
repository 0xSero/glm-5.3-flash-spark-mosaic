"""TEST ONLY: observe physical W4A16 launches and compare packed route reduction.

Install LaunchRecorder BEFORE the first expert forward/warmup. After constructing
native-weight NVFP4 experts, populate kernel._plans through kernel._plan for the
intended shapes, then call replace_unwarmed_plans(..., confirm_before_first_forward=True).
This replaces only this experiment's uncaptured plan cache. Never use on a serving
model or after any warmup/capture. Fast math stays True; no generic deterministic
flag is set. Run --self-test --source-root /path/to/pinned/b12x for CPU contracts.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE_PINS = {'b12x/moe/_shared/kernels/w4a16/kernel.py': 'c2c18fc5964a832256179a6d0dba127213592de69c1fd907d3855fe97c3bd821', 'b12x/moe/fused_moe/_impl.py': '4c1f497eaa561090c713e8aa6e4e8fc684658fe2f474d66d43f3e6a73af1eed3', 'b12x/moe/fused_moe/_vllm_compat.py': '232c6623e7a4f0915d605031c44871c81c3911e5a56ae8ddcc49c080034136f8', 'b12x/moe/fused_moe/_policy.py': '537270b834a731eebac4af95dc06b2aac1992d60c2f684eeb2675e7093291d50', 'b12x/policy/context.py': '71ff7618c6a6acdc126561b7093ad7617d65aa5350527c677340109926ffc300'}
SOURCE_COMMIT = '3b862805d2b7fd52e6fe507fd28038bd48797cf5'
MODULE = 'b12x.moe._shared.kernels.w4a16.kernel'
FUNCTION = 'compile_w4a16_fused_moe'
FIELDS = ('size_m', 'hidden_size', 'intermediate_size', 'num_experts', 'top_k',
          'activation', 'element_dtype', 'fast_math', 'weight_layout', 'scale_format',
          'direct_topk_routes', 'tc_decode_fused_sum', 'use_expert_map',
          'swiglu_limit', 'swiglu_alpha', 'swiglu_beta', '_require_cached')
RESULT_FIELDS = ('size_m', 'direct_topk_routes', 'tc_decode_fused_sum',
                 'use_expert_map', 'fast_math')


def verify_sources(root):
    root = Path(root)
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise ValueError('Installed B12x source differs from the audited pin')
    return actual


def scalar(value):
    # Never recurse into compiled kernels, tensor storage, launch arguments or caches.
    if value is None or type(value) in (bool, int, float, str):
        return value
    return str(value)


def _wrap_compile(original, calls):
    signature = inspect.signature(original)
    missing = set(FIELDS) - set(signature.parameters)
    if missing:
        raise ValueError('Compile signature changed: ' + repr(sorted(missing)))
    @functools.wraps(original)
    def observed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        row = {'call_index': len(calls),
               'requested': {k: scalar(bound.arguments[k]) for k in FIELDS}}
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            row.update(state='RAISED', error_type=type(error).__name__)
            calls.append(row)
            raise
        row.update(state='RETURNED', result_type=type(result).__name__,
                   returned={k: scalar(getattr(result, k, None)) for k in RESULT_FIELDS})
        calls.append(row)
        return result
    return observed


class LaunchRecorder(AbstractContextManager):
    """Records actual compile/cache-return flags without modifying launch arguments."""
    def __init__(self):
        self.calls = []
        self.active = False
        self.module = None
        self.source_sha256 = None
        self.plan_replacement_done = False

    def __enter__(self):
        if self.active:
            raise ValueError('Recorder already active')
        package = importlib.import_module('b12x')
        self.source_sha256 = verify_sources(Path(package.__file__).resolve().parent.parent)
        self.module = importlib.import_module(MODULE)
        self.original = getattr(self.module, FUNCTION)
        if getattr(self.original, '_glm_test_launch_recorder', False):
            raise ValueError('Another launch recorder is already installed')
        self.wrapper = _wrap_compile(self.original, self.calls)
        self.wrapper._glm_test_launch_recorder = True
        setattr(self.module, FUNCTION, self.wrapper)
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if getattr(self.module, FUNCTION) is not self.wrapper:
            raise RuntimeError('Compile hook changed concurrently; refusing to overwrite it')
        setattr(self.module, FUNCTION, self.original)
        self.active = False
        return False

    def receipt(self):
        return {'schema': 'glm53-test-w4a16-physical-launches-v1',
                'source_commit': SOURCE_COMMIT, 'source_sha256': self.source_sha256,
                'calls': self.calls.copy(), 'gpu_correctness_claim': False}


def _replace_plans(kernel, recorder, factory, confirm_before_first_forward):
    # Caller attestation is necessary: Python cannot prove a recorder was installed
    # before an earlier forward in a process it did not observe. Also refuse any
    # observed compile/cache lookup, even if it was only a warmup.
    if confirm_before_first_forward is not True or not recorder.active:
        raise ValueError('Active recorder and explicit pre-forward attestation required')
    if recorder.calls or recorder.plan_replacement_done:
        raise ValueError('A launch/warmup or prior replacement was already observed')
    if kernel._quant_mode != 'w4a16' or kernel._prepared().plan.source_format != 'modelopt_nvfp4':
        raise ValueError('Only this native NVFP4 W4A16 component experiment is supported')
    old = kernel._plans
    if not old:
        raise ValueError('Populate intended unwarmed plans via kernel._plan first')
    replacement = {}
    proof = []
    for key, plan in old.items():
        caps = plan.caps
        if (caps.quant_mode != 'w4a16' or caps.w4a16_fast_math is not True
                or caps.deterministic_output is not None or not caps.frozen):
            raise ValueError('Original fast-math/default-reduction/frozen contract changed')
        resolution = plan.launch_plan.policy_resolution
        if resolution is None or resolution.config.backend != 'w4a16':
            raise ValueError('Missing actual W4A16 policy resolution')
        config = replace(resolution.config, w4a16_route_mode='packed')
        context = caps.policy_context.with_override('moe.decode', config)
        new_caps = replace(caps, policy_context=context)
        rebuilt = factory(new_caps)
        if (rebuilt.launch_plan.policy_resolution.config.w4a16_route_mode != 'packed'
                or rebuilt.caps.w4a16_fast_math is not True
                or rebuilt.caps.deterministic_output is not None
                or rebuilt.caps.weight_plan is not caps.weight_plan):
            raise ValueError('Rebuilt plan changed precision or failed packed-route selection')
        replacement[key] = rebuilt
        proof.append({'key': repr(key), 'max_tokens': caps.max_tokens,
                      'core_token_counts': list(caps.core_token_counts or ()),
                      'old_route_mode': resolution.config.w4a16_route_mode,
                      'new_route_mode': 'packed', 'fast_math': True,
                      'old_scratch_bytes': plan.layout.total_nbytes,
                      'new_scratch_bytes': rebuilt.layout.total_nbytes})
    # Commit atomically only after every replacement succeeds. Keep old plans in
    # the caller's returned handle for evidence; never reuse them in captured work.
    if kernel._plans is not old or recorder.calls:
        raise ValueError('Concurrent plan/launch activity detected')
    kernel._plans = replacement
    recorder.plan_replacement_done = True
    return {'old_plans': old, 'receipt': {'state': 'TEST_ONLY_UNWARMED_PACKED_PLANS',
            'caller_attested_before_first_forward': True, 'plans': proof,
            'must_allocate_scratch_through_normal_workspace_shapes': True,
            'must_verify_physical_tc_decode_fused_sum_false': True,
            'determinism_proven': False}}


def replace_unwarmed_plans(kernel, recorder, *, confirm_before_first_forward=False):
    """Change route planning only; caller must subsequently warm and recapture anew."""
    if not isinstance(recorder, LaunchRecorder):
        raise TypeError('A real LaunchRecorder is required')
    if getattr(recorder.module, FUNCTION, None) is not getattr(recorder, 'wrapper', None):
        raise ValueError('Launch observer is not installed')
    from b12x.moe import fused_moe
    return _replace_plans(kernel, recorder, fused_moe.plan, confirm_before_first_forward)


def _source_contract(root):
    pins = verify_sources(root)
    tree = ast.parse((Path(root) / 'b12x/moe/_shared/kernels/w4a16/kernel.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    assert set(FIELDS) <= {a.arg for a in [*fn.args.args, *fn.args.kwonlyargs]}
    tree = ast.parse((Path(root) / 'b12x/moe/fused_moe/_impl.py').read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    caps_fields = {n.target.id for n in classes['TPMoEScratchCaps'].body if isinstance(n, ast.AnnAssign)}
    assert {'policy_context', 'w4a16_fast_math', 'deterministic_output', 'weight_plan', 'frozen'} <= caps_fields
    plan_fields = {n.target.id for n in classes['TPMoEScratchPlan'].body if isinstance(n, ast.AnnAssign)}
    assert {'caps', 'launch_plan', 'layout'} <= plan_fields
    return pins


def self_test(source_root):
    _source_contract(source_root)
    class Tests(unittest.TestCase):
        def setup_objects(self):
            @dataclass(frozen=True)
            class Config:
                backend: str = 'w4a16'
                w4a16_route_mode: str = 'direct'
            @dataclass(frozen=True)
            class Context:
                config: object
                def with_override(self, key, config):
                    assert key == 'moe.decode'
                    return Context(config)
            @dataclass(frozen=True)
            class Caps:
                policy_context: object
                weight_plan: object
                quant_mode: str = 'w4a16'
                w4a16_fast_math: bool = True
                deterministic_output: object = None
                frozen: bool = True
                max_tokens: int = 1
                core_token_counts: tuple = (1,)
            def factory(caps):
                return SimpleNamespace(caps=caps, layout=SimpleNamespace(total_nbytes=123),
                    launch_plan=SimpleNamespace(policy_resolution=SimpleNamespace(config=caps.policy_context.config)))
            cap = Caps(Context(Config()), object())
            old = factory(cap)
            kernel = SimpleNamespace(_quant_mode='w4a16', _plans={1: old},
                    _prepared=lambda: SimpleNamespace(plan=SimpleNamespace(source_format='modelopt_nvfp4')))
            recorder = SimpleNamespace(active=True, calls=[], plan_replacement_done=False)
            return kernel, recorder, factory
        def test_scope_and_unchanged_original(self):
            k,r,f=self.setup_objects();old=k._plans
            result=_replace_plans(k,r,f,True)
            self.assertIs(result['old_plans'],old)
            self.assertEqual(old[1].launch_plan.policy_resolution.config.w4a16_route_mode,'direct')
            self.assertEqual(k._plans[1].launch_plan.policy_resolution.config.w4a16_route_mode,'packed')
            self.assertIs(k._plans[1].caps.weight_plan,old[1].caps.weight_plan)
            self.assertTrue(k._plans[1].caps.w4a16_fast_math)
        def test_refuse_observed_launch_or_missing_attestation(self):
            for launched,attest in ((True,True),(False,False)):
                k,r,f=self.setup_objects();old=k._plans
                if launched:r.calls.append({})
                with self.assertRaises(ValueError):_replace_plans(k,r,f,attest)
                self.assertIs(k._plans,old)
        def test_factory_failure_does_not_mutate_cache(self):
            k,r,f=self.setup_objects();old=k._plans
            def fail(caps):raise RuntimeError('compile failure')
            with self.assertRaises(RuntimeError):_replace_plans(k,r,fail,True)
            self.assertIs(k._plans,old)
        def test_precision_change_rejected(self):
            k,r,f=self.setup_objects();k._plans[1].caps=replace(k._plans[1].caps,w4a16_fast_math=False)
            with self.assertRaises(ValueError):_replace_plans(k,r,f,True)
        def test_compile_signature_and_return_identity(self):
            # Exact pinned argument names, with no B12x/CUDA import or execution.
            namespace={};exec('def compile('+','.join(n+'=None' for n in FIELDS)+'):\n return result',namespace)
            result=SimpleNamespace(tc_decode_fused_sum=True,direct_topk_routes=True)
            namespace['result']=result;calls=[];wrapped=_wrap_compile(namespace['compile'],calls)
            self.assertIs(wrapped(size_m=1,fast_math=True,tc_decode_fused_sum=True),result)
            self.assertTrue(calls[0]['returned']['tc_decode_fused_sum'])
            self.assertEqual(calls[0]['requested']['size_m'],1)
            with self.assertRaises(ValueError):_wrap_compile(lambda bad:None,[])
        def test_compile_error_preserved(self):
            namespace={};exec('def compile('+','.join(n+'=None' for n in FIELDS)+'):\n raise RuntimeError("original")',namespace)
            calls=[]
            with self.assertRaisesRegex(RuntimeError,'original'):_wrap_compile(namespace['compile'],calls)()
            self.assertEqual(calls[0]['state'],'RAISED')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
    print(json.dumps({'state':'CPU_HELPER_CONTRACT_PASS_NOT_GPU_TESTED','tests':result.testsRun,
                      'source_sha256':SOURCE_PINS,'actual_source_signature_checked':True}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true',required=True)
    p.add_argument('--source-root',type=Path,required=True)
    a=p.parse_args();self_test(a.source_root)
