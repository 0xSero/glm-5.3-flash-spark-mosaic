"""Actual installed warmup orchestration, CPU providers and mocked CUDA only."""
import json, unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from vllm.model_executor.warmup import b12x_warmup as warm
from vllm.utils.b12x import B12xWarmupUnit

class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.events=[]
        def model(name):
            layer=torch.nn.Linear(1,1,device='cpu')
            def unit(layer, counts, dtype):
                self.events.append(('collect',name,counts,str(dtype)))
                return B12xWarmupUnit(name,name,lambda:self.events.append(('compile',name)))
            layer.b12x_warmup_provider=NS(get_b12x_warmup_unit=unit)
            return layer
        self.target=model('target')
        self.draft=model('MoE')
        self.worker=NS(model_config=NS(dtype=torch.bfloat16),
            vllm_config=NS(compilation_config=NS(compile_sizes=[]),num_speculative_tokens=1),
            scheduler_config=NS(max_num_batched_tokens=2048,max_num_scheduled_tokens=None),
            model_runner=NS(cudagraph_manager=NS(planned_token_counts=lambda:[2])),
            get_model=lambda:self.target,get_draft_model=lambda:self.draft)
    def run_warmup(self, enabled):
        with patch.dict('os.environ',{'GLM53_MTP_EXPERT_NVFP4':str(enabled)}), \
             patch.object(warm.current_platform,'is_cuda',return_value=True), \
             patch.object(warm.current_platform,'is_device_capability_family',return_value=True), \
             patch.object(torch.accelerator,'synchronize',side_effect=lambda:self.events.append(('sync',))):
            warm.b12x_warmup(self.worker,[2])
    def test_native_draft_precompiled_before_completion(self):
        self.run_warmup(1)
        self.assertEqual([x for x in self.events if x[0]=='compile'],[('compile','target'),('compile','MoE')])
        self.assertEqual(self.events[-1],('sync',))
        self.assertEqual(self.events[1][2],(1,2,2047,2048))
    def test_off_preserves_target_only_and_does_not_touch_draft(self):
        self.worker.get_draft_model=lambda:(_ for _ in ()).throw(AssertionError('draft touched'))
        self.run_warmup(0)
        self.assertEqual([x for x in self.events if x[0]=='compile'],[('compile','target')])
    def test_absent_draft_fails_before_compile(self):
        self.draft=None
        with self.assertRaisesRegex(RuntimeError,'loaded native draft'): self.run_warmup(1)
        self.assertFalse(any(x[0]=='compile' for x in self.events))
    def test_missing_provider_fails_before_compile(self):
        self.draft=torch.nn.Linear(1,1)
        with self.assertRaisesRegex(RuntimeError,'no draft MoE provider'): self.run_warmup(1)
        self.assertFalse(any(x[0]=='compile' for x in self.events))
    def test_draft_failure_propagates_before_completion(self):
        self.draft.b12x_warmup_provider.get_b12x_warmup_unit=lambda *a:B12xWarmupUnit('MoE','MoE',lambda:(_ for _ in ()).throw(RuntimeError('compile failed')))
        with self.assertRaisesRegex(RuntimeError,'compile failed'): self.run_warmup(1)
        self.assertNotIn(('sync',),self.events)

if __name__=='__main__':
    assert not torch.cuda.is_initialized()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Lifecycle)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized()
    print(json.dumps({'actual_installed_orchestration':True,'cuda_initialized':False,'tests':result.testsRun,'success':result.wasSuccessful(),'gpu_execution':False}))
    raise SystemExit(not result.wasSuccessful())
