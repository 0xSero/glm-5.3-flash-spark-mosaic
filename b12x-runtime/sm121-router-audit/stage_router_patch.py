"""Emit an isolated, source-pinned router candidate; never edit the input."""
import argparse,hashlib,pathlib
SOURCE_SHA='8ad2f44f02fbb709783c17dcceb628b16ae5adb381155f573b0cd0663243f596'
def patch(data):
 if hashlib.sha256(data).hexdigest()!=SOURCE_SHA:raise ValueError('Unrecognized GateLinear source; refusing patch')
 text=data.decode()
 edits=[
 ('        is_blackwell_rtx = current_platform.is_device_capability((12, 0))\n', '        is_blackwell_rtx = current_platform.is_device_capability((12, 0))\n        # Qualified only for GLM-5.3-Flash native router geometry on GB10.\n        self._sm121_glm_router = (\n            current_platform.is_device_capability((12, 1))\n            and input_size == 4096 and output_size == 288 and not bias\n        )\n'),
 ('            and (is_hopper or is_blackwell or is_blackwell_rtx)\n','            and (is_hopper or is_blackwell or is_blackwell_rtx or self._sm121_glm_router)\n'),
 ('        self._sm120_graph_pool_lifetime_guard = is_blackwell_rtx\n','        self._sm120_graph_pool_lifetime_guard = is_blackwell_rtx or self._sm121_glm_router\n'),
 ('        if self.allow_ll_bf16_gemm and x.shape[0] <= 16 and x.dtype == torch.bfloat16:\n','        if (\n            self.allow_ll_bf16_gemm and x.shape[0] <= 16\n            and x.dtype == torch.bfloat16\n            # Clustered split-K M5-16 regressed on GB10. M3 remains unqualified.\n            and (not self._sm121_glm_router or x.shape[0] in (1, 2, 4))\n        ):\n')]
 for old,new in edits:
  if text.count(old)!=1:raise ValueError('Expected source anchor absent or repeated')
  text=text.replace(old,new)
 compile(text,'gate_linear.candidate.py','exec')
 return text.encode()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
 if args.source.resolve()==args.output.resolve():raise ValueError('Input must stay immutable')
 result=patch(args.source.read_bytes());args.output.write_bytes(result);print(hashlib.sha256(result).hexdigest())
