import hashlib, importlib.metadata, importlib.util, inspect, json, pathlib, torch
base=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm')
result={'cuda_initialized_before':torch.cuda.is_initialized(),'gpu_exposed':False,'packages':{}}
for package in ['torch','nvidia-cutlass-dsl','quack-kernels','cuda-python','cuda-bindings']:
    try: result['packages'][package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: result['packages'][package]=None
for relative in ['model_executor/layers/fused_moe/router/gate_linear.py','model_executor/kernels/linear/cute_dsl/ll_bf16.py','model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py','model_executor/kernels/linear/cute_dsl/_ll_bf16_splitk.py']:
    path=base/relative
    result.setdefault('source',{})[relative]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
try:
    spec=importlib.util.spec_from_file_location('audit_ll_bf16',base/'model_executor/kernels/linear/cute_dsl/ll_bf16.py')
    import sys
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    result['ll_bf16_available']=module.is_available()
except Exception as e: result['ll_bf16_import_error']=repr(e)
try:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    result['actual_imported_gate_class']={'module':GateLinear.__module__,'qualname':GateLinear.__qualname__,'sourcefile':inspect.getfile(GateLinear)}
except Exception as e: result['gate_import_error']=repr(e)
result['cuda_initialized_after']=torch.cuda.is_initialized()
assert not result['cuda_initialized_before'] and not result['cuda_initialized_after']
print(json.dumps(result,indent=2))
