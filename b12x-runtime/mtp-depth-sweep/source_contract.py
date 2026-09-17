"""Execute pure graph-selector helpers extracted from the audited installed source."""
import ast,hashlib,json,types
from common import ROOT,locked_dependencies

def helper(source,name,namespace=None):
 tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 scope={'BatchExecutionDescriptor':object,**(namespace or {})}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned installed source>','exec'),scope)
 return scope[name]

def validate():
 locked_dependencies();source=json.loads((ROOT/'b12x-runtime/sm121-router-audit/mtp-depth-installed-sources.json').read_text())
 sparse=helper(source['v1/worker/gpu/spec_decode/autoregressive/speculator.py'],'_sparse_full_capture_request_sizes')
 compatible=helper(source['v1/worker/gpu/cudagraph_utils.py'],'_is_compatible')
 results=[]
 for depth in (1,2,3,5):
  for slots in (1,2,4,8):
   caps=sorted(sparse(slots));q=depth+1
   for n in range(1,slots+1):
    descriptors=[types.SimpleNamespace(uniform_token_count=q,max_query_len=None,num_reqs=c,num_tokens=c*q,num_active_loras=0) for c in caps]
    if not any(compatible(d,n,n*q,q,0,q) for d in descriptors):raise ValueError('Installed sparse capture selector lacks coverage')
   results.append({'depth':depth,'slots':slots,'target_token_sizes':[q*n for n in range(1,slots+1)],'draft_prefill_request_capacities':caps})
 return {'state':'PINNED_INSTALLED_PURE_SELECTOR_CPU_PASS','configuration_count':len(results),'configurations':results,'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()},'gpu_test':False}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
