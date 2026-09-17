"""vLLM general plugin: write descriptors only after successful graph capture."""
import functools,json,os,pathlib,time,uuid

def install():
 from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
 if getattr(CudaGraphManager.capture,'_glm_depth_receipts',False):return
 original=CudaGraphManager.capture
 @functools.wraps(original)
 def capture(self,*args,**kwargs):
  result=original(self,*args,**kwargs)
  rows=[]
  for desc in self.graphs:
   rows.append({k:(getattr(desc,k).name if k=='cg_mode' else getattr(desc,k)) for k in ('cg_mode','num_tokens','num_reqs','uniform_token_count','max_query_len','num_active_loras')})
  record={'schema':'glm53-completed-graph-capture-v1','pid':os.getpid(),'manager_id':id(self),'manager_class':type(self).__name__,'completed_at_ns':time.time_ns(),'capture_completed':True,'profile_sample':self._max_full_descs_to_capture is not None or self._capture_mem_samples is not None,'progress_description':kwargs.get('progress_bar_desc',args[1] if len(args)>1 else 'Capturing CUDA graphs'),'descriptors':rows}
  root=pathlib.Path(os.environ['GLM53_CAPTURE_RECEIPTS']);root.mkdir(parents=True,exist_ok=True)
  path=root/f'capture-{os.getpid()}-{uuid.uuid4().hex}.json';path.write_text(json.dumps(record,indent=2)+'\n')
  return result
 capture._glm_depth_receipts=True;CudaGraphManager.capture=capture
