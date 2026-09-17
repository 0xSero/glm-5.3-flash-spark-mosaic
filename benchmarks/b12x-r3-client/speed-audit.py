"""Offline reproduction of sealed C1 timing and MTP proofs; no API calls."""
import hashlib,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'acceptance'))
from timing import matched_window
from mtp_proof import prove
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
new=ROOT/'benchmarks/b12x-r3-c1-1024'
legacy=ROOT/'benchmarks/k2full-fp8-legacy-c1-1024-r4'
inspect_path=ROOT/'benchmarks/b12x-r3-client/speed-final-inspect.json'
i=json.loads(inspect_path.read_text())[0]
assert i['Id']=='58dc544a56a624f0dbae7393e6befad9ffc4827e97bc3be17a33a7d2bdbd9968'
assert i['State']['Status']=='exited' and i['State']['ExitCode']==0 and not i['State']['OOMKilled']
records=[]
for label,path,runtime_path in [('Legacy FlashInfer',legacy,ROOT/'native-mtp/runtime-receipt-k2full-fp8draft-r4.json'),('B12x/Jovian R3',new,ROOT/'b12x-runtime/acceptance-receipt-r3.json')]:
 run=json.loads((path/'run.json').read_text());runtime=json.loads(runtime_path.read_text())
 assert run['runtime_receipt_sha256']==sha(runtime_path)
 for name,expected in run['source_sha256'].items():assert sha(ROOT/'acceptance'/name)==expected
 assert run['max_num_seqs']==1 and run['output_reserve']==2048 and run['template']=={'enable_thinking':True}
 for cell_path in sorted(path.glob('cell-*.json')):
  data=json.loads(cell_path.read_text());cell=data['cell'];rows=data['requests'];assert len(rows)==1
  row=rows[0];assert row['success'] and row['done'] and row['cached_tokens'] in (None,0) and runtime['prefix_caching'] is False
  assert row['usage']['completion_tokens']==2048
  window=matched_window(rows);reported=cell['summary']['matched_decode'];assert window==reported
  assert window['status']=='MATCHED_SUSTAINED' and window['window_seconds']>=30
  proof=prove(runtime,cell);assert proof['passed'] and all(math.isfinite(v) for v in cell['speculative_counter_deltas'].values())
  content=row['content'];reasoning=row['reasoning'];numbered=re.findall(r'^\s*#\s*\d+[.):]?\s+',content,re.M)
  records.append({'runtime':label,'repeat':cell['repeat'],'actual_prompt_tokens':row['usage']['prompt_tokens'],'active_concurrency':1,'output_tokens':2048,'matched_decode_tokens':sum(x['tokens'] for x in window['streams']),'TOTAL_decode_tok_s':window['total_aggregate_decode_tok_s'],'mean_per_request_decode_tok_s':window['mean_per_request_decode_tok_s'],'window_seconds':window['window_seconds'],'native_request_prefill_tok_s':cell['native_timing']['prompt_tokens_per_sum_request_prefill_second'],'TTFT_seconds':cell['summary']['ttft_p50_seconds'],'end_to_end_total_tok_s':cell['summary']['total_output_tokens_per_end_to_end_wall_second'],'mtp':proof,'cached_tokens_export':row['cached_tokens'],'prefix_cache_disabled_in_runtime':True,'exact_stream_token_accounting_passed':True,'matched_window_recomputed_exactly':True,'content_chars':len(content),'reasoning_chars':len(reasoning),'finish_reason':row['finish_reason'],'requested_comment_lines':240,'final_numbered_comment_lines':len(numbered),'content_task_completed':len(numbered)>=240 and row['finish_reason']=='stop','cell_sha256':sha(cell_path),'runtime_receipt_sha256':sha(runtime_path)})
means={}
for label in ('Legacy FlashInfer','B12x/Jovian R3'):
 rs=[x for x in records if x['runtime']==label];means[label]=sum(x['matched_decode_tokens'] for x in rs)/sum(x['window_seconds'] for x in rs)
report={'state':'MEASURED_REASONING_THROUGHPUT_NOT_QUALITY_ACCEPTED','client_exit_code':0,'client_inspect_sha256':sha(inspect_path),'target':'same full unpruned K2/288target experts; nativeMTP routed experts FP8 only','requested_context':262144,'configured_active_slots':1,'records':records,'pooled_TOTAL_decode_tok_s':means,'pooled_B12x_relative_change_percent':100*(means['B12x/Jovian R3']/means['Legacy FlashInfer']-1),'failed_content_cases':[{'runtime':x['runtime'],'repeat':x['repeat'],'reason':'2048-token cap reached with reasoning only and empty final content; requested 240-comment-line task not completed'} for x in records if not x['content_task_completed']],'infrastructure_failures':[],'caveats':['All counted output is reasoning; these are not final-answer throughput or successful task-completion rates.','Second B12x prompt is 1,023 tokens; other three are 1,024. Nonce-bearing prompts are not byteidentical.','Two repeats only, different physical GB10nodes and runtime versions; no statistically established speedup or isolated B12x-only causal claim.','First/second repeats are not a proof of identical kernel warm-state across engines.','No C2/C4/C8 or long-context speed measurements were launched.','Native MTP counters prove execution, not MTP speed gain or target quality parity.','Cached-token usage fields are absent; disabled prefix caching is established from sealed runtime configuration, not inferred as an exported zero.','K2 quality promotion remains held.']}
(new/'independent-validation.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Full K2 with FP8 MTP: legacy versus B12x','', '**Experimental reasoning-token measurements; quality promotion remains on hold.** All four requests reached the 2,048-token limit with empty final answers. They did not complete the requested 240-comment-line task. Transport, exact-token accounting, sustained-window recalculation and native MTP execution checks passed.','', '| Runtime | Repeat | Actual prompt | C | Native prefill tok/s | TTFT s | TOTAL decode tok/s | Per-request decode tok/s | Window s | End-to-end total tok/s | Accepted/drafted |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in records:
 lines.append(f"| {x['runtime']} | {x['repeat']} | {x['actual_prompt_tokens']} | 1 | {x['native_request_prefill_tok_s']:.2f} | {x['TTFT_seconds']:.3f} | **{x['TOTAL_decode_tok_s']:.3f}** | {x['mean_per_request_decode_tok_s']:.3f} | {x['window_seconds']:.2f} | {x['end_to_end_total_tok_s']:.3f} | {x['mtp']['accepted_tokens']:.0f}/{x['mtp']['drafted_tokens']:.0f} |")
lines.extend(['',f"Pooled TOTAL decode across the two matched windows: legacy **{means['Legacy FlashInfer']:.3f} tok/s**, B12x **{means['B12x/Jovian R3']:.3f} tok/s** ({report['pooled_B12x_relative_change_percent']:+.2f}%). This small experiment does not establish a decode speed improvement. Prefill improved in the first repeat and regressed in the second; warm-state equivalence is not established.",'','Both use the same full 288-expert K2 target, native MTP depth 1 with FP8 routed draft experts, 262,144 requested context, 2,048 prefill chunk, and one active slot. Legacy has 460,208 aggregate KV tokens; B12x has 751,007. Source benchmark scripts have identical hashes. The second B12x prompt has 1,023 tokens; others have 1,024. The runtime versions, kernels, packed cache layout and physical Spark differ.','','TOTAL decode counts exact token IDs in the common `(start,end]` window, excluding the first streamed bundle and preserving speculative bundles. It includes reasoning. End-to-end total includes prefill/waiting. Native prefill is request-level server timing, not an aggregate GPU-only rate. No multi-concurrency or long-context speed result is implied. Cached-token usage fields are absent; prefix caching is disabled in the sealed runtime configuration, not inferred as an exported zero.','','## Failed cases','','| Runtime | Repeats | Final content | Finish | Result |','|---|---|---|---|---|','| Legacy FlashInfer | 0,1 | Empty | length | Requested 240-line output not completed |','| B12x/Jovian R3 | 0,1 | Empty | length | Requested 240-line output not completed |','','No transport/OOM/stream-accounting failure occurred in these four cells. Semantic failure/length truncation remains distinct from measured reasoning throughput. Separate serving-quality evidence remains parent-owned.'])
(new/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
seal={p.name:sha(p) for p in sorted(new.iterdir()) if p.is_file() and p.name!='integrity-seal.json'}
(new/'integrity-seal.json').write_text(json.dumps({'state':'SEALED_NOT_RELEASE_ACCEPTED','files_sha256':seal,'audit_script_sha256':sha(Path(__file__)),'client_inspect_sha256':sha(inspect_path),'mtp_proof_script_sha256':sha(ROOT/'acceptance/mtp_proof.py')},indent=2)+'\n')
print(json.dumps({'pooled':means,'relative_change_percent':report['pooled_B12x_relative_change_percent'],'content_failures':len(report['failed_content_cases']),'independent_checks':'PASS'},indent=2))
