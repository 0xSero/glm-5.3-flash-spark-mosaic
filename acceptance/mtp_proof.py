#!/usr/bin/env python3
"""Offline native-MTP execution proof from pinned runtime and isolated cell receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def prove(runtime, cell):
    spec=runtime.get('speculative_config') or {}
    delta=cell.get('speculative_counter_deltas') or {}
    drafted=sum(v for k,v in delta.items() if k.endswith('num_draft_tokens_total'))
    accepted=sum(v for k,v in delta.items() if k.endswith('num_accepted_tokens_total'))
    rejected=any(v<0 for v in delta.values())
    active=spec.get('method')=='mtp' and int(spec.get('num_speculative_tokens',0))>0
    return {'native_mtp_configured':active,'drafted_tokens':drafted,'accepted_tokens':accepted,
            'accepted_fraction':accepted/drafted if drafted>0 else None,
            'passed':bool(active and cell.get('status')=='COMPLETED' and cell.get('native_timing',{}).get('status')=='matched_isolated_cell' and not rejected and drafted>0 and 0<accepted<=drafted),
            'scope':'Configuration plus isolated advancing speculative token counters; no MTP-on speed gain or quality parity inferred.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-receipt',type=Path,required=True);p.add_argument('--cell',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('use a fresh output file')
    result=prove(json.loads(a.runtime_receipt.read_text()),json.loads(a.cell.read_text())['cell'])
    result['input_sha256']={label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in [('runtime',a.runtime_receipt),('cell',a.cell)]}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(not result['passed'])


if __name__=='__main__':main()
