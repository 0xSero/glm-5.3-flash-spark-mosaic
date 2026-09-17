"""Test pinned addon through native Transformers classes, without GPUs."""
import tempfile,json,hashlib
from pathlib import Path
from transformers import AutoTokenizer,AutoProcessor
import torch
assert not torch.cuda.is_initialized()
addon=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 for path in (addon/'payload').iterdir():(root/path.name).symlink_to(path)
 (root/'config.json').symlink_to(addon/'reference/source-config.json')
 tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
 processor=AutoProcessor.from_pretrained(root,local_files_only=True,trust_remote_code=False)
 print(json.dumps({'state':'CPU_TOKENIZER_PROCESSOR_LOAD_PASS','tokenizer_class':type(tokenizer).__name__,
                  'processor_class':type(processor).__name__,'tokenizer_vocab_size':tokenizer.vocab_size,
                  'probe_ids':tokenizer.encode('The quick brown fox',add_special_tokens=False),
                  'addon_manifest_sha256':hashlib.sha256((addon/'ADDON_MANIFEST.json').read_bytes()).hexdigest(),
                  'cuda_initialized':torch.cuda.is_initialized()}))
