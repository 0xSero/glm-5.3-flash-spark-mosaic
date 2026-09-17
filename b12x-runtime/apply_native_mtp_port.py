#!/usr/bin/env python3
"""Port the established native BF16 MTP fixes to pinned Jovian source."""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"MTP source anchor mismatch: {old[:90]!r}")
    return text.replace(old, new, 1)


def patch(root, check_only=False):
    mtp = root / "vllm/models/glm5next/nvidia/mtp.py"
    proposer = root / "vllm/v1/spec_decode/llm_base_proposer.py"
    eagle = root / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py"
    fp8 = root / "vllm/model_executor/layers/quantization/glm_mtp_expert_fp8.py"
    if fp8.exists():
        raise ValueError("MTP FP8 policy already exists; refuse to overwrite")
    fp8_text = Path(__file__).with_name("mtp_expert_fp8.py").read_text()
    text = mtp.read_text()
    # Latest Jovian filters checkpoint keys before load_weights. Include the
    # companion's real shared weights, so strict completeness checks stay real.
    text = replace_once(text,
        '        if self.has_own_lm_head:\n            prefixes += (\n',
        '        prefixes += ("model.embed_tokens.", "model.language_model.embed_tokens.")\n'
        '        if True:  # Native companion head is loaded before sharing.\n            prefixes += (\n')
    text = replace_once(text,
        '            if name in (\n                "lm_head.weight",',
        '            if name == "model.embed_tokens.weight":\n'
        '                param = params_dict[name]\n'
        '                loader = getattr(param, "weight_loader", default_weight_loader)\n'
        '                loader(param, loaded_weight)\n'
        '                loaded_params.add(name)\n'
        '                continue\n'
        '            if name in (\n                "lm_head.weight",')
    text = replace_once(text,
        '                if self.has_own_lm_head:\n                    for layer_idx in self.model.layers:\n',
        '                if True:  # Load real BF16 shared heads before strict validation.\n'
        '                    for layer_idx in self.model.layers:\n')
    # Remove temporary indentation guards rather than leave artificial branches.
    lines = text.splitlines(keepends=True)
    for start, stop in [('        if True:  # Native companion', '        return prefixes'),
                        ('                if True:  # Load real BF16', '                continue')]:
        index = next(i for i, line in enumerate(lines) if line.startswith(start))
        end = next(i for i in range(index + 1, len(lines)) if lines[i].startswith(stop))
        lines[index:end] = [line[4:] for line in lines[index + 1:end]]
    text = ''.join(lines)
    text = replace_once(text,
        'from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix\n',
        'from vllm.model_executor.models.interfaces import (\n'
        '    MultiModalEmbeddings, SupportsMultiModalEmbeddings, _require_is_multimodal,\n'
        ')\n'
        'from vllm.model_executor.models.utils import (\n'
        '    WeightsMapper, maybe_prefix, _merge_multimodal_embeddings,\n'
        ')\n')
    text = replace_once(text,
        'class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):\n',
        'class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts, SupportsMultiModalEmbeddings):\n')
    text = replace_once(text,
        '    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n'
        '        return self.model.embed_input_ids(input_ids)\n',
        '''    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The target owns the vision encoder. Reuse its image/video features,
        # masking placeholders before lookup so OOV media IDs are also safe.
        text_ids = input_ids
        if is_multimodal is not None:
            text_ids = input_ids.masked_fill(
                is_multimodal.to(device=input_ids.device, non_blocking=True), 0
            )
        inputs_embeds = self.model.embed_input_ids(text_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )
''')
    base = proposer.read_text()
    anchor = '        return base\n\n    def _get_model(self)'
    binding = '''        # Keep native GLM MTP independent of target EXL3 and REAP counts.
        if "Glm5NextMTPModel" in (
            self.draft_model_config.hf_config.architectures or []
        ):
            from vllm.model_executor.models.utils import get_draft_quant_config

            base = replace(
                base,
                model_config=self.draft_model_config,
                quant_config=get_draft_quant_config(base),
            )
            from vllm.model_executor.layers.quantization.glm_mtp_expert_fp8 import (
                maybe_enable_mtp_expert_fp8,
            )
            base = maybe_enable_mtp_expert_fp8(base)
'''
    base = replace_once(base, anchor, binding + anchor)
    base = replace_once(base,
        '                "GlmOcrForConditionalGeneration",\n',
        '                "GlmOcrForConditionalGeneration",\n'
        '                "Glm5NextForConditionalGeneration",\n')
    v2 = replace_once(eagle.read_text(),
        '    draft_model_config = speculative_config.draft_model_config\n',
        '''    draft_model_config = speculative_config.draft_model_config
    # Native GLM MTP must not inherit target EXL3 or REAP geometry in V2.
    if "Glm5NextMTPModel" in (draft_model_config.hf_config.architectures or []):
        from vllm.model_executor.models.utils import get_draft_quant_config

        vllm_config = replace(
            vllm_config,
            model_config=draft_model_config,
            quant_config=get_draft_quant_config(vllm_config),
        )
        from vllm.model_executor.layers.quantization.glm_mtp_expert_fp8 import (
            maybe_enable_mtp_expert_fp8,
        )
        vllm_config = maybe_enable_mtp_expert_fp8(vllm_config)
''')
    results = {}
    for path, content in [(mtp, text), (proposer, base), (eagle, v2), (fp8, fp8_text)]:
        compile(content, str(path), 'exec')
        results[str(path.relative_to(root))] = hashlib.sha256(content.encode()).hexdigest()
    if not check_only:
        mtp.write_text(text)
        proposer.write_text(base)
        eagle.write_text(v2)
        fp8.write_text(fp8_text)
    return {"state": "SOURCE_CHECKED" if check_only else "PATCHED_NOT_RUNTIME_VALIDATED", "sha256": results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(patch(args.root, args.check_only), indent=2))
