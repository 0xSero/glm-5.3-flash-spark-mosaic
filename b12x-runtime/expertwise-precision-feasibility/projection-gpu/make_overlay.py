"""Stage an isolated Python-only projection-precision overlay; never edit input."""
from pathlib import Path
import hashlib
import difflib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
source=(ROOT/'exl3-port/exl3.py').read_bytes()
assert hashlib.sha256(source).hexdigest()=='2d62a8bd6f0d2e1df7556ce086f8c294d6c20faa4e4af123300aa595e264584d'
s=source.decode()
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace('            return Exl3MoEMethod(layer.moe_config, self)', '''            import copy
            from .glm_exl3_projection_bits import resolve_layer_projection_bits

            selected = copy.copy(self)
            overrides = self.raw_config.get("layer_projection_bits", {})
            if not isinstance(overrides, dict):
                raise ValueError("layer_projection_bits must be an object")
            if overrides:
                if self.raw_config.get("layer_bits"):
                    raise ValueError("Do not combine layer_bits and layer_projection_bits")
                match = re.fullmatch(
                    r"(?:language_model\\.)?model\\.layers\\.([0-9]+)\\.mlp\\.experts(?:\\.routed_experts)?",
                    prefix,
                )
                if match is None:
                    raise ValueError("Projection precision requires a target MoE prefix")
                selected.projection_bits = resolve_layer_projection_bits(
                    self.bits, overrides, int(match[1])
                )
            else:
                selected.projection_bits = (self.bits,) * 3
            return Exl3MoEMethod(layer.moe_config, selected)''')
replace('        self.bits = quant_config.bits\n        self._logged = False','''        self.bits = quant_config.bits
        self.projection_bits = getattr(quant_config, "projection_bits", (self.bits,) * 3)
        self._logged = False''')
replace('        k_words = self.bits * 16','''        k_words = self.projection_bits[0] * 16
        down_k_words = self.projection_bits[2] * 16''')
replace('num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16','num_experts, out_tiles, in_tiles, down_k_words, dtype=torch.int16')
replace('        layer._exl3_bits = self.bits','''        layer._exl3_bits = self.bits
        layer._exl3_projection_bits = self.projection_bits''')
replace('    k = int(getattr(layer, "_exl3_k", 4))\n    args = (','''    k = int(getattr(layer, "_exl3_k", 4))
    projection_bits = getattr(layer, "_exl3_projection_bits", (k, k, k))
    args = (''')
replace('        MOE_ACT_SILU,\n        k,\n        k,\n        k,','''        MOE_ACT_SILU,
        projection_bits[0],
        projection_bits[1],
        projection_bits[2],''')
replace('                inners.append({"gate": gate, "up": up, "down": down})','''                if (gate.K, up.K, down.K) != self.projection_bits:
                    raise ValueError("Loaded projection precision differs from metadata")
                inners.append({"gate": gate, "up": up, "down": down})''')
(HERE/'exl3.projection.py').write_text(s)
(HERE/'glm_exl3_projection_bits.py').write_bytes((HERE.parent/'projection_contract.py').read_bytes())
(HERE/'projection-precision.patch').write_text(''.join(difflib.unified_diff(source.decode().splitlines(True),s.splitlines(True),fromfile='a/exl3.py',tofile='b/exl3.py')))
print(hashlib.sha256(s.encode()).hexdigest())
