"""Install the exact GPU-qualified projection overlay into a new image only."""
import argparse
import hashlib
import json
from pathlib import Path

BASE_WRAPPER = '2d62a8bd6f0d2e1df7556ce086f8c294d6c20faa4e4af123300aa595e264584d'
EXPECTED = {
    'exl3.projection.py': '30321f703b39b9391d3b2fb861f84c175be48dafc05f8bd8b32be5be336224f2',
    'glm_exl3_projection_bits.py': '2794ddbb9d0b92bdd8926e308826d261c31111eafca0c1be5b262d6ab810deab',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def install(directory, source):
    files = {name: (source / name).read_bytes() for name in EXPECTED}
    for name, data in files.items():
        if digest(data) != EXPECTED[name]:
            raise ValueError('Qualified overlay hash changed: ' + name)
        compile(data, name, 'exec')
    wrapper = directory / 'exl3.py'
    before = digest(wrapper.read_bytes())
    if before not in (BASE_WRAPPER, EXPECTED['exl3.projection.py']):
        raise ValueError('Unexpected base wrapper; refusing overwrite')
    helper = directory / 'glm_exl3_projection_bits.py'
    if helper.exists() and digest(helper.read_bytes()) != EXPECTED[helper.name]:
        raise ValueError('Conflicting projection helper')
    for target, data in ((helper, files[helper.name]), (wrapper, files['exl3.projection.py'])):
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)
    return {'state': 'GPU_QUALIFIED_SOURCE_INSTALLED_FULL_SERVING_UNTESTED',
        'before_wrapper_sha256': before, 'installed_sha256': EXPECTED,
        'scope': 'No weight conversion; target projection metadata only. Native MTP remains on its separate FP8 expert path.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path(__file__).parent)
    a = p.parse_args()
    print(json.dumps(install(a.directory, a.source)))
