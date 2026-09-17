# Local staged overlay; supply an independently verified source-pinned base.
# No weight conversion occurs at image build. NVFP4 remains OFF by default.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY install.py SOURCE_PINS.json glm_mtp_expert_precision.py /opt/glm53-mtp-nvfp4/
RUN python3 -c 'import importlib.util as u, pathlib as p, subprocess as s; v=p.Path(u.find_spec("vllm").origin).parent; b=p.Path(u.find_spec("b12x").origin).parent.parent; s.run(["python3","/opt/glm53-mtp-nvfp4/install.py","--vllm-root",str(v),"--b12x-root",str(b)],check=True)'
