# Tiny overlay over a separately verified R3 image; does not alter a live server.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY install.py exl3.projection.py glm_exl3_projection_bits.py /opt/glm53-projection/
RUN python3 -c 'import importlib.util, pathlib, subprocess; root=pathlib.Path(importlib.util.find_spec("vllm").origin).parent; subprocess.run(["python3","/opt/glm53-projection/install.py","--directory",str(root/"model_executor/layers/quantization")],check=True)'
