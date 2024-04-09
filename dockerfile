FROM mambaorg/micromamba:focal-cuda-11.6.2
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install --yes --file /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
WORKDIR /app
COPY . /app
USER root
RUN apt-get update && \
    apt-get install -y git
RUN pip install comet-ml==3.21.0
RUN pip install flask==2.0.2
RUN pip install werkzeug==2.2.2
RUN pip install git+https://github.com/openai/CLIP.git
CMD ["python", "app.py"]