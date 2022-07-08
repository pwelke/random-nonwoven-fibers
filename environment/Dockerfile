# hash:sha256:9f10dd19dc191f4acfd00b41cf2e4f2243e618fa6dfd6f9e602a7802aa8b60a8
FROM registry.codeocean.com/codeocean/matlab:2021b-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python-is-python3=3.8.2-4 \
        python3=3.8.2-0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    pip==22.1.2
RUN pip3 install -U --no-cache-dir \
    matplotlib==3.5.2 \
    networkx==2.6 \
    numpy==1.22.3 \
    pandas==1.4.2 \
    pot==0.7.0 \
    requests==2.27.1 \
    scikit-learn==1.0.2 \
    scipy==1.8.1 \
    tqdm==4.63.0

COPY postInstall /
RUN /postInstall
