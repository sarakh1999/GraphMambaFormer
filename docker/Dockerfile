# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# GraphMambaFormer container images — everything preinstalled, so a user needs
# only Docker.
#
#   docker/build.sh                 -> graphmambaformer:latest (target: full)
#   TARGET=fig6 docker/build.sh     -> fig6-runner:1.6.1       (benchmark only)
#
# Stages:
#   htsbuild  samtools / bcftools 1.19 compiled against libcurl
#   fig6      + vg, BWA, DeepVariant: the Fig 6a CPU benchmark stack
#   full      + PyTorch (CPU) and the graphmambaformer package  [default]
#
# Everything is linux/amd64: DeepVariant publishes no arm64 build, so on Apple
# Silicon this runs under Rosetta (correct, but slower).
# ---------------------------------------------------------------------------
ARG DV_IMAGE=google/deepvariant:1.6.1

# --- stage 1: htslib / samtools / bcftools ---------------------------------
# The base image ships samtools+bcftools 1.10 from conda; the pipeline pins
# 1.19, and fetch_reads.sh needs libcurl to slice a remote GIAB BAM over HTTPS.
FROM ${DV_IMAGE} AS htsbuild

ARG HTSLIB_VERSION=1.19
ARG SAMTOOLS_VERSION=1.19
ARG BCFTOOLS_VERSION=1.19

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake build-essential bzip2 ca-certificates curl \
        libbz2-dev libcurl4-openssl-dev liblzma-dev libncurses5-dev \
        libssl-dev make perl zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

RUN curl -fsSL -O "https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2" \
    && tar -xjf "htslib-${HTSLIB_VERSION}.tar.bz2" \
    && cd "htslib-${HTSLIB_VERSION}" \
    && ./configure --prefix=/usr/local --enable-libcurl \
    && make -j"$(nproc)" && make install

RUN curl -fsSL -O "https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2" \
    && tar -xjf "samtools-${SAMTOOLS_VERSION}.tar.bz2" \
    && cd "samtools-${SAMTOOLS_VERSION}" \
    && ./configure --prefix=/usr/local --with-htslib=system \
    && make -j"$(nproc)" && make install

RUN curl -fsSL -O "https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2" \
    && tar -xjf "bcftools-${BCFTOOLS_VERSION}.tar.bz2" \
    && cd "bcftools-${BCFTOOLS_VERSION}" \
    && ./configure --prefix=/usr/local --with-htslib=system \
    && make -j"$(nproc)" && make install

# --- stage 2: the Fig 6a benchmark runner -----------------------------------
FROM ${DV_IMAGE} AS fig6

ARG VG_VERSION=v1.75.1
ARG DOCKER_CLI_VERSION=27.3.1

LABEL org.opencontainers.image.source="https://github.com/sarakh1999/GraphMambaFormer" \
      org.opencontainers.image.title="fig6-runner" \
      org.opencontainers.image.description="vg Giraffe + BWA-MEM + DeepVariant CPU baselines for HPRC Fig 6a"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        bwa bzip2 ca-certificates curl gawk git jq less parallel procps \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=htsbuild /usr/local/bin/ /usr/local/bin/
COPY --from=htsbuild /usr/local/lib/ /usr/local/lib/
COPY --from=htsbuild /usr/local/include/ /usr/local/include/
RUN ldconfig

# Static release binary, so the Ubuntu 20.04 glibc of the base image is fine.
RUN curl -fsSL -o /usr/local/bin/vg \
        "https://github.com/vgteam/vg/releases/download/${VG_VERSION}/vg" \
    && chmod +x /usr/local/bin/vg

# Client only — used solely to reach an external hap.py container.
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
        | tar -xz -C /tmp docker/docker \
    && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker

# plot_pr.py runs in its own venv so pandas/matplotlib cannot disturb the
# numpy/TensorFlow versions DeepVariant is pinned to.
RUN python3 -m venv /opt/plotenv \
    && /opt/plotenv/bin/pip install --no-cache-dir --upgrade "pip<24.1" \
    && /opt/plotenv/bin/pip install --no-cache-dir \
        "pandas==2.0.3" "matplotlib==3.7.5"

COPY scripts/fig6/ /opt/fig6/scripts/fig6/
COPY docker/gmf-doctor /usr/local/bin/gmf-doctor
RUN chmod +x /opt/fig6/scripts/fig6/*.sh /usr/local/bin/gmf-doctor

# /usr/local/bin must precede the base image's /opt/conda/bin (samtools 1.10).
ENV PATH=/usr/local/bin:/opt/fig6/scripts/fig6:${PATH} \
    FIG6_ROOT=/work \
    FIG6_NATIVE=1 \
    PLOT_PY=/opt/plotenv/bin/python \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /work
CMD ["bash"]

# --- stage 3: + the model stack ---------------------------------------------
FROM fig6 AS full

# torch 2.4.x is the last series supporting the base image's Python 3.8.
ARG TORCH_VERSION=2.4.1

LABEL org.opencontainers.image.title="graphmambaformer" \
      org.opencontainers.image.description="GraphMambaFormer + PyTorch (CPU) + the full Fig 6a genomics stack"

# A dedicated venv keeps PyTorch away from DeepVariant's pinned TensorFlow/numpy.
RUN python3 -m venv /opt/gmf \
    && /opt/gmf/bin/pip install --no-cache-dir --upgrade "pip<24.1" \
    && /opt/gmf/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && /opt/gmf/bin/pip install --no-cache-dir \
        "einops>=0.7" "numpy<2" "pysam>=0.22"

# Baked so the image is self-contained; a bind-mounted /work takes precedence
# via PYTHONPATH, so live edits on the host still win.
COPY graphmambaformer/ /opt/graphmambaformer/graphmambaformer/
COPY scripts/ /opt/graphmambaformer/scripts/
COPY README.md requirements.txt /opt/graphmambaformer/

RUN ln -sf /opt/gmf/bin/python /usr/local/bin/gmf-python

ENV PYTHONPATH=/work:/opt/graphmambaformer \
    GMF_HOME=/opt/graphmambaformer

WORKDIR /work
CMD ["bash"]
