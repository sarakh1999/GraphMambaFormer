# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# GraphMambaFormer container images — everything preinstalled, so a user needs
# only Docker.
#
#   docker/build.sh                 -> graphmambaformer:latest (target: full)
#   TARGET=fig6 docker/build.sh     -> fig6-runner:1.6.1       (benchmark only)
#   TARGET=gpu docker/build.sh      -> graphmambaformer:gpu    (CUDA by default)
#
# Stages:
#   htsbuild  samtools / bcftools 1.19 compiled against libcurl
#   fig6      + vg, BWA, DeepVariant: the Fig 6a CPU benchmark stack
#   full      + PyTorch (CPU) and the graphmambaformer package  [default]
#   gpu       full, with torch swapped for a GPU build (CUDA or ROCm)
#
# Everything is linux/amd64: DeepVariant publishes no arm64 build, so on Apple
# Silicon this runs under Rosetta (correct, but slower). Apple's own GPU is not
# reachable from a container at all -- run natively on the host for MPS.
#
# The GPU stage is vendor-agnostic via TORCH_CHANNEL:
#   cu124 / cu121  NVIDIA; the pip wheels bundle the CUDA runtime, so only a
#                  host driver and `--gpus all` are needed
#   rocm6.0        AMD; needs --device /dev/kfd --device /dev/dri
# The code itself detects the vendor at runtime (graphmambaformer/accel), so one
# image works across GPU generations without a rebuild.
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
        bwa bzip2 ca-certificates curl gawk git jq less nano parallel procps \
        python3-venv tree vim \
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
# matplotlib lives here too (not only in /opt/plotenv) because the training run
# plots its own losses, validation metrics and model behaviour.
RUN python3 -m venv /opt/gmf \
    && /opt/gmf/bin/pip install --no-cache-dir --upgrade "pip<24.1" \
    && /opt/gmf/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && /opt/gmf/bin/pip install --no-cache-dir \
        "einops>=0.7" "numpy<2" "pysam>=0.22" "matplotlib==3.7.5"

# Baked so the image is self-contained; a bind-mounted /work takes precedence
# via PYTHONPATH, so live edits on the host still win.
COPY graphmambaformer/ /opt/graphmambaformer/graphmambaformer/
COPY scripts/ /opt/graphmambaformer/scripts/
COPY README.md requirements.txt /opt/graphmambaformer/

RUN ln -sf /opt/gmf/bin/python /usr/local/bin/gmf-python \
    && printf '%s\n' \
         '#!/bin/sh' \
         'if [ "$#" -lt 1 ]; then' \
         '  echo "usage: gmf-script <script.py> [args...]" >&2' \
         '  exit 2' \
         'fi' \
         'script=$1; shift' \
         'exec gmf-python "/opt/graphmambaformer/scripts/$script" "$@"' \
       > /usr/local/bin/gmf-script \
    && chmod +x /usr/local/bin/gmf-script

# Scripts stay reachable without a bind-mounted clone (gmf-script smoke_test.py).
# /work still wins on PYTHONPATH when the repo is mounted via docker/run.sh.
ENV PATH=/usr/local/bin:/opt/fig6/scripts/fig6:/opt/graphmambaformer/scripts:${PATH} \
    PYTHONPATH=/work:/opt/graphmambaformer \
    GMF_HOME=/opt/graphmambaformer \
    MPLBACKEND=Agg

WORKDIR /work
CMD ["bash"]

# --- stage 4: GPU-enabled (NVIDIA CUDA or AMD ROCm) -------------------------
FROM full AS gpu

# cu124 covers Volta through Blackwell. Use cu121 for older drivers, or
# rocm6.0 for AMD. Pinning torch>=2.4 keeps the accel stack's SDP and
# bf16 paths available.
ARG TORCH_CHANNEL=cu124
ARG GPU_TORCH_VERSION=2.4.1
#: Triton powers the fused LN+Linear+GELU kernel; it supports CUDA and ROCm.
ARG INSTALL_TRITON=1
#: CuPy provides the NVRTC raw kernels, which are NVIDIA-only by construction.
ARG INSTALL_CUPY=0

LABEL org.opencontainers.image.title="graphmambaformer-gpu" \
      org.opencontainers.image.description="GraphMambaFormer + GPU PyTorch (CUDA or ROCm) + the Fig 6a genomics stack"

# Replace the CPU wheel in place; the venv layout and PATH are unchanged, so
# nothing downstream has to know which channel was used.
RUN /opt/gmf/bin/pip install --no-cache-dir --force-reinstall \
        --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
        "torch==${GPU_TORCH_VERSION}" \
    && if [ "$INSTALL_TRITON" = "1" ]; then \
         /opt/gmf/bin/pip install --no-cache-dir "triton" || \
         echo "triton unavailable for this channel; the torch fallback is used"; \
       fi \
    && if [ "$INSTALL_CUPY" = "1" ]; then \
         /opt/gmf/bin/pip install --no-cache-dir "cupy-cuda12x" || \
         echo "cupy unavailable; raw-kernel tier stays off"; \
       fi

# Surfaced by gmf-doctor so a container that silently fell back to CPU is
# obvious rather than merely slow.
ENV GMF_GPU_CHANNEL=${TORCH_CHANNEL}

WORKDIR /work
CMD ["bash"]
