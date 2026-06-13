# syntax=docker/dockerfile:1.6
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Build from this repository with BuildKit:
# DOCKER_BUILDKIT=1 docker build --build-context hrgym=../human-robot-gym -t allostatic-handover:dev .
COPY --from=hrgym . /workspace/human-robot-gym
COPY . /workspace/allostatic-handover

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e /workspace/human-robot-gym[training] \
    && python -m pip install -e /workspace/allostatic-handover[training]

WORKDIR /workspace/allostatic-handover
EXPOSE 7860
CMD ["python", "-m", "allostatic_handover.dashboard.app", "--host", "0.0.0.0", "--log-dir", "outputs"]
