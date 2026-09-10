# Headless PPO training image for the TonyPi walking task.
FROM python:3.12-slim

# mujoco's compiled bindings dynamically link against a GL library even when
# training with no on-screen rendering, so these need to be present at import.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# no on-screen renderer available in the container; osmesa is the software fallback
ENV MUJOCO_GL=osmesa

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

# copy dependency manifests first so `uv sync` is cached across source-only changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY sim/ sim/
COPY configs/ configs/
COPY README.md ./

ENTRYPOINT ["uv", "run", "sim/scripts/train_ppo.py"]
CMD ["--config", "configs/ppo_flat.yaml"]
