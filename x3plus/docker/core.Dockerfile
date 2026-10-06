# Demo-safe core: arm control + ArUco + table geometry. No CUDA, no torch.
FROM python:3.10-slim-bookworm

# uv comes from its own published image -- no curl|sh in the build.
COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONPATH=/app:/app/vendor

WORKDIR /app

# Dependencies first, so editing our own code does not re-resolve the world.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Yahboom's SDK is not on PyPI. tools/vendor_sdk.sh extracts it from the robot's
# own installed egg, so the vendored copy always matches this hardware.
COPY vendor/ /app/vendor/

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd -u 1000 -m robo && mkdir -p /app/data && chown -R robo:robo /app
USER robo

CMD ["python", "-c", "print('roboarm core ready')"]
