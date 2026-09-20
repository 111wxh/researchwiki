# 后端镜像：uv + Python 3.12（单阶段：venv 的解释器路径与镜像一致，避免跨镜像拷贝的兼容问题）
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    RESEARCHWIKI_HOST=0.0.0.0

WORKDIR /app

# 先只拷贝依赖清单：依赖没变时这层直接命中缓存，改代码不会重装依赖
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

# 默认 mock 模式：无需任何 key 即可演示完整链路。
# 接真实模型时挂载自己的 config.toml，并通过环境变量传 key（见 docker-compose.full.yml）。
CMD ["uv", "run", "--no-sync", "researchwiki", "serve"]
