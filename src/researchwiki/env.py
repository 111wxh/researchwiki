"""极小的 .env 加载器（不引依赖）。

为什么自己写：项目对依赖克制，而这里需要的语法只有 KEY=VALUE 一层。
约定：**真实环境变量优先**于文件——容器/CI 注入的值不会被 .env 覆盖，
这样同一份 .env 在本地开发方便、在部署环境不添乱。
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = ".env"


def parse_env_text(text: str) -> dict[str, str]:
    """解析 KEY=VALUE 行：忽略空行与 # 注释，去掉成对引号，允许 export 前缀。"""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(
    path: str | Path = DEFAULT_ENV_FILE, *, override: bool = False
) -> list[str]:
    """把 .env 读进 os.environ，返回实际写入的变量名。

    文件不存在时静默返回空列表——没有 .env 是正常状态（mock 模式无需 key）。
    """
    env_path = Path(path)
    if not env_path.is_file():
        return []
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - 权限/编码异常的兜底
        return []
    applied: list[str] = []
    for key, value in parse_env_text(text).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
