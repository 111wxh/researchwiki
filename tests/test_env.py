"""`.env` 加载器测试：解析规则、真实环境优先、文件缺失静默。"""

from __future__ import annotations

import os
from pathlib import Path

from researchwiki.env import load_env_file, parse_env_text


def test_parse_basic_and_comments():
    text = (
        "# 注释行\n"
        "\n"
        "A=1\n"
        "B = spaced value \n"
        "export C=with-export\n"
        'D="quoted"\n'
        "E='single'\n"
        "NOVALUE\n"
        "=nokey\n"
    )
    assert parse_env_text(text) == {
        "A": "1",
        "B": "spaced value",
        "C": "with-export",
        "D": "quoted",
        "E": "single",
    }


def test_load_missing_file_is_silent(tmp_path: Path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_load_writes_env(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("RW_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("RW_TEST_KEY", raising=False)
    assert load_env_file(env_file) == ["RW_TEST_KEY"]
    assert os.environ["RW_TEST_KEY"] == "from-file"


def test_real_env_wins(tmp_path: Path, monkeypatch):
    """真实环境变量优先：容器/CI 注入的值不能被 .env 覆盖。"""
    env_file = tmp_path / ".env"
    env_file.write_text("RW_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("RW_TEST_KEY", "from-real-env")
    assert load_env_file(env_file) == []
    assert os.environ["RW_TEST_KEY"] == "from-real-env"


def test_override_flag_forces_file(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("RW_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("RW_TEST_KEY", "from-real-env")
    assert load_env_file(env_file, override=True) == ["RW_TEST_KEY"]
    assert os.environ["RW_TEST_KEY"] == "from-file"
