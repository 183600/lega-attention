"""容器镜像（Dockerfile）与依赖锁的一致性：可复现环境不能只是约定。

背景
====
镜像安装“最新 torch”时，它构建的是**另一套研究环境**：流水线自己的
fail-closed 环境闸门（``lega.repro.environment_gate``）会在正式实验里以
``environment_lock_mismatch`` 中止，镜像也就不再是可复现环境；而一旦有人
放宽锁文件，结果就会静默来自未声明的依赖栈。因此镜像必须与
``requirements-lock.txt`` 一致，并且**在构建期**就执行同一把环境闸门。

本文件把这条约定变成可执行断言：

  * 当前 Dockerfile 必须通过 ``scripts/check_requirements_lock.py`` 的容器
    一致性检查（镜像里的 torch 版本来自 lock、镜像构建跑环境闸门、基础
    镜像可按 digest 固定、运行时阶段不装未声明的包）；
  * 每一类退化都必须被检查器拒绝（去掉环境闸门、改成不锁版本、改成与锁
    不一致的版本、装未声明的包、写死基础镜像 tag）——检查器没有牙齿时，
    “通过”只是自我安慰；
  * ``docker-compose.yml`` 的构建参数必须与 Dockerfile 的接口一致，不能
    再出现已被移除的 ``CUDA_VERSION`` 之类的空转参数；
  * 环境闸门 CLI 的退出码语义（一致=0 / 不一致=1 / 无法核对=2）必须成立，
    因为镜像构建与流水线都依赖它做 fail-closed 判定。
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.check_requirements_lock import (  # noqa: E402
    ENV_GATE_COMMAND,
    check_container_lock,
    parse_lock,
)

LOCK_PATH = ROOT / "requirements-lock.txt"
CI_LOCK_PATH = ROOT / "requirements-ci.txt"
DOCKERFILE_PATH = ROOT / "Dockerfile"
COMPOSE_PATH = ROOT / "docker-compose.yml"


def _locked():
    return parse_lock(LOCK_PATH), parse_lock(CI_LOCK_PATH)


def _dockerfile_text():
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


def _mutate(old: str, new: str) -> str:
    """在当前 Dockerfile 上做一处替换；锚点缺失时立即失败（避免空断言）。"""
    text = _dockerfile_text()
    assert old in text, f"Dockerfile 里找不到待替换的锚点：{old!r}"
    return text.replace(old, new, 1)


def test_dockerfile_builds_the_locked_environment():
    """当前 Dockerfile 必须与依赖锁一致（否则镜像不是可复现环境）。"""
    locked, ci_locked = _locked()
    problems = check_container_lock(_dockerfile_text(), locked, ci_locked)
    assert not problems, "Dockerfile 与依赖锁不一致：\n" + "\n".join(problems)
    print("  [ok] Dockerfile 构建的是锁定环境")


def test_dockerfile_runs_the_environment_gate_at_build_time():
    """镜像构建期必须执行与流水线同一把环境闸门。"""
    text = _dockerfile_text()
    assert ENV_GATE_COMMAND in text, (
        f"Dockerfile 没有执行环境闸门 {ENV_GATE_COMMAND}："
        "镜像可能带着与 lock 不一致的依赖栈发布"
    )
    # 闸门要在“装完运行时依赖之后”执行：出现在 COPY 源码之前没有意义。
    install_at = text.find("requirements-lock.txt")
    gate_at = text.find(ENV_GATE_COMMAND)
    assert gate_at > install_at, "环境闸门必须排在依赖安装之后"
    print("  [ok] 镜像构建期执行环境闸门")


def test_dockerfile_derives_the_torch_version_from_the_lock():
    """镜像里的 torch 版本必须来自 lock 文件，不能写死字面量。"""
    text = _dockerfile_text()
    assert "requirements-lock.txt" in text
    # 任何 torch 的版本约束都必须写成 ${...} 变量（由 lock 派生），
    # 而不是直接写一个可能与 lock 脱节的字面量版本。
    # 排除 `s/^torch==//p` 这类“从 lock 里取版本”的 sed/grep 表达式。
    for match in re.finditer(r"(?<![\w^/])torch==([^\s\"']+)", text):
        version = match.group(1)
        assert "${" in version, (
            f"Dockerfile 把 torch 版本写死为 {version}："
            "版本漂移后镜像会与 lock 不一致"
        )
    print("  [ok] torch 版本由 requirements-lock.txt 派生")


def test_missing_environment_gate_is_rejected():
    """去掉构建期环境闸门必须被检查器拒绝。"""
    locked, ci_locked = _locked()
    broken = _mutate(f"RUN {ENV_GATE_COMMAND}", "RUN true")
    problems = check_container_lock(broken, locked, ci_locked)
    assert problems, "删掉环境闸门后检查器仍然放行：闸门形同虚设"
    assert any("环境闸门" in problem for problem in problems), problems
    print("  [ok] 缺少环境闸门会被拒绝")


def test_unpinned_or_upgraded_runtime_deps_are_rejected():
    """`pip install --upgrade torch` 这类漂移安装必须被拒绝。"""
    locked, ci_locked = _locked()
    broken = _mutate(
        'python -m pip install --no-cache-dir "torch==${TORCH_PIN}"'
        ' --index-url "${CPU_TORCH_INDEX_URL}"',
        "python -m pip install --no-cache-dir --upgrade torch",
    )
    problems = check_container_lock(broken, locked, ci_locked)
    assert problems, "不锁版本的 torch 安装没有被拒绝"
    assert any("torch" in problem for problem in problems), problems
    print("  [ok] 不锁版本/--upgrade 的依赖安装会被拒绝")


def test_version_differing_from_the_lock_is_rejected():
    """与 lock 不一致的字面量版本必须被拒绝。"""
    locked, ci_locked = _locked()
    broken = _mutate('torch==${TORCH_PIN}', "torch==0.0.1")
    problems = check_container_lock(broken, locked, ci_locked)
    assert problems, "与 lock 不一致的 torch 版本没有被拒绝"
    assert any("torch" in problem for problem in problems), problems
    print("  [ok] 与锁不一致的版本会被拒绝")


def test_undeclared_packages_in_runtime_stage_are_rejected():
    """运行时阶段装未声明的包（torchvision 之类）必须被拒绝。"""
    locked, ci_locked = _locked()
    broken = _mutate(
        'python -m pip install --no-cache-dir "torch==${TORCH_PIN}"'
        ' --index-url "${CPU_TORCH_INDEX_URL}"',
        'python -m pip install --no-cache-dir "torch==${TORCH_PIN}" torchvision'
        ' --index-url "${CPU_TORCH_INDEX_URL}"',
    )
    problems = check_container_lock(broken, locked, ci_locked)
    assert problems, "未声明依赖没有被拒绝"
    assert any("torchvision" in problem for problem in problems), problems
    print("  [ok] 未声明依赖会被拒绝")


def test_base_image_must_be_pinnable_by_digest():
    """基础镜像写死 tag 时必须被拒绝（正式复现需要 digest 固定）。"""
    locked, ci_locked = _locked()
    broken = _mutate("FROM ${PYTHON_IMAGE} AS base", "FROM python:3.12-slim AS base")
    problems = check_container_lock(broken, locked, ci_locked)
    assert problems, "写死的基础镜像 tag 没有被拒绝"
    assert any("digest" in problem for problem in problems), problems
    print("  [ok] 不可按 digest 固定的基础镜像会被拒绝")


def test_compose_build_args_match_the_dockerfile_interface():
    """docker-compose 的构建参数必须与 Dockerfile 的接口一致。"""
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "CUDA_VERSION" not in text, (
        "docker-compose 仍在传 CUDA_VERSION：Dockerfile 已改用 "
        "TORCH_INDEX_URL 选择 CUDA wheel 索引"
    )
    assert "TORCH_INDEX_URL" in text, (
        "docker-compose 的 GPU 服务没有指定 TORCH_INDEX_URL："
        "镜像会退回 PyPI 默认 wheel，CUDA 变体随上游漂移"
    )
    # compose 里出现的构建参数必须在 Dockerfile 里有对应 ARG。
    dockerfile = _dockerfile_text()
    for arg in set(re.findall(r"^\s{4,}([A-Z_][A-Z0-9_]*):", text, re.MULTILINE)):
        if arg in ("PYTHONPATH", "LEGA_DATA_DIR", "LEGA_RESULTS_DIR"):
            continue
        assert f"ARG {arg}" in dockerfile, f"Dockerfile 没有声明 ARG {arg}"
    print("  [ok] compose 构建参数与 Dockerfile 接口一致")


def test_environment_gate_exit_codes_are_fail_closed(capsys):
    """环境闸门的退出码语义：一致=0，不一致=1，无法核对=2。"""
    from lega import repro

    locked, _ = _locked()
    runtime = repro.runtime_info("cpu")
    installed = runtime.get("installed_distributions") or {}
    missing = [name for name in sorted(locked) if name not in installed]
    matching = ROOT / "tmp_env_gate_matching.lock"
    mismatching = ROOT / "tmp_env_gate_mismatching.lock"
    broken = ROOT / "tmp_env_gate_broken.lock"
    try:
        lines = ["# python: " + str(runtime.get("python", "3.12"))]
        for name in sorted(locked):
            if name in installed:
                lines.append(f"{name}=={installed[name]}")
        matching.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # 版本与 lock 明确不同 → 必须判“不一致”（退出码 1）。
        mismatching.write_text(
            "\n".join(f"{name}==0.0.1" for name in sorted(locked)) + "\n",
            encoding="utf-8",
        )
        # lock 文件损坏（没有 name==version 条目）→ 无法核对（退出码 2）。
        broken.write_text("# python: 3.12\nnot-a-requirement\n", encoding="utf-8")

        if missing:
            # 环境里缺了被锁定的包：无法构造“应当一致”的 lock，只核对另外两档。
            print(f"  [skip] 当前环境未安装 {missing}，跳过“一致=0”分支")
        else:
            assert repro.main(["--check-env", "--lock", str(matching)]) == 0
            capsys.readouterr()
        assert repro.main(["--check-env", "--lock", str(mismatching)]) == 1
        capsys.readouterr()
        assert repro.main(["--check-env", "--lock", str(broken)]) == 2
        capsys.readouterr()
    finally:
        for path in (matching, mismatching, broken):
            if os.path.exists(path):
                os.remove(path)
    print("  [ok] 环境闸门退出码语义正确（0/1/2）")


def test_locked_runtime_packages_are_not_installed_unpinned_anywhere():
    """整份 Dockerfile 里不允许出现被锁定包的无版本安装。"""
    locked, _ = _locked()
    text = _dockerfile_text()
    for name in sorted(locked):
        # 形如 `pip install torch` / `pip install torch --upgrade`
        pattern = re.compile(
            r"pip\s+install[^\n]*?(?<![\w=])" + re.escape(name) + r"(?![\w=])"
        )
        for match in pattern.finditer(text):
            statement = text[match.start() : text.find("\n", match.start())]
            assert "==" in statement, (
                f"Dockerfile 里出现未固定版本的 {name} 安装：{statement.strip()}"
            )
    print("  [ok] Dockerfile 中没有未锁版本的运行时依赖安装")
