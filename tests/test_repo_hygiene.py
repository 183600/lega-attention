"""仓库工程规范自检：受保护文件、产物卫生、配置契约与元数据一致性。

这些不变量以前只靠约定与 CI 脚本维持，任何一次“清理多余文件”的自动化
都可能把它们悄悄删掉。本文件把它们变成可执行断言：

  * **受保护文件**：``README*.md``（全部语言版本）与 ``.github/``（CI 配置）
    必须存在、非空且被 git 跟踪；``.github/`` 工作区不得有未提交改动。
    这两条直接对应项目约定：CI 配置与说明文档不是“多余内容”。
  * **产物卫生**：远程仓库不得包含实验结果文件（``results/`` 只保留
    ``.gitkeep``）、``REPORT*.md``、``autodl_budget*.json`` 等本地产物。
  * **配置契约**：``configs/*.json`` 全部通过 ``lega.config_schema`` 校验。
  * **元数据一致**：``pyproject.toml`` 与 ``lega.__version__`` 版本一致；
    顶层模块 ``run_all`` 必须声明为 ``py-modules``（否则 wheel 不含该模块，
    ``lega-run`` 控制台脚本在非 editable 安装下直接失效）；依赖声明与
    ``requirements.txt`` 一致。
  * **测试入口覆盖**：``tests/run_tests.py`` 的模块清单必须覆盖全部
    ``tests/test_*.py``，否则无 pytest 环境下的默认入口会静默漏测。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+；3.10 环境跳过需要 tomllib 的断言
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lega.utils import load_config_json  # noqa: E402

# 允许出现在仓库里的 README（前缀匹配，覆盖所有语言版本）
README_PREFIX = "README"
# 远程仓库中禁止出现的本地产物
FORBIDDEN_PATTERNS = ("REPORT", "autodl_budget")
# docs/ 目录只保留设计笔记（流水线清理阶段的既定策略）
ALLOWED_DOCS = {"design_notes.md"}


def _git(args):
    """执行 git 命令；不可用时返回 None（源码包/无 git 环境）。"""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _tracked_files():
    out = _git(["ls-files"])
    if out is None:
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def test_protected_readme_files_exist_and_are_tracked():
    """README*.md 必须存在、非空且被 git 跟踪（防误删/误移）。"""
    readmes = sorted(
        p for p in ROOT.iterdir() if p.is_file() and p.name.startswith(README_PREFIX)
    )
    assert readmes, "仓库根目录没有任何 README*.md：说明文档被删除或移动了"
    tracked = _tracked_files()
    for path in readmes:
        assert path.stat().st_size > 0, f"{path.name} 是空文件"
        if tracked is not None:
            rel = path.relative_to(ROOT).as_posix()
            assert rel in tracked, f"{rel} 未被 git 跟踪：删除后无法从历史恢复"
    # 至少要有中文主文档与英文镜像，两者都不可缺失
    names = {p.name for p in readmes}
    assert "README.md" in names, "README.md 缺失"
    print(f"  [ok] 受保护 README 存在且被跟踪: {sorted(names)}")


def test_github_ci_config_is_present_and_unmodified():
    """``.github/`` 是 CI 配置：必须存在、被跟踪，且工作区无未提交改动。"""
    github = ROOT / ".github"
    assert github.is_dir(), ".github/ 目录缺失：CI 配置被删除或移动了"
    workflows = github / "workflows"
    assert workflows.is_dir(), ".github/workflows/ 缺失"
    yml_files = sorted(workflows.glob("*.yml")) + sorted(workflows.glob("*.yaml"))
    assert yml_files, ".github/workflows/ 下没有任何工作流文件"
    tracked = _tracked_files()
    for path in yml_files:
        assert path.stat().st_size > 0, f"{path.name} 是空文件"
        if tracked is not None:
            rel = path.relative_to(ROOT).as_posix()
            assert rel in tracked, f"{rel} 未被 git 跟踪"
    status = _git(["status", "--porcelain", "--", ".github"])
    if status is not None:
        changed = [line for line in status.splitlines() if line.strip()]
        assert not changed, "`.github/` 存在未提交改动：CI 配置不得被修改：" + str(
            changed
        )
    print(f"  [ok] .github CI 配置完整且未被修改: {[p.name for p in yml_files]}")


def test_repository_contains_no_experiment_artifacts():
    """远程仓库不得包含实验结果/预算/报告类本地产物。"""
    tracked = _tracked_files()
    if tracked is None:
        return
    offenders = []
    for rel in sorted(tracked):
        name = os.path.basename(rel)
        if rel.startswith("results/"):
            if rel != "results/.gitkeep":
                offenders.append(rel)
            continue
        if name.startswith(FORBIDDEN_PATTERNS):
            offenders.append(rel)
    assert not offenders, f"仓库里混入了实验产物文件: {offenders}"
    print("  [ok] 仓库不含实验结果/报告/预算类产物")


def test_all_configs_satisfy_the_config_contract():
    """configs/*.json 全部通过配置契约校验（未知键/类型/跨字段冲突）。"""
    config_dir = ROOT / "configs"
    files = sorted(config_dir.glob("*.json"))
    assert files, "configs/ 下没有任何配置"
    for path in files:
        load_config_json(str(path), validate=True)
    print(f"  [ok] {len(files)} 个配置全部通过契约校验")


def test_packaging_metadata_is_consistent():
    """pyproject 与包元数据必须一致，且顶层模块 run_all 必须随包分发。"""
    if tomllib is None:
        return
    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject["project"]
    import lega

    assert project["version"] == lega.__version__, (
        f"pyproject 版本 {project['version']} 与 lega.__version__ "
        f"{lega.__version__} 不一致"
    )
    setuptools_cfg = pyproject["tool"]["setuptools"]
    assert "run_all" in setuptools_cfg.get("py-modules", []), (
        "pyproject 未声明 py-modules=['run_all']：wheel 不含该顶层模块，"
        "lega-run 控制台脚本在非 editable 安装下会 ImportError"
    )
    declared = {}
    for spec in project.get("dependencies", []):
        name, _, version = spec.partition(">")
        declared[name.strip().lower()] = (">" + version.strip()).strip()
    from scripts.check_requirements_lock import parse_requirements

    reqs = parse_requirements(ROOT / "requirements.txt")
    assert set(declared) == set(
        reqs
    ), f"pyproject 依赖 {sorted(declared)} 与 requirements.txt {sorted(reqs)} 不一致"
    for name, spec in reqs.items():
        assert (
            declared[name] == spec
        ), f"{name} 的版本约束不一致：pyproject={declared[name]} vs requirements={spec}"
    print("  [ok] 打包元数据、版本号与依赖声明一致")


def test_pytest_free_runner_covers_every_test_module():
    """``python -m tests.run_tests`` 必须覆盖全部测试模块，不能静默漏测。"""
    from tests.run_tests import MODULES

    listed = {name.split(".")[-1] for name in MODULES}
    actual = {p.stem for p in (ROOT / "tests").glob("test_*.py")}
    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    assert not missing, f"tests/run_tests.py 的模块清单缺少: {missing}"
    assert not stale, f"tests/run_tests.py 的模块清单含已删除模块: {stale}"
    print(f"  [ok] 无 pytest 入口覆盖全部 {len(actual)} 个测试模块")


def test_docs_directory_only_keeps_design_notes():
    """docs/ 只保留 design_notes.md（流水线清理阶段的既定策略）。"""
    docs = ROOT / "docs"
    if not docs.is_dir():
        return
    names = {p.name for p in docs.iterdir()}
    extra = sorted(names - ALLOWED_DOCS)
    assert not extra, f"docs/ 目录存在计划外文件: {extra}"
    print("  [ok] docs/ 只保留 design_notes.md")


def test_pipeline_reports_record_the_new_determinism_audit():
    """汇总页必须能展示哈希盐/确定性档位/非确定性算子告警的审计结论。"""
    from lega.pipeline.report import sections

    source = Path(sections.__file__).read_text(encoding="utf-8")
    for needle in (
        "pythonhashseed",
        "deterministic_mode",
        "non_deterministic_op_warnings",
    ):
        assert needle in source, f"汇总页未渲染确定性审计字段: {needle}"
    print("  [ok] 汇总页渲染确定性审计字段")


def test_json_report_helpers_keep_native_types():
    """溯源字段必须都是 JSON 原生类型（可直接写入结果 JSON/checkpoint）。"""
    from lega.repro import runtime_info

    info = runtime_info("cpu")
    json.dumps(info)  # 非原生类型会在这里直接抛错
    assert isinstance(info["pythonhashseed"], dict)
    assert isinstance(info["pythonhashseed"]["fixed"], bool)
    assert info["deterministic_mode"] in ("strict", "warn_only", "invalid")
    assert isinstance(info["non_deterministic_op_warnings"], int)
    print("  [ok] 运行环境快照保持 JSON 原生类型")
