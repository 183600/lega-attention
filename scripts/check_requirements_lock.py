#!/usr/bin/env python3
"""Check dependency declarations for internal consistency.

The script ensures that the dependency stack of the project is declared in
exactly one place per concern, so that a stale copy cannot silently drift:

  * ``requirements.txt``       — direct runtime dependencies (lower bounds);
  * ``requirements-lock.txt``  — exact pins for the formal experiment env;
  * ``requirements-ci.txt``    — exact pins for the CI/test toolchain;
  * ``pyproject.toml``         — packaging metadata (install requirements).

Checks performed:

  1. every direct dependency in ``requirements.txt`` is pinned in
     ``requirements-lock.txt`` and the pin satisfies the lower bound;
  2. packages shared by ``requirements-lock.txt`` and ``requirements-ci.txt``
     are pinned to the **same** version (CI must exercise the locked stack);
  3. ``pyproject.toml`` dependencies match ``requirements.txt`` one by one
     (same set, same version specifiers) — otherwise ``pip install .`` and
     ``pip install -r requirements.txt`` install different stacks;
  4. the Python version declared in the lock header is compatible with
     ``requires-python`` from ``pyproject.toml``;
  5. ``Dockerfile`` builds the *same* environment: the runtime stack comes
     from ``requirements-lock.txt`` only, every locked package is installed
     with the locked version (never ``--upgrade``, never a literal that
     drifts), no undeclared package enters the runtime image, the base image
     can be pinned by digest, and the image build runs the same environment
     gate the pipeline applies to formal runs
     (``python -m lega.repro --check-env``).

Why the container needs the same discipline
===========================================
A container that installs "whatever torch is newest" is a *different* research
environment, not a reproducible one: the pipeline's fail-closed environment
gate would abort inside such an image (``environment_lock_mismatch``), and if
the lock were ever relaxed the results would silently come from an undeclared
dependency stack. The image is therefore held to the lock, and the check is
executable rather than a convention.
"""

import re
import sys
from pathlib import Path


# 构建工具本身不属于运行时依赖栈：pip/setuptools/wheel 在镜像里可以按任意
# 版本安装，它们不参与实验数值，也不进入环境锁审计。
BUILD_TOOL_PACKAGES = frozenset({"pip", "setuptools", "wheel"})

# 镜像构建期必须执行的环境闸门：与流水线正式实验用的是同一把尺子
# （lega.repro.environment_gate / environment_lock_report）。
ENV_GATE_COMMAND = "python -m lega.repro --check-env"

# 会吃掉下一个 token 的 pip 选项：它们的取值不是"要安装的包"。
PIP_OPTIONS_WITH_VALUE = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-f",
        "--find-links",
        "-i",
        "--index-url",
        "--extra-index-url",
        "--trusted-host",
        "-e",
        "--editable",
        "--target",
        "--prefix",
        "--proxy",
        "--python-version",
        "--platform",
    }
)


def parse_requirements(path: Path) -> dict[str, str]:
    """Parse requirements file into {package: version_spec} dict."""
    specs = {}
    content = path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle package[extra]>=version format
        match = re.match(r"^([A-Za-z0-9._\-]+(?:\[[^\]]+\])?)(.*)$", line)
        if not match:
            continue
        pkg = match.group(1).lower().replace("_", "-")
        version_spec = match.group(2).strip()
        specs[pkg] = version_spec
    return specs


def parse_lock(path: Path) -> dict[str, str]:
    """Parse lock file into {package: exact_version} dict."""
    locked = {}
    content = path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle package==version format
        match = re.match(r"^([A-Za-z0-9._\-]+)==([^\s#]+)", line)
        if not match:
            continue
        pkg = match.group(1).lower().replace("_", "-")
        version = match.group(2)
        locked[pkg] = version
    return locked


def parse_python_header(path: Path) -> str | None:
    """Lock 文件头部声明的 Python 版本（``# python: 3.12``）；没有则 None。"""
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#\s*python\s*:\s*([0-9]+(?:\.[0-9]+){1,2})\s*$", line, re.I)
        if match:
            return match.group(1)
    return None


def load_pyproject(path: Path) -> dict | None:
    """读取 pyproject.toml 的 [project] 段；无 tomllib 或文件缺失时返回 None。"""
    if not path.exists():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:
        return None
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError):
        return None
    project = data.get("project")
    return project if isinstance(project, dict) else None


def version_satisfies(version: str, spec: str) -> bool:
    """Check if a version satisfies a version specifier (simplified)."""
    if not spec:
        return True
    # Handle >=, >, <=, <, ==, !=, ~= operators
    ops = [">=", ">", "<=", "<", "==", "!=", "~="]
    for op in ops:
        if spec.startswith(op):
            required = spec[len(op) :].strip()
            if op == ">=":
                return compare_versions(version, required) >= 0
            elif op == ">":
                return compare_versions(version, required) > 0
            elif op == "<=":
                return compare_versions(version, required) <= 0
            elif op == "<":
                return compare_versions(version, required) < 0
            elif op == "==":
                return compare_versions(version, required) == 0
            elif op == "!=":
                return compare_versions(version, required) != 0
            elif op == "~=":
                # Compatible release: ~= X.Y means >= X.Y, == X.*
                return compare_versions(version, required) >= 0 and version.startswith(
                    required.rsplit(".", 1)[0] + "."
                )
    return True


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    # Simple version comparison - split by dots and compare numerically
    parts1 = [int(x) for x in re.split(r"[.\-]", v1) if x.isdigit()]
    parts2 = [int(x) for x in re.split(r"[.\-]", v2) if x.isdigit()]

    # Pad shorter with zeros
    max_len = max(len(parts1), len(parts2))
    parts1 += [0] * (max_len - len(parts1))
    parts2 += [0] * (max_len - len(parts2))

    for p1, p2 in zip(parts1, parts2):
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
    return 0


def normalize_spec(spec: str) -> str:
    """版本约束的归一化形式：去空白、统一大小写与包名写法。"""
    text = re.sub(r"\s+", "", spec or "")
    return text.lower()


# --------------------------------------------------------------------------
# Dockerfile（容器镜像）与依赖锁的一致性检查
# --------------------------------------------------------------------------

LOCK_FILE_NAME = "requirements-lock.txt"


def _logical_lines(text: str) -> list[str]:
    """把 Dockerfile 文本拼成逻辑行（处理行尾续行 ``\\``）。"""
    lines: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        lines.append(buffer)
        buffer = ""
    if buffer:
        lines.append(buffer)
    return [line.strip() for line in lines if line.strip()]


def _split_tokens(command: str) -> list[str]:
    """按 shell 词法切分命令（尊重引号；引号本身不进入 token）。"""
    tokens: list[str] = []
    current = ""
    quote = ""
    for ch in command:
        if quote:
            if ch == quote:
                quote = ""
            else:
                current += ch
            continue
        if ch in "\"'":
            quote = ch
            continue
        if ch.isspace():
            if current:
                tokens.append(current)
                current = ""
            continue
        current += ch
    if current:
        tokens.append(current)
    return tokens


def parse_dockerfile_stages(text: str) -> tuple[list[str], list[dict]]:
    """解析 Dockerfile 的构建阶段。

    返回 ``(全局指令, 阶段列表)``：全局指令是第一个 FROM 之前的部分（ARG 等）；
    每个阶段是 ``{"name", "from_line", "text"}``——``name`` 取 ``AS <name>``，
    匿名阶段按序号命名；``from_line`` 是该阶段的 FROM 指令原文。
    """
    preamble: list[str] = []
    stages: list[dict] = []
    for line in _logical_lines(text):
        if line.upper().startswith("FROM "):
            tokens = _split_tokens(line)
            name = ""
            upper = [token.upper() for token in tokens]
            if "AS" in upper:
                index = upper.index("AS")
                if index + 1 < len(tokens):
                    name = tokens[index + 1]
            if not name:
                name = f"stage{len(stages)}"
            stages.append({"name": name, "from_line": line, "text": ""})
            continue
        if not stages:
            preamble.append(line)
        else:
            stages[-1]["text"] += line + "\n"
    return preamble, stages


def _install_statements(stage_text: str) -> list[dict]:
    """取出阶段内所有 ``pip install`` 语句。

    先按逻辑行拼接（处理 ``\\`` 续行），再按 ``;``/``&&`` 拆成 shell 语句：
    一条 RUN 里常有多个 pip 调用，混成一条命令会把后一个命令的参数当成
    依赖名，凭空造出假告警。每条语句带上它所属的逻辑行——“版本变量是否从
    lock 文件派生”只能在同一逻辑行（同一个 RUN）里判断。
    """
    statements: list[dict] = []
    for logical in _logical_lines(stage_text):
        for statement in re.split(r";|&&", logical):
            statement = statement.strip()
            if not statement:
                continue
            tokens = _split_tokens(statement)
            if "pip" not in tokens or "install" not in tokens:
                continue
            index = tokens.index("install")
            statements.append(
                {
                    "line": logical,
                    "statement": statement,
                    "tokens": tokens[index + 1 :],
                }
            )
    return statements


def _canonical_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _parse_requirement_spec(spec: str) -> tuple[str, str | None]:
    """``name==version`` → ``(name, version)``；其它写法 → ``(name, None)``。

    无版本约束、范围约束（``>=``/``~=``）与直接 URL（``name @ url``）都按
    “没有精确版本”处理：它们都无法证明镜像里的包就是锁文件里的那一个。
    """
    text = spec.strip()
    if not text:
        return "", None
    if " @ " in text:
        return _canonical_name(text.split(" @ ", 1)[0]), None
    match = re.match(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?(.*)$", text)
    if not match:
        return _canonical_name(text), None
    name, rest = match.group(1), match.group(2).strip()
    if rest.startswith("=="):
        return _canonical_name(name), rest[2:].strip()
    return _canonical_name(name), None


def _is_path_like(name: str) -> bool:
    """本地路径/可编辑安装（``.``、``./x``）不是依赖包名。"""
    return not name or name.startswith(".") or "/" in name or "\\" in name


def _requirement_targets(tokens: list[str]) -> tuple[list[str], list[str]]:
    """从 pip install 的 token 里取出 ``(依赖规格, 引用的 requirements 文件)``。"""
    specs: list[str] = []
    files: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("-r", "--requirement", "-c", "--constraint"):
            index += 1
            if index < len(tokens):
                files.append(tokens[index])
        elif token in PIP_OPTIONS_WITH_VALUE:
            index += 1
        elif token.startswith("-"):
            pass
        else:
            specs.append(token)
        index += 1
    return specs, files


def check_container_lock(
    dockerfile_text: str,
    locked: dict[str, str],
    ci_locked: dict[str, str] | None = None,
) -> list[str]:
    """检查 Dockerfile 是否构建了与依赖锁一致的环境；返回问题列表。

    规则：

      * 运行时阶段（文本里出现 ``requirements-lock.txt`` 的阶段）必须执行
        环境闸门 ``python -m lega.repro --check-env``——镜像构建期就拒绝对
        不上锁的依赖栈，而不是等正式实验被流水线中止；
      * 被锁定的包必须按锁定版本安装：不允许无版本约束，不允许字面量版本与
        lock 不一致；版本写成 shell 变量时，该语句必须从 lock 文件派生
        （否则“变量里装的是哪个版本”无从核对）；
      * 不允许 ``--upgrade`` 升级被锁定的包：那会把镜像换成“最新版本”，
        与流水线自己的 fail-closed 环境闸门直接冲突；
      * 运行时阶段不得安装 lock/CI 文件之外的包（torchvision/torchaudio 这类
        未声明依赖会悄悄改变数值栈）；
      * 外部基础镜像必须经 ARG 传入，才能按 digest 固定成同一个镜像。
    """
    ci_locked = ci_locked or {}
    problems: list[str] = []
    preamble, stages = parse_dockerfile_stages(dockerfile_text)
    if not stages:
        return ["Dockerfile 没有 FROM 指令：无法审计构建环境"]

    declared_args = set()
    for line in preamble:
        match = re.match(r"^ARG\s+([A-Za-z0-9_.-]+)", line, re.IGNORECASE)
        if match:
            declared_args.add(match.group(1).upper())

    known_stages: set[str] = set()
    for stage in stages:
        name = stage["name"]
        text = stage["text"]
        from_line = stage["from_line"]
        tokens = _split_tokens(from_line)
        image = tokens[1] if len(tokens) >= 2 else ""
        external = image not in known_stages
        if external:
            if "${" not in image:
                problems.append(
                    f"阶段 {name} 的基础镜像 {image!r} 是写死的 tag："
                    "无法按 digest 固定，正式复现请通过 ARG 传入镜像"
                )
            else:
                for arg in re.findall(r"\$\{([A-Za-z0-9_.-]+)\}", image):
                    if arg.upper() not in declared_args:
                        problems.append(
                            f"阶段 {name} 引用了未声明的构建参数 {arg}"
                        )
        known_stages.add(name)

        statements = _install_statements(text)
        # 运行时阶段的判定：文本里引用了 lock 文件，或安装了任何一个被锁定的
        # 包。只按“引用 lock”判定会漏掉 `FROM base AS gpu` 这种不重复引用
        # lock、却照样装 torch 的阶段——那里最容易混进未声明依赖。
        installs_locked_package = False
        for entry in statements:
            specs, _ = _requirement_targets(entry["tokens"])
            for spec in specs:
                pkg, _version = _parse_requirement_spec(spec)
                if pkg in locked:
                    installs_locked_package = True
        runtime_stage = (LOCK_FILE_NAME in text) or installs_locked_package
        for entry in statements:
            statement = entry["statement"]
            install_tokens = entry["tokens"]
            specs, files = _requirement_targets(install_tokens)
            upgrade = any(token in ("--upgrade", "-U") for token in install_tokens)
            derived_from_lock = any(
                LOCK_FILE_NAME in referenced for referenced in files
            ) or LOCK_FILE_NAME in entry["line"]
            for spec in specs:
                pkg, version = _parse_requirement_spec(spec)
                if not pkg:
                    continue
                if pkg in locked:
                    if version is None:
                        problems.append(
                            f"阶段 {name} 安装 {pkg} 时没有精确版本：镜像里的"
                            f"{pkg} 不保证是锁定的 {locked[pkg]}"
                        )
                    elif "${" in version:
                        if not derived_from_lock:
                            problems.append(
                                f"阶段 {name} 的 {pkg} 版本来自变量 {version}，"
                                "但该语句没有从 requirements-lock.txt 派生："
                                "无法核对镜像里的版本"
                            )
                    elif version.split("+", 1)[0] != locked[pkg].split("+", 1)[0]:
                        problems.append(
                            f"阶段 {name} 安装 {pkg}=={version}，与锁定的 "
                            f"{locked[pkg]} 不一致"
                        )
                    if upgrade:
                        problems.append(
                            f"阶段 {name} 用 --upgrade 安装 {pkg}："
                            "镜像里的版本会漂移到“最新”，不再是锁定环境"
                        )
                elif pkg in ci_locked or pkg in BUILD_TOOL_PACKAGES:
                    continue
                elif _is_path_like(pkg):
                    continue
                elif runtime_stage:
                    problems.append(
                        f"阶段 {name} 安装了未在 requirements-lock.txt/"
                        "requirements-ci.txt 中声明的包 "
                        f"{pkg}：未声明依赖会悄悄改变数值栈"
                    )
        if runtime_stage and ENV_GATE_COMMAND not in text:
            problems.append(
                f"阶段 {name} 安装了锁定运行时依赖却没有执行环境闸门"
                f"（{ENV_GATE_COMMAND}）：镜像可能带着与 lock 不一致的依赖栈发布"
            )
    return problems


def main():
    root = Path(__file__).parent.parent
    req_file = root / "requirements.txt"
    lock_file = root / "requirements-lock.txt"
    ci_file = root / "requirements-ci.txt"
    pyproject_file = root / "pyproject.toml"

    if not req_file.exists():
        print(f"ERROR: {req_file} not found")
        return 1

    if not lock_file.exists():
        print(f"ERROR: {lock_file} not found")
        return 1

    reqs = parse_requirements(req_file)
    locked = parse_lock(lock_file)

    errors = []
    warnings = []

    # Check all requirements have corresponding lock entries
    for pkg, spec in reqs.items():
        if pkg not in locked:
            errors.append(
                f"Package '{pkg}' in requirements.txt not found in requirements-lock.txt"
            )
            continue

        locked_version = locked[pkg]
        if not version_satisfies(locked_version, spec):
            errors.append(
                f"Locked version '{locked_version}' for '{pkg}' does not satisfy "
                f"requirement '{spec}'"
            )

    # Check for extra packages in lock file (warn only)
    for pkg in locked:
        if pkg not in reqs:
            warnings.append(
                f"Package '{pkg}' in requirements-lock.txt not in requirements.txt"
            )

    # CI 必须测试被锁定的同一套版本：共享包的固定版本必须完全一致，
    # 否则 CI 通过并不能说明锁定环境可用。
    ci_locked: dict[str, str] = {}
    if ci_file.exists():
        ci_locked = parse_lock(ci_file)
        for pkg, version in locked.items():
            if pkg not in ci_locked:
                warnings.append(
                    f"Package '{pkg}' in requirements-lock.txt not pinned in requirements-ci.txt"
                )
                continue
            if ci_locked[pkg] != version:
                errors.append(
                    f"Version mismatch for '{pkg}': lock={version} vs ci={ci_locked[pkg]}"
                )
    else:
        warnings.append(f"{ci_file.name} not found: skipped lock/CI cross-check")

    # 容器镜像必须构建同一个环境：镜像里装的是“最新 torch”时，流水线自己的
    # fail-closed 环境闸门会在正式实验里中止，镜像也就不是可复现环境。
    dockerfile = root / "Dockerfile"
    if dockerfile.exists():
        container_problems = check_container_lock(
            dockerfile.read_text(encoding="utf-8"), locked, ci_locked
        )
        errors.extend(f"Dockerfile: {problem}" for problem in container_problems)
    else:
        warnings.append("Dockerfile not found: skipped container/lock cross-check")

    # pyproject 与 requirements.txt 必须声明同一套直接依赖：两处各写一份时，
    # `pip install .` 与 `pip install -r requirements.txt` 会装出不同的栈。
    project = load_pyproject(pyproject_file)
    if project is None:
        warnings.append(
            "pyproject.toml unavailable (missing file or Python < 3.11): "
            "skipped packaging cross-check"
        )
    else:
        declared = project.get("dependencies")
        if isinstance(declared, list):
            parsed_declared = parse_requirements_lines(declared)
            for pkg, spec in parsed_declared.items():
                if pkg not in reqs:
                    errors.append(
                        f"Package '{pkg}' in pyproject.toml not found in requirements.txt"
                    )
                elif normalize_spec(reqs[pkg]) != normalize_spec(spec):
                    errors.append(
                        f"Version specifier mismatch for '{pkg}': "
                        f"pyproject='{spec}' vs requirements.txt='{reqs[pkg]}'"
                    )
            for pkg in reqs:
                if pkg not in parsed_declared:
                    errors.append(
                        f"Package '{pkg}' in requirements.txt not declared in pyproject.toml"
                    )
        else:
            warnings.append(
                "pyproject.toml has no [project].dependencies: skipped check"
            )

        lock_python = parse_python_header(lock_file)
        requires_python = project.get("requires-python")
        if lock_python and isinstance(requires_python, str) and requires_python:
            if not version_satisfies(lock_python, requires_python):
                errors.append(
                    f"Locked Python '{lock_python}' does not satisfy "
                    f"requires-python '{requires_python}'"
                )

    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("OK: requirements-lock.txt is consistent with requirements.txt")
    print("OK: requirements-ci.txt pins match requirements-lock.txt")
    print("OK: pyproject.toml dependencies match requirements.txt")
    print("OK: Dockerfile builds the locked environment (and runs the env gate)")
    return 0


def parse_requirements_lines(lines: list[str]) -> dict[str, str]:
    """解析 pyproject 的 dependencies 列表（每行一个包+约束）。"""
    specs = {}
    for line in lines:
        text = str(line).strip()
        if not text or text.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9._\-]+(?:\[[^\]]+\])?)(.*)$", text)
        if not match:
            continue
        pkg = match.group(1).lower().replace("_", "-")
        specs[pkg] = match.group(2).strip()
    return specs


if __name__ == "__main__":
    sys.exit(main())
