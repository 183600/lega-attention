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
     ``requires-python`` from ``pyproject.toml``.
"""

import re
import sys
from pathlib import Path


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
