#!/usr/bin/env python3
"""Check that requirements-lock.txt is consistent with requirements.txt.

This script ensures that the locked versions in requirements-lock.txt are
compatible with the lower bounds specified in requirements.txt, and that
all direct dependencies from requirements.txt are present in the lock file.
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
        match = re.match(r'^([A-Za-z0-9._\-]+(?:\[[^\]]+\])?)(.*)$', line)
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
        match = re.match(r'^([A-Za-z0-9._\-]+)==([^\s#]+)', line)
        if not match:
            continue
        pkg = match.group(1).lower().replace("_", "-")
        version = match.group(2)
        locked[pkg] = version
    return locked


def version_satisfies(version: str, spec: str) -> bool:
    """Check if a version satisfies a version specifier (simplified)."""
    if not spec:
        return True
    # Handle >=, >, <=, <, ==, !=, ~= operators
    ops = [">=", ">", "<=", "<", "==", "!=", "~="]
    for op in ops:
        if spec.startswith(op):
            required = spec[len(op):].strip()
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
                return compare_versions(version, required) >= 0 and version.startswith(required.rsplit(".", 1)[0] + ".")
    return True


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    # Simple version comparison - split by dots and compare numerically
    parts1 = [int(x) for x in re.split(r'[.\-]', v1) if x.isdigit()]
    parts2 = [int(x) for x in re.split(r'[.\-]', v2) if x.isdigit()]
    
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


def main():
    root = Path(__file__).parent.parent
    req_file = root / "requirements.txt"
    lock_file = root / "requirements-lock.txt"
    
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
            errors.append(f"Package '{pkg}' in requirements.txt not found in requirements-lock.txt")
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
            warnings.append(f"Package '{pkg}' in requirements-lock.txt not in requirements.txt")
    
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    
    print("OK: requirements-lock.txt is consistent with requirements.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())