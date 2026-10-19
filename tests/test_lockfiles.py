"""The hashed lockfiles must install on every OS we run on, not just the one that made them.

pip-compile only follows a dependency whose environment marker matches the machine it
runs on. The lockfiles are compiled on Windows, so ``ipython``'s
``pexpect; sys_platform != "win32"`` was silently dropped, and CI's Linux
``pip install --require-hashes`` failed on the first push. This test evaluates every
installed, pinned package's requirements under both Linux and Windows markers.
"""

from __future__ import annotations

import re
from importlib.metadata import Distribution, distributions
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements"
PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==", re.MULTILINE)
PLATFORMS: dict[str, dict[str, str]] = {
    "linux": {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix"},
    "windows": {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt"},
}


def _pinned(*locks: str) -> set[NormalizedName]:
    names: set[NormalizedName] = set()
    for lock in locks:
        text = (REQUIREMENTS / f"{lock}.txt").read_text(encoding="utf-8")
        names.update(canonicalize_name(m) for m in PIN_RE.findall(text))
    return names


def _installed() -> dict[NormalizedName, Distribution]:
    found: dict[NormalizedName, Distribution] = {}
    for dist in distributions():
        found.setdefault(canonicalize_name(dist.metadata["Name"]), dist)
    return found


def _missing(pinned: set[NormalizedName], platform: dict[str, str]) -> dict[str, str]:
    env: dict[str, Any] = {**default_environment(), **platform, "extra": ""}
    installed = _installed()
    missing: dict[str, str] = {}
    for name in sorted(pinned & installed.keys()):
        for raw in installed[name].requires or []:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            if canonicalize_name(req.name) not in pinned:
                missing[req.name] = f"needed by {name}"
    return missing


@pytest.mark.parametrize("platform", sorted(PLATFORMS))
def test_ci_lockfiles_are_complete_on_every_platform(platform: str) -> None:
    # base + dev is exactly what CI and `make setup` install.
    assert _missing(_pinned("base", "dev"), PLATFORMS[platform]) == {}


def test_known_linux_only_dependency_is_pinned() -> None:
    assert {"pexpect", "ptyprocess"} <= _pinned("dev")
