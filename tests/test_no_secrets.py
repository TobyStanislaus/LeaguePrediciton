"""Guards against ever committing the Riot API key or raw cache data.

These run in CI / pre-commit alongside the feature-leakage tests.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_scanner():
    path = REPO_ROOT / "scripts" / "check_secrets.py"
    spec = importlib.util.spec_from_file_location("check_secrets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_ignored(relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relpath],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.parametrize("relpath", [".env", ".env.local", "data/cache/matches.sqlite"])
def test_secret_and_cache_paths_are_gitignored(relpath):
    assert _is_ignored(relpath), f"{relpath} is NOT gitignored"


def test_env_example_is_committable():
    assert not _is_ignored(".env.example"), ".env.example must stay tracked"


def test_env_example_holds_only_a_placeholder():
    """.env.example is committed, so its value must never be a real key.

    Asserted by shape rather than exact text -- the wording of the placeholder
    is allowed to change, its being a placeholder is not.
    """
    scanner = _load_scanner()
    lines = [
        line.strip()
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == [lines[0]], f".env.example should hold one setting, got {lines}"

    name, _, value = lines[0].partition("=")
    assert name.strip() == "RIOT_API_KEY"
    assert scanner.PLACEHOLDER_VALUE.match(value.strip()), (
        f".env.example value {value.strip()!r} is not a recognised placeholder"
    )
    assert scanner._line_problem(lines[0]) is None


def test_env_is_not_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert ".env" not in tracked
    assert not any(p.startswith("data/cache/") for p in tracked)


def test_no_api_key_literal_in_committable_files():
    scanner = _load_scanner()
    problems = scanner.scan_committable()
    assert problems == [], "possible secret in committable files: " + "; ".join(problems)


def test_scanner_covers_untracked_files_too():
    """.env.example is committable while still untracked -- it must be scanned.

    Scanning only ``git ls-files`` would skip a brand-new file holding a real
    key right up until the moment someone stages it.
    """
    scanner = _load_scanner()
    paths = scanner.committable_paths()
    assert ".env.example" in paths, "untracked-but-committable files are not being scanned"
    assert ".env" not in paths, ".env is gitignored and must stay out of the scan"


def test_scanner_detects_a_key_literal():
    """The scanner must not be silently broken."""
    scanner = _load_scanner()
    fake_key = "RGAPI-" + "0123abcd-4567-89ef-0123-456789abcdef"
    assert scanner.KEY_PATTERN.search(f'key = "{fake_key}"')
    assert not scanner.KEY_PATTERN.search("RIOT_API_KEY=your_key_here")


def test_scanner_flags_the_right_lines():
    """Regression cases: real-looking keys flagged, documentation prose is not."""
    scanner = _load_scanner()
    fake_key = "RGAPI-" + "0123abcd-4567-89ef-0123-456789abcdef"
    cases = [
        (f'API_KEY = "{fake_key}"', True),
        (f"RIOT_API_KEY={fake_key}", True),
        ('RIOT_API_KEY: "abcd1234efgh"', True),  # pragma: allowlist secret
        ("RIOT_API_KEY=your_key_here", False),
        ("RIOT_API_KEY=your_api_key_here", False),
        ("RIOT_API_KEY=changeme", False),
        ('key = os.environ["RIOT_API_KEY"]', False),
        ("RIOT_API_KEY=<paste your key here>", False),
        ("The RIOT_API_KEY variable is read from the environment", False),
    ]
    for text, should_flag in cases:
        flagged = scanner._line_problem(text) is not None
        assert flagged is should_flag, f"wrong verdict for: {text}"


def test_allowlist_pragma_suppresses_a_line():
    scanner = _load_scanner()
    fake_key = "RGAPI-" + "0123abcd-4567-89ef-0123-456789abcdef"
    assert scanner._line_problem(f'k = "{fake_key}"') == "Riot API key literal"
    assert scanner._line_problem(f'k = "{fake_key}"  # {scanner.ALLOWLIST_MARKER}') is None
