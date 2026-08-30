"""Refuse to commit anything that looks like a Riot API key.

Run before every commit:

    python scripts/check_secrets.py        # scans the staged diff
    python scripts/check_secrets.py --all  # scans every committable file

"Committable" means tracked *plus* untracked-but-not-ignored. A file like
.env.example is dangerous precisely while it is still untracked: scanning only
`git ls-files` would skip it right up until the moment it gets staged.

Exit code 0 = clean, 1 = suspected secret (do not commit), 2 = usage error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# A real Riot key is the literal prefix followed by a UUID. Matching the full
# shape (rather than the bare prefix) means this file, the README and
# .env.example can talk about keys without tripping the scanner.
KEY_PATTERN = re.compile(
    r"RGAPI-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# Anything key-shaped assigned to the key variable. Requiring >=8 chars of key
# alphabet keeps documentation prose ("set RIOT_API_KEY=<your key>") from
# tripping the scanner.
ASSIGN_PATTERN = re.compile(r"RIOT_API_KEY\s*[=:]\s*[\"']?([A-Za-z0-9_-]{8,})")

# Recognised placeholder values. Matching on the *value* rather than exempting
# .env.example wholesale means that file still gets scanned for real secrets.
PLACEHOLDER_VALUE = re.compile(
    r"^(your[_-]?\w*[_-]?key[_-]?here|changeme|placeholder|todo|xxx+)$", re.IGNORECASE
)

FORBIDDEN_PATHS = {".env"}

# Escape hatch for lines that must *look* like a key -- the scanner's own test
# fixtures, mainly. Deliberately explicit and greppable.
ALLOWLIST_MARKER = "pragma: allowlist secret"


def _line_problem(line: str) -> str | None:
    """Classify a single line. Returns a short reason, or None if it looks clean."""
    if ALLOWLIST_MARKER in line:
        return None
    if KEY_PATTERN.search(line):
        return "Riot API key literal"
    if "os.environ" in line or "getenv" in line:
        return None
    match = ASSIGN_PATTERN.search(line)
    if match and not PLACEHOLDER_VALUE.match(match.group(1)):
        return "non-placeholder key assignment"
    return None


def _git(*args: str) -> str:
    """Run git and return stdout as UTF-8.

    The encoding is explicit because Python otherwise decodes with the locale
    codec -- cp1252 on Windows -- and any non-Latin-1 character anywhere in the
    diff (an emoji in a README is enough) raises UnicodeDecodeError inside a
    reader thread, leaving stdout as None. A scanner that dies on the content
    it is meant to inspect is worse than no scanner, so decode defensively and
    never let this return None.
    """
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout or ""


def _staged_paths() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [p for p in out.splitlines() if p]


def scan_staged() -> list[str]:
    problems: list[str] = []

    for path in _staged_paths():
        if path in FORBIDDEN_PATHS or path.startswith("data/cache/"):
            problems.append(f"{path}: this file must never be committed")

    diff = _git("diff", "--cached", "--unified=0")
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:]
        reason = _line_problem(body)
        if reason:
            problems.append(f"staged diff: {reason} in -> {body.strip()[:80]}")
    return problems


def committable_paths() -> list[str]:
    """Every file git would let you commit: tracked + untracked, minus ignored.

    ``--exclude-standard`` applies .gitignore, so .env is correctly left out --
    that file is *supposed* to hold the key.
    """
    out = _git("ls-files", "--cached", "--others", "--exclude-standard")
    return [p for p in out.splitlines() if p]


def scan_committable() -> list[str]:
    problems: list[str] = []
    for path in committable_paths():
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            reason = _line_problem(line)
            if reason:
                problems.append(f"{path}:{lineno}: {reason}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        "--tracked",
        dest="scan_all",
        action="store_true",
        help="scan every committable file (tracked + untracked, excluding gitignored)"
        " instead of just the staged diff",
    )
    args = parser.parse_args()

    problems = scan_committable() if args.scan_all else scan_staged()

    if problems:
        print("SECRET SCAN FAILED -- do not commit:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    scope = "committable files" if args.scan_all else "staged diff"
    print(f"secret scan clean ({scope})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
