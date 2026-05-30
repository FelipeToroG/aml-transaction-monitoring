#!/usr/bin/env python3
"""Refresh the README results section from the training summary.

Reads ``mlruns/training_summary.json`` (written by
``src.models.train``), generates the Markdown results block via
``src.evaluation.reports.build_results_markdown``, and replaces the
content between the README's ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->`` markers in place.

This script is invoked automatically at the end of
``scripts/train.sh``. It is also safe to run by hand any time the
summary JSON changes.

Exit codes:
    0   README updated.
    1   Summary JSON missing or unreadable.
    2   README markers missing or malformed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make the project root importable so this script can import the
# reports module regardless of where it is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reports import build_results_markdown  # noqa: E402

# Marker tokens in the README that bound the results block. Must match
# the markers in README.md exactly; the regex below builds the matcher
# from these constants so any change here propagates.
START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"


def update_readme(
    *,
    summary_path: Path,
    readme_path: Path,
) -> None:
    """Read the summary, render the block, splice it into the README."""
    if not summary_path.exists():
        print(
            f"ERROR: Training summary not found at {summary_path}. "
            "Run scripts/train.sh first to produce the summary.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: Training summary at {summary_path} is not valid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not readme_path.exists():
        print(f"ERROR: README not found at {readme_path}.", file=sys.stderr)
        sys.exit(2)

    readme_content = readme_path.read_text()
    rendered = build_results_markdown(summary)

    # Build the replacement block. The leading and trailing newlines
    # frame the rendered content so the result reads cleanly against
    # the surrounding README prose without depending on the existing
    # whitespace inside the markers.
    replacement = f"{START_MARKER}\n\n{rendered}\n{END_MARKER}"

    # The (?s) inline flag enables DOTALL so the regex captures across
    # newlines. The non-greedy quantifier prevents the match from
    # spilling past the END marker if multiple blocks ever coexist.
    pattern = re.compile(
        f"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )

    if not pattern.search(readme_content):
        print(
            f"ERROR: Could not find the results markers in {readme_path}.\n"
            f"  Expected: {START_MARKER} ... {END_MARKER}",
            file=sys.stderr,
        )
        sys.exit(2)

    new_content = pattern.sub(replacement, readme_content)
    readme_path.write_text(new_content)
    print(f"Updated results block in {readme_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "mlruns" / "training_summary.json",
        help="Path to the training summary JSON.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=PROJECT_ROOT / "README.md",
        help="Path to the README to update.",
    )
    args = parser.parse_args()
    update_readme(summary_path=args.summary, readme_path=args.readme)


if __name__ == "__main__":
    main()
