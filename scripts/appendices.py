r"""Write paste-ready thesis appendices A-D (T-22).

Usage::

    uv run python scripts/appendices.py --out docs/appendices
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sim.appendices import write_appendices


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble ``a.md``-``d.md`` from the canonical FSM, prompts and cases."""
    parser = argparse.ArgumentParser(
        description="Write paste-ready appendices A-D into a directory."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/appendices"),
        help="directory for a.md, b.md, c.md and d.md",
    )
    args = parser.parse_args(argv)
    write_appendices(args.out)
    print(
        f"sim: wrote {args.out / 'a.md'}, {args.out / 'b.md'}, "
        f"{args.out / 'c.md'} and {args.out / 'd.md'}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
