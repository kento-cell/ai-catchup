"""Command-line interface.

    aicatchup run              # full pipeline → Slack (or stdout)
    aicatchup run --dry-run    # no Slack post, no dedup marking
    aicatchup run --top 5      # override delivered item count
    aicatchup sources          # list configured sources
    aicatchup stats            # knowledge base size
"""
from __future__ import annotations

import argparse
import logging
import os

from .config import Config, load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(prog="aicatchup", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full catchup pipeline")
    run_p.add_argument("--dry-run", action="store_true",
                       help="print to stdout, do not post to Slack or mark dedup")
    run_p.add_argument("--top", type=int, default=None, help="items to deliver")
    run_p.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("sources", help="list configured sources")
    sub.add_parser("stats", help="show knowledge base stats")

    args = parser.parse_args()
    load_dotenv()

    if args.command == "sources":
        from .sources import SOURCES
        for name, kind, _url, tier in SOURCES:
            print(f"tier {tier} | {kind:>9} | {name}")
        return

    if args.command == "stats":
        from .knowledge import Knowledge
        cfg = Config()
        kb = Knowledge(cfg.data_dir / "knowledge.db")
        print(f"knowledge base: {kb.count()} items ({cfg.data_dir / 'knowledge.db'})")
        kb.close()
        return

    # run
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.top:
        os.environ["CATCHUP_TOP_N"] = str(args.top)
    cfg = Config()
    from .graph import run
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
