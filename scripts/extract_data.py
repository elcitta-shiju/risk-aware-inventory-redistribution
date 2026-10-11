"""Extract the original dataset archives into data/raw/ (originals untouched)."""

import _bootstrap  # noqa: F401

from inventory_rl.data.extract import extract_all

if __name__ == "__main__":
    for rec in extract_all():
        print(f"{rec.source:14s} {rec.target_dir}")
        print(f"   extracted={rec.extracted} skipped={rec.skipped_existing} rejected={rec.rejected}")
