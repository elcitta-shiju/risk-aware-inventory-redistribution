"""Safe, read-only extraction of the original dataset archives into data/raw/.

The original archives are never modified. Each archive is extracted into its own
sub-folder; members with absolute paths, '..' traversal or executable extensions are
skipped, and existing files are never overwritten.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from inventory_rl.utils.paths import RAW_DIR, source_data_dir

# Known archives -> (raw sub-folder, source label). Unknown archives get a slugged folder.
KNOWN_ARCHIVES: dict[str, tuple[str, str]] = {
    "archive (1).zip": ("kaggle_sales_data", "Kaggle"),
    "archive (2).zip": ("kaggle_retail_store_inventory", "Kaggle"),
    "Retail Transactions and Stocks Data.zip": ("mendeley_retail_transactions_stocks", "Mendeley Data"),
}

ALLOWED_SUFFIXES = {".csv", ".txt", ".md", ".json", ".xlsx", ".xls", ".parquet", ".pdf"}


@dataclass
class ExtractionRecord:
    archive: str
    target_dir: str
    source: str
    extracted: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def is_safe_member(name: str) -> bool:
    """Reject absolute paths, drive letters, parent traversal and non-data file types."""
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        return False
    if name.endswith("/"):
        return True  # directory entry
    return p.suffix.lower() in ALLOWED_SUFFIXES


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in Path(name).stem).strip("_").lower()


def extract_archive(archive: Path, raw_dir: Path = RAW_DIR) -> ExtractionRecord:
    sub, source = KNOWN_ARCHIVES.get(archive.name, (_slug(archive.name), "unknown"))
    target = raw_dir / sub
    rec = ExtractionRecord(archive=str(archive), target_dir=str(target), source=source)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not is_safe_member(info.filename):
                rec.rejected.append(info.filename)
                continue
            # Flatten the member's leading folder, if it just repeats the archive name.
            out = target / Path(info.filename).name
            if out.exists() and out.stat().st_size == info.file_size:
                rec.skipped_existing.append(out.name)
                continue
            if out.exists():
                rec.rejected.append(f"{info.filename} (exists with different size; not overwritten)")
                continue
            with zf.open(info) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            rec.extracted.append(out.name)
    return rec


def extract_all(source_dir: Path | None = None, raw_dir: Path = RAW_DIR) -> list[ExtractionRecord]:
    source_dir = source_dir or source_data_dir()
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {source_dir} (set INVENTORY_DATA_DIR)")
    records = [extract_archive(a, raw_dir) for a in sorted(source_dir.glob("*.zip"))]
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "extraction_manifest.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
    )
    return records
