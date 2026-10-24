"""Apply a line-by-line review to the pages the pairer sent to ``review/``.

A decisions file records what a reviewer decided for each line, so the review
can be applied again to a fresh run of ``ticha lines`` (same
segmentation, same recognition model) without redoing it by hand::

    # doc page line action [text]
    Al642 2 4 fix 50 p[eso]s cama xiteni xico
    Al642 2 11 drop
    Al683aT 3 5 keep

``line`` is the number of the line among the page's non-fragment lines, in
reading order (the ``line`` column of ``lines.csv``). ``keep`` keeps the paired
text, ``drop`` leaves the line without text, ``fix`` replaces the text. Lines
not listed keep their text if the pair is confident (``conf >= 0.5``) and are
left without text otherwise.

The reviewed pages are written as PAGE XML to ``reviewed_dir`` with the same
file names, where the notebook and ``scripts/build_dataset.py`` pick them up.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from .line_pairing import PAGE_NS

ACTIONS = ("keep", "drop", "fix")
Key = Tuple[str, int, int]


@dataclass
class Decision:
    action: str
    text: Optional[str] = None


def read_decisions(path: Path) -> Dict[Key, Decision]:
    """Parse a decisions file: ``doc page line action [text]`` per line."""
    decisions: Dict[Key, Decision] = {}
    for number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split(" ", 4)
        if len(parts) < 4 or parts[3] not in ACTIONS:
            raise ValueError(
                f"{path}:{number}: expected 'doc page line keep|drop|fix [text]'"
            )
        doc, page, line, action = parts[:4]
        text = parts[4].strip() if len(parts) > 4 else None
        if action == "fix" and not text:
            raise ValueError(f"{path}:{number}: 'fix' needs the corrected text")
        decisions[(doc, int(page), int(line))] = Decision(action, text)
    return decisions


def _q(tag: str) -> str:
    return f"{{{PAGE_NS}}}{tag}"


def main_lines(root: ET.Element) -> list:
    """Non-fragment TextLines of a page, in reading order (line ids l1, l2...)."""
    lines = [
        el for el in root.iter(_q("TextLine")) if "fragment" not in el.get("custom", "")
    ]
    return sorted(lines, key=lambda el: int(re.sub(r"\D", "", el.get("id")) or 0))


def apply_to_page(
    xml_path: Path,
    target: Path,
    doc_id: str,
    page: int,
    decisions: Dict[Key, Decision],
    min_conf: float = 0.5,
    note: str = "",
) -> Dict[str, int]:
    """Write a reviewed copy of one review page; returns counts per outcome."""
    ET.register_namespace("", PAGE_NS)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    counts = {"kept": 0, "fixed": 0, "dropped": 0}
    for number, el in enumerate(main_lines(root), 1):
        equiv = el.find(_q("TextEquiv"))
        text = equiv.findtext(_q("Unicode")) if equiv is not None else None
        conf = (
            float(equiv.get("conf"))
            if equiv is not None and equiv.get("conf")
            else None
        )
        decision = decisions.get((doc_id, page, number))
        if decision and decision.action == "fix":
            new = decision.text
            counts["fixed"] += 1
        elif (decision and decision.action == "keep") or (
            decision is None and text and conf is not None and conf >= min_conf
        ):
            new = text
            counts["kept"] += bool(text)
        else:
            new = None
            counts["dropped"] += bool(text)
        if equiv is not None:
            el.remove(equiv)
        if new:
            equiv = ET.SubElement(el, _q("TextEquiv"), conf="1.000")
            ET.SubElement(equiv, _q("Unicode")).text = new
    if note:
        meta = root.find(_q("Metadata"))
        comments = meta.find(_q("Comments"))
        if comments is None:
            comments = ET.SubElement(meta, _q("Comments"))
        comments.text = f"{comments.text or ''} | {note}".strip(" |")
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return counts


def apply_review(
    paired_dir: Path,
    decisions_path: Path,
    reviewed_dir: Path,
    min_conf: float = 0.5,
    note: str = "reviewed line by line against the image (see the decisions file)",
    skip_existing: bool = False,
) -> pd.DataFrame:
    """
    Apply a decisions file to every ``review`` page of ``paired_dir/manifest.csv``.

    Returns one row per page with the counts of kept, fixed and dropped lines,
    plus the decisions for pages that are no longer in review in
    ``DataFrame.attrs["unused"]``. With ``skip_existing``, pages that already
    have a reviewed XML (e.g. corrected by hand in eScriptorium) are left as
    they are.
    """
    paired_dir, reviewed_dir = Path(paired_dir), Path(reviewed_dir)
    decisions = read_decisions(decisions_path)
    manifest = pd.read_csv(paired_dir / "manifest.csv")
    rows, used = [], set()
    for _, row in manifest[manifest["status"] == "review"].iterrows():
        xml = Path(row["xml"])
        if not xml.is_absolute() and not xml.exists():
            xml = paired_dir / "review" / xml.name
        page = int(row["page"])
        used.update(k for k in decisions if k[0] == row["doc_id"] and k[1] == page)
        if skip_existing and (reviewed_dir / xml.name).exists():
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "page": page,
                    "kept": 0,
                    "fixed": 0,
                    "dropped": 0,
                    "skipped": True,
                }
            )
            continue
        counts = apply_to_page(
            xml, reviewed_dir / xml.name, row["doc_id"], page, decisions, min_conf, note
        )
        rows.append({"doc_id": row["doc_id"], "page": page, **counts, "skipped": False})
    result = pd.DataFrame(
        rows, columns=["doc_id", "page", "kept", "fixed", "dropped", "skipped"]
    )
    result.attrs["unused"] = sorted(set(decisions) - used)
    return result


__all__: Iterable[str] = [
    "Decision",
    "read_decisions",
    "apply_to_page",
    "apply_review",
    "main_lines",
]
