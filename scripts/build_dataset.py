"""Build the line dataset (steps 6-10 of notebooks/ticha_pipeline.ipynb).

Reads the output of ``ticha-scraper lines`` (``data/paired``) and the pages
corrected by hand (``data/reviewed``), and writes two separate datasets:

- ``data/splits/``: the original lines, split by document into train /
  validation / test (80/10/10 of the lines within each language).
  ``splits/<split>/auto/`` has the final pages (image + PAGE XML with only the
  usable text), ``splits/<split>/lines/`` the line crops + ``.gt.txt``, and
  ``splits/<split>.txt`` the list of line images (``ketos train -f path``).
- ``data/augmented/``: augmented copies, never mixed into ``splits/``:
  ``augmented/train/`` (``--copies`` per train line, ``degrade`` and
  ``variety`` presets) and ``augmented/test_degraded/`` (one ``degrade``
  copy per test line, no ``variety`` and no geometry) and
  ``augmented/validation_degraded/`` (the same for validation lines), to measure
  how much worse the model reads low-quality scans. They are separate sets,
  never mixed with the clean ones. ``--only-validation-degraded`` adds the
  validation set to an existing build without touching anything else.

Usage (from the repository root):

    python scripts/build_dataset.py [--data data] [--copies 4] [--degrade-ratio 0.5]
"""

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ticha.core.augmentation import Augmenter, DatasetAugmenter  # noqa: E402
from ticha.core.line_pairing import crop_line  # noqa: E402

from dataset_exclusions import DATASET_VERSION, EXCLUDED_DOCS  # noqa: E402

SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(el, name):
    return [c for c in el if local(c.tag) == name]


def read_page_lines(xml_path: Path) -> list:
    """Lines of a PAGE XML in document order: boundary, text and confidence."""
    lines = []
    for el in ET.parse(xml_path).getroot().iter():
        if local(el.tag) != "TextLine":
            continue
        coords = children(el, "Coords")
        if not coords or not coords[0].get("points"):
            continue
        boundary = [[int(float(v)) for v in p.split(",")] for p in coords[0].get("points").split()]
        text, conf = None, None
        for equiv in children(el, "TextEquiv"):
            unicode_ = children(equiv, "Unicode")
            if unicode_ and (unicode_[0].text or "").strip():
                text = unicode_[0].text.strip()
                conf = float(equiv.get("conf")) if equiv.get("conf") else None
                break
        lines.append({"id": el.get("id"), "boundary": boundary, "text": text, "conf": conf})
    return lines


def usable(line: dict, source: str, min_conf: float) -> bool:
    # Reviewed text is always trusted; otherwise the pair must be confident.
    if not line["text"]:
        return False
    return source == "reviewed" or line["conf"] is None or line["conf"] >= min_conf


def page_image(xml_path: Path) -> Path:
    found = [p for p in xml_path.parent.glob(xml_path.stem + ".*") if p.suffix.lower() in IMAGE_EXTS]
    if not found:
        raise FileNotFoundError(f"No image for {xml_path}")
    return found[0]


def write_clean_xml(source: Path, target: Path, image_name: str, kind: str, min_conf: float) -> None:
    """Copy a PAGE XML pointing to image_name, keeping text only on usable lines."""
    tree = ET.parse(source)
    root = tree.getroot()
    ET.register_namespace("", root.tag[1:].split("}")[0])
    for el in root.iter():
        if local(el.tag) == "Page":
            el.set("imageFilename", image_name)
    for el in list(root.iter()):
        if local(el.tag) != "TextLine":
            continue
        line = {"text": None, "conf": None}
        for equiv in children(el, "TextEquiv"):
            unicode_ = children(equiv, "Unicode")
            if unicode_ and (unicode_[0].text or "").strip():
                line.update(text=unicode_[0].text.strip(),
                            conf=float(equiv.get("conf")) if equiv.get("conf") else None)
                break
        keep = usable(line, kind, min_conf)
        for equiv in children(el, "TextEquiv"):
            if not keep:
                el.remove(equiv)
            elif kind == "reviewed":
                equiv.attrib.pop("conf", None)
    tree.write(target, encoding="utf-8", xml_declaration=True)


def link_or_copy(source: Path, target: Path) -> None:
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def final_pages(data: Path, include_unreviewed: bool, min_conf: float) -> pd.DataFrame:
    """Step 6: the XML to use for each page (auto, reviewed or pending)."""
    paired, reviewed = data / "paired", data / "reviewed"
    manifest = pd.read_csv(paired / "manifest.csv")
    texts = pd.read_csv(data / "texts_transcribed.csv")
    language_of = texts.set_index("ticha_id")["language"].to_dict()
    rows = []
    for _, row in manifest[manifest["status"].isin(["auto", "review"])].iterrows():
        xml = Path(row["xml"])
        if not xml.is_absolute():
            xml = ROOT / xml
        image = page_image(xml)
        if row["status"] == "auto":
            kind = "auto"
        elif (reviewed / xml.name).exists():
            kind, xml = "reviewed", reviewed / xml.name
        elif include_unreviewed:
            kind = "unreviewed"
        else:
            kind = "pending review"
        n = 0
        if kind != "pending review":
            n = sum(usable(ln, kind, min_conf) and len(ln["boundary"]) >= 3 for ln in read_page_lines(xml))
        rows.append({"doc_id": row["doc_id"], "page": int(row["page"]), "stem": xml.stem, "source": kind,
                     "language": language_of.get(row["doc_id"], "(unknown)"),
                     "xml": str(xml), "image": str(image), "usable_lines": n})
    return pd.DataFrame(rows)


def split_documents(pages: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Step 7: documents to subsets, balancing lines within each language."""
    table = pages.groupby("doc_id").agg(language=("language", "first"), pages=("page", "size"),
                                        lines=("usable_lines", "sum"))
    rng = random.Random(seed)
    assignment = {}
    for _, group in table.groupby("language"):
        ids = sorted(group.index)
        rng.shuffle(ids)
        filled = {s: 0 for s in SPLIT_RATIOS}
        for doc_id in ids:
            split = min(SPLIT_RATIOS, key=lambda s: (filled[s] / SPLIT_RATIOS[s], -SPLIT_RATIOS[s]))
            assignment[doc_id] = split
            filled[split] += group.loc[doc_id, "lines"]
    table["split"] = table.index.map(assignment)
    return table


def cut_lines(pages: pd.DataFrame, table: pd.DataFrame, splits_dir: Path, min_conf: float) -> pd.DataFrame:
    """Step 8: final pages and line crops of each subset."""
    rows = []
    for split in SPLIT_RATIOS:
        shutil.rmtree(splits_dir / split, ignore_errors=True)
    for _, page in pages.iterrows():
        split = table.loc[page["doc_id"], "split"]
        out = splits_dir / split
        image_src, xml_src = Path(page["image"]), Path(page["xml"])
        (out / "auto").mkdir(parents=True, exist_ok=True)
        image_name = f"{page['stem']}{image_src.suffix}"
        link_or_copy(image_src, out / "auto" / image_name)
        write_clean_xml(xml_src, out / "auto" / f"{page['stem']}.xml", image_name, page["source"], min_conf)
        crops = out / "lines" / page["stem"]
        crops.mkdir(parents=True, exist_ok=True)
        with Image.open(image_src) as im:
            number = 0
            for line in read_page_lines(xml_src):
                if not (usable(line, page["source"], min_conf) and len(line["boundary"]) >= 3):
                    continue
                number += 1
                crop = crops / f"line_{number:03d}.png"
                crop_line(im, line["boundary"]).save(crop)
                crop.with_name(crop.stem + ".gt.txt").write_text(line["text"] + "\n", encoding="utf-8")
                rows.append({"split": split, "doc_id": page["doc_id"], "language": page["language"],
                             "page": page["page"], "source": page["source"], "line": number,
                             "text": line["text"], "conf": line["conf"], "image": str(crop)})
    lines = pd.DataFrame(rows)
    for split, group in lines.groupby("split"):
        group.to_csv(splits_dir / split / "lines.csv", index=False)
    return lines


def write_list(files, path: Path) -> int:
    files = sorted(str(p) for p in files)
    path.write_text("\n".join(files) + "\n", encoding="utf-8")
    return len(files)


def line_images(folder: Path):
    return (p for p in folder.rglob("*.png") if p.with_name(p.stem + ".gt.txt").exists())


def augment_job(splits_dir: Path, aug_dir: Path, seed: int, min_conf: float, job) -> int:
    """Augment one split into ``aug_dir/<name>`` and list its line images in ``aug_dir/<name>.txt``."""
    split, name, copies, ratio, geometry, offset = job
    augmenter = DatasetAugmenter(Augmenter(seed=seed + offset), copies=copies, degrade_ratio=ratio,
                                 min_conf=min_conf, include_review=True, geometry=geometry)
    augmenter.run(splits_dir / split, aug_dir / name)
    count = write_list(line_images(aug_dir / name / "lines"), aug_dir / f"{name}.txt")
    print(f"{name}: {count} augmented lines ({copies} per {split} line, "
          f"{round(100 * ratio)}% degrade{'' if geometry else ', no geometry'})")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Use confident lines of review pages without a reviewed XML")
    parser.add_argument("--copies", type=int, default=4, help="Augmented copies per train line")
    parser.add_argument("--degrade-ratio", type=float, default=0.5,
                        help="Share of train copies using the degrade preset (rest: variety)")
    parser.add_argument("--test-degraded-copies", type=int, default=1,
                        help="Degraded copies per test line (0: no degraded test set)")
    parser.add_argument("--validation-degraded-copies", type=int, default=1,
                        help="Degraded copies per validation line (0: no degraded validation set)")
    parser.add_argument("--only-validation-degraded", action="store_true",
                        help="Only (re)build augmented/validation_degraded from the existing splits/validation, "
                             "without cutting the splits again or touching the other augmented sets")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    data = args.data.resolve()
    splits_dir, aug_dir, reports = data / "splits", data / "augmented", data / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    if args.only_validation_degraded:
        if not list((splits_dir / "validation" / "lines").rglob("*.gt.txt")):
            raise SystemExit("splits/validation/lines is empty or missing: run the full build first")
        if not args.validation_degraded_copies:
            raise SystemExit("--only-validation-degraded needs --validation-degraded-copies >= 1")
        shutil.rmtree(aug_dir / "validation_degraded", ignore_errors=True)
        augment_job(splits_dir, aug_dir, args.seed, args.min_conf,
                    ("validation", "validation_degraded", args.validation_degraded_copies, 1.0, False, 2))
        return

    pages = final_pages(data, args.include_unreviewed, args.min_conf)
    pages.to_csv(reports / "final_pages.csv", index=False)
    pending = pages[pages["source"] == "pending review"]
    pages = pages[(pages["source"] != "pending review") & (pages["usable_lines"] > 0)]
    print(f"Pages used: {len(pages)} ({int(pages['usable_lines'].sum())} lines); "
          f"by source: {pages['source'].value_counts().to_dict()}; pending review: {len(pending)}")

    table = split_documents(pages, args.seed)
    # Excluded after the split, so every other document keeps the split it had before.
    excluded = [doc_id for doc_id in EXCLUDED_DOCS if doc_id in table.index]
    table = table.drop(index=excluded)
    pages = pages[~pages["doc_id"].isin(excluded)]
    if excluded:
        print(f"Excluded documents: {excluded}")
    splits_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(splits_dir / "splits.csv")
    lines = cut_lines(pages, table, splits_dir, args.min_conf)
    for split in SPLIT_RATIOS:
        write_list(line_images(splits_dir / split / "lines"), splits_dir / f"{split}.txt")

    aug_counts = {}
    if not args.no_augment:
        shutil.rmtree(aug_dir, ignore_errors=True)
        # The last element is the seed offset, fixed per set so that adding or removing
        # one set never changes the images of another.
        jobs = [("train", "train", args.copies, args.degrade_ratio, True, 0)]
        if args.test_degraded_copies:
            jobs.append(("test", "test_degraded", args.test_degraded_copies, 1.0, False, 1))
        if args.validation_degraded_copies:
            jobs.append(("validation", "validation_degraded", args.validation_degraded_copies, 1.0, False, 2))
        for job in jobs:
            aug_counts[job[1]] = augment_job(splits_dir, aug_dir, args.seed, args.min_conf, job)

    lines["chars"] = lines["text"].str.len()
    metrics = pd.DataFrame({
        "documents": lines.groupby("split")["doc_id"].nunique(),
        "pages": lines.groupby("split")[["doc_id", "page"]].apply(lambda g: len(g.drop_duplicates())),
        "lines": lines.groupby("split").size(),
        "characters": lines.groupby("split")["chars"].sum(),
        "lines from reviewed pages": lines[lines["source"] == "reviewed"].groupby("split").size(),
    }).reindex(SPLIT_RATIOS).fillna(0).astype(int)
    metrics["% lines"] = (100 * metrics["lines"] / metrics["lines"].sum()).round(1)
    metrics.to_csv(reports / "final_metrics.csv")
    print(metrics.to_string())
    print("Lines per language:")
    print(lines.pivot_table(index="language", columns="split", values="line", aggfunc="size",
                            fill_value=0).reindex(columns=list(SPLIT_RATIOS), fill_value=0).to_string())
    train_chars = set("".join(lines.loc[lines["split"] == "train", "text"]))
    for split in ("validation", "test"):
        unseen = set("".join(lines.loc[lines["split"] == split, "text"])) - train_chars
        print(f"Characters in {split} missing from train: {''.join(sorted(unseen)) or '(none)'}")

    summary = {
        "pages": int(len(pages)), "pages pending review": int(len(pending)),
        "lines": {s: int(metrics.loc[s, "lines"]) for s in SPLIT_RATIOS},
        "documents": {s: int(metrics.loc[s, "documents"]) for s in SPLIT_RATIOS},
        "augmented lines": aug_counts,
        "dataset version": DATASET_VERSION,
        "excluded documents": {doc_id: EXCLUDED_DOCS[doc_id] for doc_id in excluded},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (reports / "final_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
