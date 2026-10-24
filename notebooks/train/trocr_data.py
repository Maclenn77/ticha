"""Loading, checking and cleaning the line dataset for the TrOCR notebooks.

Every notebook trains and scores on the same lines:

1. ``load_clean_splits`` loads the ``original`` config (train, validation,
   test) and applies ``ticha.core.htr.clean_text``: expansions and glosses are
   deleted, lines with damage, illegible text or editorial notes are dropped.
   Uncertain readings such as ``(Santigo?)`` are kept as text in train and
   dropped from validation and test, so scores count only confident
   transcriptions. The original text stays in ``text_raw``.
2. ``check_document_overlap`` stops if a document is in two splits.
3. ``duplicate_train_docs`` finds train documents that repeat validation or test
   lines (Ticha catalogues some manuscripts twice).
4. ``load_train_aug`` adds the augmented copies of the train lines, cleaned the
   same way, and ``load_test_degraded`` loads the degraded copy of the test lines.

The text is cleaned in memory and lines are kept by index: ``map``/``filter``
would write a full copy of the images to the datasets cache on every run.
"""

import re
from typing import Iterable, Optional, Set, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from trocr_setup import DATASET_ID, DATASET_REVISION, SEED

from ticha.core.htr import clean_text

SPLITS = ("train", "validation", "test")


def clean_split(split: Dataset, keep_guesses: bool) -> Tuple[Dataset, int, int]:
    """Clean the text of a split; returns (kept lines, lines before, lines changed)."""
    before = len(split)
    raw = split["text"]
    cleaned = [clean_text(t, keep_guesses=keep_guesses) for t in raw]
    split = split.remove_columns("text").add_column("text", cleaned)
    if "text_raw" not in split.column_names:
        split = split.add_column("text_raw", raw)
    kept = [i for i, t in enumerate(cleaned) if t is not None]
    changed = sum(1 for i in kept if cleaned[i] != raw[i])
    return split.select(kept), before, changed


def load_clean_splits(
    revision: str = DATASET_REVISION, keep_guesses_in_train: bool = True
) -> DatasetDict:
    """Train, validation and test of the ``original`` config, cleaned."""
    ds = load_dataset(DATASET_ID, "original", revision=revision)
    for name in SPLITS:
        keep_guesses = keep_guesses_in_train and name == "train"
        ds[name], before, changed = clean_split(ds[name], keep_guesses)
        print(
            f"{name}: kept {len(ds[name]):,} / {before:,} lines "
            f"(dropped {before - len(ds[name]):,}, cleaned text in {changed:,})"
        )
    return ds


def check_document_overlap(ds: DatasetDict) -> None:
    """Stop if a document is shared between splits (its hand would leak into
    evaluation) or if a line has no document id."""
    docs = {name: set(ds[name]["doc_id"]) for name in SPLITS}
    missing = {name: sum(d is None for d in ds[name]["doc_id"]) for name in SPLITS}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    shared = {f"{a}-{b}": docs[a] & docs[b] - {None} for a, b in pairs}
    for pair, found in shared.items():
        print(f"{pair}: {len(found)} shared documents", sorted(found)[:10] or "")
    if any(shared.values()) or any(missing.values()):
        raise RuntimeError(
            f"Fix the document-level splits first (rows without doc_id: {missing})"
        )
    print("No document is shared between splits.")


def line_key(text: str) -> str:
    """A line's text lowercased, with punctuation and spaces collapsed."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def duplicate_train_docs(
    ds: DatasetDict, share: float = 0.5, min_chars: int = 15
) -> Set[str]:
    """Train documents more than ``share`` of whose lines appear verbatim in
    validation or test (lines of at least ``min_chars`` characters)."""
    eval_keys = {
        line_key(text)
        for name in ("validation", "test")
        for text in ds[name]["text"]
        if len(text) >= min_chars
    }
    hits = {}
    for doc_id, text in zip(ds["train"]["doc_id"], ds["train"]["text"], strict=True):
        hits.setdefault(doc_id, []).append(line_key(text) in eval_keys)
    return {doc for doc, found in hits.items() if sum(found) > share * len(found)}


def translation_pair_docs(ds: DatasetDict) -> Set[str]:
    """Train documents whose translation (or original) is in validation or test:
    ``Al710`` and ``Al710T`` are different manuscripts in different languages,
    but share names, places and dates."""

    def base_id(doc_id: str) -> str:
        return doc_id[:-1] if doc_id.endswith("T") else doc_id

    eval_bases = {
        base_id(d) for name in ("validation", "test") for d in ds[name]["doc_id"]
    }
    return {d for d in set(ds["train"]["doc_id"]) if base_id(d) in eval_bases}


def drop_train_docs(ds: DatasetDict, docs: Iterable[str]) -> DatasetDict:
    """Remove the lines of ``docs`` from train."""
    docs = set(docs)
    before = len(ds["train"])
    keep = [i for i, d in enumerate(ds["train"]["doc_id"]) if d not in docs]
    ds["train"] = ds["train"].select(keep)
    print(f"train: kept {len(ds['train']):,} / {before:,} lines")
    return ds


def load_train_aug(
    train: Dataset,
    revision: str = DATASET_REVISION,
    seed: int = SEED,
    columns: Tuple[str, ...] = ("image", "text", "doc_id"),
) -> Dataset:
    """The original train lines plus their augmented copies (cleaned like train,
    uncertain readings kept), shuffled with ``seed``."""
    aug = load_dataset(DATASET_ID, "augmented", revision=revision)["train"]
    aug, before, _ = clean_split(aug, keep_guesses=True)
    original = train.select_columns(list(columns))
    aug = aug.select_columns(list(columns))
    if aug.features != original.features:
        aug = aug.cast(original.features)
    train_aug = concatenate_datasets([original, aug]).shuffle(seed=seed)
    print(
        f"augmented train: kept {len(aug):,} / {before:,} lines -> "
        f"{len(train_aug):,} training lines with the originals "
        f"({len(train_aug) / len(train):.1f}x the original train set)"
    )
    return train_aug


def load_test_degraded(test: Dataset, revision: str = DATASET_REVISION) -> Dataset:
    """The degraded copy of the test lines that survived cleaning (same lines)."""
    degraded = load_dataset(DATASET_ID, "augmented", revision=revision)
    degraded, _, _ = clean_split(degraded["test_degraded"], keep_guesses=False)
    test_ids = set(test["id"])
    keep = [i for i, s in enumerate(degraded["source_id"]) if s in test_ids]
    degraded = degraded.select(keep)
    print(f"test_degraded: {len(degraded):,} lines")
    return degraded


def fixed_sample(split: Dataset, size: int, seed: Optional[int] = SEED) -> Dataset:
    """A fixed random sample of ``size`` lines of ``split`` (e.g. train lines to
    score training CER during training)."""
    return split.shuffle(seed=seed).select(range(size))
