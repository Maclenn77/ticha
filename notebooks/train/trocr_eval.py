"""Scoring shared by the TrOCR training notebooks.

- ``decode_split``: read every line of a split with beam search.
- ``evaluate``: CER and WER of ``{line id: text}`` predictions, overall and by
  language.
- ``paired_bootstrap``: is the CER difference between two systems real?

CER and WER are computed over all lines at once (total edits / total reference
length), like ``jiwer``. Empty predictions are scored as a single space, so a
line read as nothing counts as fully wrong.
"""

from typing import Callable, Dict, Mapping, Optional

import numpy as np
from jiwer import cer, process_characters, wer
from trocr_setup import SEED

Predictions = Mapping[str, str]


def decode_split(
    model,
    processor,
    split,
    num_beams: int,
    max_length: int,
    batch_size: int,
    use_bf16: bool,
    device: str = "cuda",
) -> Dict[str, str]:
    """``{line id: text}`` for every line of ``split``, decoded with beam search."""
    import torch

    out, ids = {}, split["id"]
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            rows = split.select(range(start, min(start + batch_size, len(split))))
            images = [im.convert("RGB") for im in rows["image"]]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                seqs = model.generate(
                    pixel_values=pixel_values.to(device),
                    num_beams=num_beams,
                    max_length=max_length,
                    early_stopping=True,
                )
            texts = processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)
            for k, text in enumerate(texts):
                out[ids[start + k]] = text.strip()
    return out


def evaluate(
    references: Mapping[str, str],
    predictions: Predictions,
    languages: Optional[Mapping[str, str]] = None,
    keep: Optional[Callable[[str], bool]] = None,
) -> Dict[str, float]:
    """``lines``, ``CER``, ``WER`` and ``CER <language>`` of the predictions.

    ``keep`` limits the score to some line ids (e.g. the lines of the loaded
    split, when a cached prediction file covers more)."""
    ids = [i for i in predictions if keep is None or keep(i)]
    refs = [references[i] for i in ids]
    hyps = [predictions[i] or " " for i in ids]
    row = {"lines": len(ids), "CER": cer(refs, hyps), "WER": wer(refs, hyps)}
    for lang in sorted({languages[i] for i in ids} if languages else []):
        idx = [n for n, i in enumerate(ids) if languages[i] == lang]
        row[f"CER {lang}"] = cer([refs[n] for n in idx], [hyps[n] for n in idx])
    return row


def paired_bootstrap(
    references: Mapping[str, str],
    base: Predictions,
    new: Predictions,
    keep: Optional[Callable[[str], bool]] = None,
    n: int = 5000,
    seed: int = SEED,
) -> Dict[str, float]:
    """CER difference ``new - base`` on the same lines, with a 95% interval.

    Whole documents are resampled (line ids start with the document id), since
    lines of one document share a scribe and are not independent. Returns the
    difference, the interval (``low``, ``high``) and the share of resamples in
    which ``new`` is better. A difference counts when its interval excludes 0.
    """
    ids = sorted(i for i in base if keep is None or keep(i))

    def edits(preds: Predictions) -> np.ndarray:
        out = []
        for i in ids:
            o = process_characters(references[i], preds[i] or " ")
            out.append(o.substitutions + o.deletions + o.insertions)
        return np.array(out)

    e_base, e_new = edits(base), edits(new)
    lengths = np.array([len(references[i]) for i in ids])
    docs = np.array([i.split("_")[0] for i in ids])
    groups = [np.where(docs == d)[0] for d in np.unique(docs)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = np.concatenate(
            [groups[g] for g in rng.integers(len(groups), size=len(groups))]
        )
        diffs.append((e_new[s].sum() - e_base[s].sum()) / lengths[s].sum())
    diffs = np.array(diffs)
    return {
        "difference": (e_new.sum() - e_base.sum()) / lengths.sum(),
        "low": np.percentile(diffs, 2.5),
        "high": np.percentile(diffs, 97.5),
        "share better": float(np.mean(diffs < 0)),
    }
