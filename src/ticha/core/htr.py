"""Handwritten text recognition with the fine-tuned TrOCR models.

The best system reads a line in two steps:

1. The left-to-right model (TrOCR base fine-tuned on the Ticha lines) proposes
   its 10 best readings with beam search.
2. The right-to-left model, trained to write each line's tokens in reverse,
   scores the same readings backwards. Each candidate gets

       ((1 - w) * log p_left-to-right + w * log p_right-to-left) / (tokens + 2)

   and the best one is the reading. ``w = 0.3`` was chosen on validation.

Both models are on the Hub in ``maclenn77/ticha-trocr-base`` (the right-to-left
one in ``r2l/``). They were trained with the decoder start token ``0``, which
``configure`` sets explicitly instead of trusting a saved configuration.

``clean_text`` is the transcription rule used to train and evaluate them: it
turns a Ticha transcription into diplomatic text (what is on the page) or drops
the line when its image and text don't match.

Only ``LineReader`` needs ``torch`` and ``transformers``; they are imported when
a model is loaded, so the text functions work without them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    from PIL import Image

REPO_ID = "maclenn77/ticha-trocr-base"
START = 0  # decoder start token the fine-tuned models were trained with
MAX_LENGTH = 64
BEAMS = 10
RERANK_WEIGHT = 0.3  # chosen on validation for 10 beams

# Transcriber notes: [] and () mark expansions, glosses and damage.
BRACKET_RE = re.compile(r"\[([^\[\]]*)\]|\(([^()]*)\)")
NOTE_RE = re.compile(
    r"[?…]|^\.+$|\.\.|\s|,|^x+$|^word$"  # uncertain, illegible, or a note with spaces
    r"|cortado|roto|torn|mancha|borrado|firmas|ilegible",  # damage and editorial notes
    re.IGNORECASE,
)
GUESS_RE = re.compile(
    r"\((?!word\?)([^()?]*[^\W\d_][^()?]*)\?+\)"
)  # (letters?), not (word?)


def clean_text(text: Optional[str], keep_guesses: bool = False) -> Optional[str]:
    """Diplomatic text of a Ticha transcription, or None when the line is dropped.

    - Expansions are deleted: ``p[es]os`` -> ``pos``, ``Dom(ing)o`` -> ``Domo``.
    - Glosses are deleted: ``[=Juan]``.
    - Damage, illegible text and editorial notes (``[cortado]``, ``[...]``,
      ``(?)``, ``(word?)``, ``#``) and unbalanced brackets drop the line.
    - Uncertain readings such as ``(Santigo?)`` drop the line, or become
      ``Santigo`` with ``keep_guesses=True`` (used for the train split).
    """
    text = str(text or "")
    if keep_guesses:
        text = GUESS_RE.sub(lambda m: m.group(1), text)
    notes = [square or round_ for square, round_ in BRACKET_RE.findall(text)]
    if any(NOTE_RE.search(n.strip()) for n in notes if not n.strip().startswith("=")):
        return None
    text = BRACKET_RE.sub("", text)
    if re.search(r"[\[\]()]", text) or "#" in text:
        return None
    return re.sub(r"\s+", " ", text).strip() or None


def inner_tokens(
    ids: Sequence[int], start: int, bos: int, eos: int, pad: int
) -> list[int]:
    """The tokens between the leading start/<s> tokens and the first </s>."""
    ids = [int(i) for i in ids]
    while ids and ids[0] in (start, bos):
        ids = ids[1:]
    return ids[: ids.index(eos)] if eos in ids else [i for i in ids if i != pad]


@dataclass
class Candidate:
    """One reading of a line, with its summed log-probabilities."""

    text: str
    tokens: list[int]
    l2r: float
    r2l: Optional[float] = None

    def score(self, w: float = RERANK_WEIGHT) -> float:
        """Combined score, normalized by length (<s> and </s> included)."""
        r2l = self.l2r if self.r2l is None else self.r2l
        return ((1 - w) * self.l2r + w * r2l) / (len(self.tokens) + 2)


def best_candidate(
    candidates: Sequence[Candidate], w: float = RERANK_WEIGHT
) -> Candidate:
    """The candidate with the best combined score (the first one on a tie)."""
    return max(candidates, key=lambda c: c.score(w))


ImageLike = Union["Image.Image", str, Path]


class LineReader:
    """Reads line images with the left-to-right model and, optionally, reranks
    its beams with the right-to-left model.

    >>> reader = LineReader.from_pretrained()          # doctest: +SKIP
    >>> reader.transcribe("line.png")                  # doctest: +SKIP
    'escritura de uenta'
    """

    def __init__(self, model, processor, r2l=None, device=None):
        import torch

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer
        tok = self.tokenizer
        self.bos, self.pad, self.eos = (
            tok.cls_token_id,
            tok.pad_token_id,
            tok.sep_token_id,
        )
        self.model = self.configure(model).to(self.device).eval()
        self.r2l = (
            self.configure(r2l).to(self.device).eval() if r2l is not None else None
        )

    def configure(self, model):
        """Set the decoding settings the models were trained with."""
        for cfg in (model.config, model.generation_config):
            cfg.decoder_start_token_id = START
            cfg.pad_token_id, cfg.eos_token_id = self.pad, self.eos
        model.config.vocab_size = model.config.decoder.vocab_size
        model.generation_config.max_length = MAX_LENGTH
        model.generation_config.early_stopping = True
        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = REPO_ID,
        rerank: bool = True,
        device: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        """Load from the Hub or a local folder (``r2l/`` holds the reranker)."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(repo_id, revision=revision)
        model = VisionEncoderDecoderModel.from_pretrained(repo_id, revision=revision)
        r2l = (
            VisionEncoderDecoderModel.from_pretrained(
                repo_id, subfolder="r2l", revision=revision
            )
            if rerank
            else None
        )
        return cls(model, processor, r2l=r2l, device=device)

    def pixel_values(self, images: Sequence[ImageLike]):
        from PIL import Image

        rgb = [
            (im if isinstance(im, Image.Image) else Image.open(im)).convert("RGB")
            for im in images
        ]
        return self.processor(images=rgb, return_tensors="pt").pixel_values.to(
            self.device
        )

    def sequence_log_probs(
        self, model, pixel_values, label_lists: Sequence[Sequence[int]]
    ) -> list[float]:
        """Teacher-forced summed log-probability of each label sequence (one image row per sequence)."""
        import torch
        from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (
            shift_tokens_right,
        )

        width = max(map(len, label_lists))
        labels = torch.full((len(label_lists), width), -100, dtype=torch.long)
        for row, ids in enumerate(label_lists):
            labels[row, : len(ids)] = torch.tensor(list(ids))
        labels = labels.to(pixel_values.device)
        with torch.inference_mode():
            logits = model(
                pixel_values=pixel_values,
                decoder_input_ids=shift_tokens_right(labels, self.pad, START),
            ).logits
        logp = (
            logits.float()
            .log_softmax(-1)
            .gather(-1, labels.clamp(min=0).unsqueeze(-1))
            .squeeze(-1)
        )
        return (logp * (labels != -100)).sum(-1).tolist()

    def candidates(
        self, images: Sequence[ImageLike], beams: int = BEAMS
    ) -> list[list[Candidate]]:
        """The ``beams`` best readings of each image, best first, with their
        left-to-right scores and, when the reranker is loaded, right-to-left ones."""
        import torch

        pixel_values = self.pixel_values(images)
        with torch.inference_mode():
            sequences = self.model.generate(
                pixel_values,
                num_beams=beams,
                num_return_sequences=beams,
                max_length=MAX_LENGTH,
                early_stopping=True,
            )
        inners = [
            inner_tokens(s, START, self.bos, self.eos, self.pad)
            for s in sequences.tolist()
        ]
        repeated = pixel_values.repeat_interleave(beams, 0)
        l2r = self.sequence_log_probs(
            self.model, repeated, [[self.bos, *i, self.eos] for i in inners]
        )
        r2l = (
            self.sequence_log_probs(
                self.r2l, repeated, [[self.bos, *i[::-1], self.eos] for i in inners]
            )
            if self.r2l is not None
            else [None] * len(inners)
        )
        out = []
        for line in range(len(images)):
            rows = range(line * beams, (line + 1) * beams)
            out.append(
                [
                    Candidate(
                        self.tokenizer.decode(
                            inners[k], skip_special_tokens=True
                        ).strip(),
                        inners[k],
                        l2r[k],
                        r2l[k],
                    )
                    for k in rows
                ]
            )
        return out

    def transcribe(
        self,
        images: Union[ImageLike, Sequence[ImageLike]],
        rerank: bool = True,
        beams: int = BEAMS,
        w: float = RERANK_WEIGHT,
        batch_size: int = 8,
    ):
        """Text of one line image, or a list of texts for a list of images.

        With ``rerank`` (and the reranker loaded) the beams are rescored with the
        right-to-left model; otherwise the first beam is the reading.
        """
        single = not isinstance(images, (list, tuple))
        images = [images] if single else list(images)
        rerank = rerank and self.r2l is not None
        texts = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            if rerank:
                texts += [
                    best_candidate(c, w).text for c in self.candidates(batch, beams)
                ]
            else:
                import torch

                with torch.inference_mode():
                    seqs = self.model.generate(
                        self.pixel_values(batch),
                        num_beams=beams,
                        max_length=MAX_LENGTH,
                        early_stopping=True,
                    )
                texts += [
                    t.strip()
                    for t in self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                ]
        return texts[0] if single else texts
