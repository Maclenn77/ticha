"""Model pieces shared by the TrOCR training notebooks.

Nothing here sets a training hyperparameter: the notebooks pass the target
length and the number of beams, and keep their ``Seq2SeqTrainingArguments``
(learning rate, epochs, smoothing, schedule, batch size) in the notebook.
"""

from typing import Sequence, Tuple

import numpy as np
from transformers import VisionEncoderDecoderModel
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (
    shift_tokens_right,
)
from trocr_setup import START


def max_target_length(
    tokenizer,
    texts: Sequence[str],
    floor: int = 64,
    ceiling: int = 256,
    margin: int = 8,
) -> Tuple[int, int, int]:
    """Label length from the train texts: the 99th percentile of their token
    counts plus ``margin``, between ``floor`` and ``ceiling``.

    Returns (length, 99th percentile, train lines longer than the length)."""
    lengths = [
        len(tokenizer(str(t or "").strip(), add_special_tokens=True).input_ids)
        for t in texts
    ]
    p99 = int(np.percentile(lengths, 99))
    length = min(ceiling, max(floor, p99 + margin))
    return length, p99, sum(n > length for n in lengths)


class TrOCRDataCollator:
    """Builds each batch when it is needed, so the image tensors of the whole
    dataset are never held in memory: pixel values from the images, token ids
    from the texts, padding set to -100 so the loss ignores it."""

    def __init__(self, processor, max_target_length: int):
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, batch):
        images = [item["image"].convert("RGB") for item in batch]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        texts = [str(item["text"] or "").strip() for item in batch]
        labels = self.processor.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


class TrOCRSmoothingCollator(TrOCRDataCollator):
    """With label smoothing the Trainer removes ``labels`` before calling the
    model, so the model can no longer build its decoder inputs from them: add
    them to the batch here."""

    def __call__(self, batch):
        out = super().__call__(batch)
        tokenizer = self.processor.tokenizer
        out["decoder_input_ids"] = shift_tokens_right(
            out["labels"], tokenizer.pad_token_id, tokenizer.cls_token_id
        )
        return out


def prepare_for_training(model, tokenizer, max_length: int, num_beams: int):
    """Set the special tokens and the decoding used to score validation during
    training (the start token in ``model.config``)."""
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.max_length = max_length
    model.generation_config.num_beams = num_beams
    model.generation_config.early_stopping = True
    return model


def pretrained_model(model_id: str, tokenizer, max_length: int, num_beams: int):
    """A fresh copy of the pretrained checkpoint, ready to fine-tune."""
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    return prepare_for_training(model, tokenizer, max_length, num_beams)


def configure_for_decoding(model, tokenizer, max_length: int, start: int = START):
    """Decode with the start token the models were trained with, set in both
    ``config`` and ``generation_config`` (the saved checkpoints say ``2``)."""
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = start
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.max_length = max_length
    model.generation_config.early_stopping = True
    return model


def cer_wer_metrics(processor):
    """``compute_metrics`` for the Trainer: CER and WER of the decoded validation
    predictions against the labels."""
    from jiwer import cer, wer

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = np.array(eval_pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred = [
            t.strip()
            for t in processor.batch_decode(pred_ids, skip_special_tokens=True)
        ]
        ref = [
            t.strip()
            for t in processor.batch_decode(label_ids, skip_special_tokens=True)
        ]
        return {"cer": cer(ref, pred), "wer": wer(ref, pred)}

    return compute_metrics
