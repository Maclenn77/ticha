"""Tests for line recognition: transcription cleaning, candidate scoring, and
LineReader on a tiny random TrOCR model (no network, no downloaded weights)."""

import pytest
from PIL import Image

from ticha.core.htr import (
    START,
    Candidate,
    LineReader,
    best_candidate,
    clean_text,
    inner_tokens,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3 p[es]os ciivixo", "3 pos ciivixo"),
        ("Alc[alde]s pasq[ua]l", "Alcs pasql"),
        ("Dom(ing) o garola", "Dom o garola"),
        ("[=Juan] Pedro", "Pedro"),
        ("nique = che", "nique = che"),
        ("x ...", "x ..."),
        ("guee [cortado]", None),
        ("xia[cortado]", None),
        ("ri[..]nan", None),
        ("cava[n?]a", None),
        ("a [.] b", None),
        ("(word?) x", None),
        ("(?) x", None),
        ("ue[tto p[edr", None),
        ("huara#", None),
        ("[margin note] x", None),
        ("", None),
        (None, None),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Felíphe (Santigo?)", "Felíphe Santigo"),
        ("(Y??)a naca", "Ya naca"),
        ("ni (J[usticia]?) mayor", "ni J mayor"),
        ("Anos (word?)", None),
        ("x (?) y", None),
        ("cava[n?]a", None),  # only () guesses are kept
    ],
)
def test_clean_text_keep_guesses(raw, expected):
    assert clean_text(raw, keep_guesses=True) == expected
    if expected is not None:
        assert clean_text(raw) is None  # dropped from validation and test


def test_inner_tokens():
    start, bos, eos, pad = 0, 0, 2, 1
    assert inner_tokens([0, 0, 5, 6, 7, 2, 1, 1], start, bos, eos, pad) == [5, 6, 7]
    assert inner_tokens([0, 5, 6], start, bos, eos, pad) == [5, 6]
    assert inner_tokens([0, 5, 1, 1], start, bos, eos, pad) == [5]
    assert inner_tokens([0, 2], start, bos, eos, pad) == []


def test_candidate_score_and_rerank():
    a = Candidate("a", [5, 6], l2r=-2.0, r2l=-8.0)  # left to right prefers a
    b = Candidate("b", [5, 7], l2r=-2.4, r2l=-4.0)  # right to left prefers b
    assert a.score(0.0) == pytest.approx(-2.0 / 4)
    assert b.score(1.0) == pytest.approx(-4.0 / 4)
    assert best_candidate([a, b], w=0.0) is a
    assert best_candidate([a, b], w=0.3) is b  # 0.7*-2.4 + 0.3*-4 > 0.7*-2 + 0.3*-8
    # without a right-to-left score the left-to-right one is used for both terms
    assert Candidate("c", [5], l2r=-3.0).score(0.3) == pytest.approx(-1.0)


# --- LineReader on a tiny random model ---------------------------------------


class TinyTokenizer:
    cls_token_id, pad_token_id, sep_token_id = 0, 1, 2

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(
            f"t{i}" for i in ids if not (skip_special_tokens and i in (0, 1, 2))
        )

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [self.decode(s.tolist(), skip_special_tokens) for s in seqs]


class TinyProcessor:
    tokenizer = TinyTokenizer()

    def __call__(self, images, return_tensors="pt"):
        import numpy as np
        import torch

        arrays = [
            np.asarray(im.resize((32, 32)), dtype="float32") / 255 for im in images
        ]
        return type(
            "Batch",
            (),
            {"pixel_values": torch.tensor(np.stack(arrays)).permute(0, 3, 1, 2)},
        )


def tiny_model(seed):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", exc_type=ImportError)
    torch.manual_seed(seed)
    encoder = transformers.ViTModel(
        transformers.ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=37,
        )
    )
    decoder = transformers.TrOCRForCausalLM(
        transformers.TrOCRConfig(
            vocab_size=20,
            d_model=32,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=37,
            max_position_embeddings=64,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            decoder_start_token_id=0,
        )
    )
    return transformers.VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)


@pytest.fixture
def reader():
    return LineReader(tiny_model(0), TinyProcessor(), r2l=tiny_model(1), device="cpu")


@pytest.fixture
def lines():
    return [
        Image.new("RGB", (200, 20), (200, 190, 170)),
        Image.new("RGB", (180, 24), (90, 80, 70)),
    ]


def test_configure_sets_training_start_token(reader):
    for m in (reader.model, reader.r2l):
        assert m.config.decoder_start_token_id == START
        assert m.generation_config.decoder_start_token_id == START


def test_sequence_log_probs_matches_logits(reader, lines):
    import torch

    pixel_values = reader.pixel_values(lines[:1]).repeat(2, 1, 1, 1)
    labels = [[0, 5, 6, 2], [0, 7, 2]]
    scores = reader.sequence_log_probs(reader.model, pixel_values, labels)
    for row, ids in enumerate(labels):
        decoder_inputs = torch.tensor([[START, *ids[:-1]]])
        with torch.inference_mode():
            logits = reader.model(
                pixel_values=pixel_values[row : row + 1],
                decoder_input_ids=decoder_inputs,
            ).logits
        expected = (
            logits.log_softmax(-1)[0, torch.arange(len(ids)), torch.tensor(ids)]
            .sum()
            .item()
        )
        assert scores[row] == pytest.approx(expected, abs=1e-4)


def test_candidates_and_transcribe(reader, lines):
    cands = reader.candidates(lines, beams=3)
    assert [len(c) for c in cands] == [3, 3]
    assert all(
        c.r2l is not None and c.text == reader.tokenizer.decode(c.tokens)
        for row in cands
        for c in row
    )
    texts = reader.transcribe(lines, beams=3)
    assert texts == [best_candidate(row).text for row in cands]
    assert reader.transcribe(lines[0], beams=3) == texts[0]  # one image -> one string
    assert isinstance(reader.transcribe(lines, rerank=False, beams=3), list)
