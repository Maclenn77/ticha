# Evaluation plan: Ticha HTR models

Instructions for the next steps: evaluate the models that are already trained, on the same data and under the same conditions, and report the performance of the final models.

## Rules

1. **Report final performance only.** Each model is decoded with the settings it was trained for: decoder start token `0` (`<s>`) for every model fine-tuned in this project, `2` for the zero-shot pretrained TrOCR. Set the start token explicitly in both `model.config` and `model.generation_config`; don't rely on the values saved with a checkpoint. The report describes how models are decoded, not how earlier numbers were measured.
2. **Never retrain to evaluate.** Every model is already trained. The evaluation only loads saved files. A model is missing only when its weights file doesn't exist.
3. **Everything trained is saved to persistent disk, whatever its type.** That includes neural weights, small models (cloze, classifiers), tokenizers and vocabularies built in memory (the character vocabulary of the char decoders), and settings chosen on validation (rerank weight `w`, cloze and fusion weight `β`). Nothing needed to rebuild a result may live only in a kernel's memory. Every training notebook follows the same pattern: if the file exists, load it; otherwise train and save it at once.
4. **Validation chooses, test reports.** Checkpoints, `w` and `β` are chosen on validation. The test split is decoded once per model, and its predictions are saved.
5. **Same data for every model.** The cleaned validation and test lines (`clean_text` with `keep_guesses=False`) and the degraded test copy, restricted to the same lines. **The test set has no Za687** (936 lines): it is the same manuscript as Th687, which the v1.0 models trained on. The notebooks drop it when they load the test split, and every saved result was recomputed without it (`scripts/fix_test_results.py`; the old files are archived in `data/archive/results_with_Za687/` and `~/ticha-runs-archive/`). Report only these numbers.
6. **Predictions are cached** per model and split (`predictions/<model>__<split>.csv`). Rerunning the notebook only recomputes tables and charts.

## Models to evaluate

All are trained and on the VM, under `~/ticha-runs/`.

| Family | Model | Folder |
|---|---|---|
| Baseline | Pretrained TrOCR, zero-shot | `microsoft/trocr-base-handwritten` |
| End-to-end fine-tuning | 5 epochs | `trocr-base-handwritten-finetuned` |
| | Smoothing + cosine, 12 epochs | `trocr-base-handwritten-finetuned-smooth-cosine` |
| | + augmentation (current best) | `trocr-base-handwritten-finetuned-smooth-cosine-aug` |
| Decoupled, no augmentation | CTC encoder only | `trocr-modified/encoder_ctc` |
| | Bridge, word pieces, `zap` / `zap_spa` | `trocr-modified/bridge_*` |
| | Char bridge `l2r`, and `l2r` + `r2l` rerank, `zap` / `zap_spa` | `trocr-modified/char_bridge_*` |
| | CTC + cloze `zap` / `zap_spa` | `trocr-modified/cloze_*` |
| Decoupled, augmentation + regularization | CTC encoder (and the head variants in `encoder_ctc_runs/`, if kept as final) | `trocr-modified-aug-reg/` |
| | Bridges `zap` / `zap_spa`, CTC + cloze `zap` / `zap_spa` | `trocr-modified-aug-reg/` |
| Language-model fusion | Best end-to-end model + Zapotec LM, rescoring and shallow fusion | `notebooks/ticha_trocr_lm_fusion.ipynb`, LMs in `trocr-modified/decoder_*` and `lm-manuscript/` |

The **decoupled bridges were selected with the old decoding**. Their saved weights are kept as they are. The evaluation decodes them with start token `0`, and that is the number reported.

## Steps

### 1. Protect the saved models (before anything else)
- The VM's disk is set to **delete with the VM** (`autoDelete: true`). Stopping the VM keeps everything; deleting it would lose every model. Either turn auto-delete off for the boot disk, or copy the final models to a Cloud Storage bucket in the same project (about $0.30/month for 15 GB). Both are changes to the cloud project, so the user confirms them first.
- The VM disk is **93% full** (12 GB free). Epoch checkpoints use about 48 GB, and they're only needed to resume training. Once a run is final, keep its selected weights (`model_state.pt`, `model.safetensors`, `encoder_ctc.pt`, `decoder_state.pt`, `cloze_lm.pt`), its `trainer_state.json` and its history files. Delete the checkpoints only with the user's explicit OK.

### 2. Save what is still only in memory
- **Character vocabulary** of the char decoders (Part 4): rebuild it with the same code as the training notebook (Part 1 corpora: `data/printed.csv` + cleaned train lines). Check that its size equals the saved embedding size. Save it as `char_vocab.json` next to each char model.
- **Chosen settings:** write `w` (rerank) and `β` (cloze, fusion) with their validation CER to a `selection.json` in each run folder. If a setting was never chosen (the no-augmentation Part 4.3 never ran), choose it on validation in the evaluation notebook and save it there.
- Add a short `model_card.json` to each final model folder: family, loader, start token, vocabulary file, training data (original or augmented, dataset version), best validation CER.

### 3. One shared loader module
Move the model classes and loaders (`MaskedLinear`, `EncoderCTC`, `ClozeCharLM`, `CharTokenizer`, `load_trained_bridge`, `load_trained_char_bridge`, the cloze correction, the LM fusion processor) into one module, e.g. `src/ticha/models.py`. The training notebooks and the evaluation notebook import it, so they can't drift apart.

### 4. The evaluation notebook (`notebooks/ticha_evaluation.ipynb`)
Base it on the existing `ticha_trocr_evaluation.ipynb`, which already has the cleaning, caching, bootstrap and chart style.

1. **Model registry:** one table with name, family, folder, loader, start token, decoding settings.
2. **Predictions:** every model on validation, test and degraded test, with the same beam width and maximum length, cached to disk.
3. **Settings from validation:** read `selection.json`. For any setting not chosen yet, choose it on validation and save it before the test is decoded.
4. **Metrics:** CER and WER with 95% bootstrap intervals (2,000 resamples of test lines). Also the share of lines read exactly, and error types (substitutions, insertions, deletions).
5. **Comparisons**, each a paired bootstrap on the same lines:
   - best end-to-end vs best decoupled model;
   - with vs without augmentation (end-to-end and decoupled);
   - characters vs word pieces (char `l2r` vs bridge);
   - with vs without the Zapotec LM (fusion and rescoring vs the end-to-end model alone; cloze vs CTC alone).
6. **Breakdowns:** by language (Zapotec, Spanish, mixed), by document, real vs degraded scans, and the distribution of CER over lines.
7. **Examples:** lines where each method helps most and hurts most.
8. **Outputs:** `evaluation/summary.csv`, the breakdown CSVs and the figures (PNG), all saved next to the predictions.

### 5. Where to run it
- **On the VM (L4):** about 12–15 models × 3 splits × about 1,000 lines, roughly 1–1.5 h, or **about $1–1.5**. The auto-stop shuts the VM down 30 minutes after the kernel goes idle. Cached predictions make an interruption harmless.
- **Locally (M2, MPS):** possible for the tables and charts once the predictions are cached. Decoding everything locally would take 10 hours or more.

### 6. Report
- One main table: the final models with test CER, WER and 95% intervals, without Za687.
- The comparisons in step 4.5, with their paired-bootstrap results.
- The language breakdown, with the Zapotec column first, since that is what the language-model work targets.
- State the decoding settings (beam width, maximum length, start token), the validation-chosen `w` and `β`, the data cleaning, and that Za687 is excluded.
