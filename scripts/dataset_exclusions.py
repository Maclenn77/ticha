"""Documents left out of the line dataset, with the reason.

Ticha catalogues some manuscripts twice. When the two copies fall into different
splits, the copy in train is left out, so the evaluation never scores pages the
model was trained on. Used by ``build_dataset.py`` and ``export_hf_dataset.py``.
"""

DATASET_VERSION = "1.1"

EXCLUDED_DOCS = {
    "Th687": "same pages and transcription as Za687 (test): the manuscript is catalogued twice in Ticha",
}
