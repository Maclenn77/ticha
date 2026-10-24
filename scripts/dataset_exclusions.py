"""Documents left out of the line dataset, with the reason.

Ticha catalogues some manuscripts twice. When the two copies fall into different
splits, the copy in test is left out, so the evaluation never scores pages the
model was trained on and train stays the same as in version 1.0 (the models in
the report were trained on it). Used by ``build_dataset.py`` and
``export_hf_dataset.py``.
"""

DATASET_VERSION = "1.1.1"

EXCLUDED_DOCS = {
    "Za687": "same pages and transcription as Th687 (train): the manuscript is catalogued twice in Ticha",
}
