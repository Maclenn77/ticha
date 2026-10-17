"""Export the line dataset as a Hugging Face dataset folder (Parquet + card).

Reads ``data/splits`` (original lines) and ``data/augmented`` (augmented
copies), written by ``scripts/build_dataset.py``, and writes:

    data/hf_dataset/
    ├── README.md                         dataset card (configs, licence, description)
    ├── original/{train,validation,test}-NNNNN-of-NNNNN.parquet
    └── augmented/{train,test_degraded}-NNNNN-of-NNNNN.parquet

The two sets are separate configs (``original`` and ``augmented``), so loading
one never mixes in the other. Images are embedded as PNG bytes with the
``datasets`` Image feature, so ``load_dataset`` decodes them to PIL images.

Upload (after checking the folder), e.g. with huggingface_hub:

    from huggingface_hub import HfApi
    HfApi().create_repo("<user>/<name>", repo_type="dataset", private=True)
    HfApi().upload_folder(folder_path="data/hf_dataset", repo_id="<user>/<name>", repo_type="dataset")
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_exclusions import DATASET_VERSION, EXCLUDED_DOCS

ROOT = Path(__file__).resolve().parents[1]
SHARD_BYTES = 300 * 1024 * 1024

ORIGINAL_FEATURES = {
    "id": "string", "image": "image", "text": "string", "doc_id": "string", "page": "int32",
    "line": "int32", "language": "string", "source": "string", "year": "string", "town": "string",
    "doc_type": "string", "document_url": "string",
}
AUGMENTED_FEATURES = {
    "id": "string", "image": "image", "text": "string", "source_id": "string", "preset": "string",
    "params": "string", "doc_id": "string", "page": "int32", "line": "int32", "language": "string",
}


def hf_features(spec: dict) -> dict:
    return {name: ({"_type": "Image"} if kind == "image" else {"dtype": kind, "_type": "Value"})
            for name, kind in spec.items()}


def arrow_schema(spec: dict) -> pa.Schema:
    types = {"string": pa.string(), "int32": pa.int32(),
             "image": pa.struct([("bytes", pa.binary()), ("path", pa.string())])}
    schema = pa.schema([(name, types[kind]) for name, kind in spec.items()])
    meta = {"info": {"features": hf_features(spec)}}
    return schema.with_metadata({"huggingface": json.dumps(meta)})


def line_id(doc_id: str, page: int, line: int) -> str:
    return f"{doc_id}_p{int(page):03d}_l{int(line):03d}"


def write_shards(rows: pd.DataFrame, spec: dict, folder: Path, split: str) -> list:
    """Write rows (with an 'image_path' column) as Parquet shards of ~SHARD_BYTES."""
    schema = arrow_schema(spec)
    sizes = rows["image_path"].map(lambda p: Path(p).stat().st_size)
    shards, current, total = [], [], 0
    for index, size in zip(rows.index, sizes):
        if current and total + size > SHARD_BYTES:
            shards.append(current)
            current, total = [], 0
        current.append(index)
        total += size
    if current:
        shards.append(current)
    paths = []
    for k, indexes in enumerate(shards):
        chunk = rows.loc[indexes]
        columns = {}
        for name in spec:
            if name == "image":
                columns[name] = [{"bytes": Path(p).read_bytes(), "path": Path(p).name}
                                 for p in chunk["image_path"]]
            else:
                columns[name] = chunk[name].tolist()
        path = folder / f"{split}-{k:05d}-of-{len(shards):05d}.parquet"
        pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)
        paths.append(path)
    return paths


def original_rows(data: Path, split: str, meta: pd.DataFrame) -> pd.DataFrame:
    lines = pd.read_csv(data / "splits" / split / "lines.csv", keep_default_na=False)
    lines = lines[~lines["doc_id"].isin(EXCLUDED_DOCS)]
    lines["id"] = [line_id(d, p, n) for d, p, n in zip(lines["doc_id"], lines["page"], lines["line"])]
    lines["image_path"] = lines["image"]
    lines = lines.merge(meta, left_on="doc_id", right_index=True, how="left").fillna("")
    for col in ("page", "line"):
        lines[col] = lines[col].astype(int)
    return lines


def augmented_rows(data: Path, name: str, originals: pd.DataFrame) -> pd.DataFrame:
    index = pd.read_csv(data / "augmented" / name / "augment.csv")
    index = index[index["kind"] == "line"]
    by_image = originals.set_index("image")
    rows = []
    for _, aug in index.iterrows():
        if aug["source"] not in by_image.index:  # copy of a line from an excluded document
            continue
        src = by_image.loc[aug["source"]]
        out = Path(aug["output"])
        params = json.loads(aug["params"])
        rows.append({
            "id": f"{src['id']}_{out.stem.rsplit('_', 1)[-1]}", "image_path": str(out),
            "text": out.with_name(out.stem + ".gt.txt").read_text(encoding="utf-8").strip(),
            "source_id": src["id"], "preset": params.get("preset", ""), "params": aug["params"],
            "doc_id": src["doc_id"], "page": int(src["page"]), "line": int(src["line"]),
            "language": src["language"],
        })
    return pd.DataFrame(rows)


# Lines shown in the card: (doc_id, page, line, caption)
SAMPLES = [
    ("SDE746dT", 1, 24, "Spanish"),
    ("Za719", 1, 17, "Zapotec"),
    ("Ay800", 1, 22, "Spanish with Zapotec place names"),
]
SAMPLE_WIDTH = 900


def _save_sample(source: Path, target: Path) -> None:
    from PIL import Image

    with Image.open(source) as im:
        im = im.convert("RGB")
        if im.width > SAMPLE_WIDTH:
            im = im.resize((SAMPLE_WIDTH, round(im.height * SAMPLE_WIDTH / im.width)))
        im.save(target, optimize=True)


def write_samples(data: Path, out: Path, originals: pd.DataFrame) -> str:
    """Copy a few line images to samples/ and return the card's Examples section."""
    folder = out / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    rows = ["| Image | Text | Language |", "|---|---|---|"]
    first = None
    for doc_id, page, line, caption in SAMPLES:
        match = originals[(originals["doc_id"] == doc_id) & (originals["page"] == page)
                          & (originals["line"] == line)]
        if match.empty:
            continue
        row = match.iloc[0]
        name = f"{row['id']}.png"
        _save_sample(Path(row["image"]), folder / name)
        text = row["text"].replace("|", "\\|")
        rows.append(f"| ![{row['id']}](samples/{name}) | {text} | {caption} |")
        if first is None:
            first = row
    section = ["## Examples", "", "Lines from the `original` config:", "", *rows]
    if first is not None:
        index = pd.read_csv(data / "augmented" / "train" / "augment.csv")
        copies = index[index["source"] == first["image"]]
        shown = []
        for preset in ("variety", "degrade"):
            pick = copies[copies["params"].str.contains(f'"preset": "{preset}"')]
            if len(pick):
                out_path = Path(pick.iloc[0]["output"])
                name = f"{first['id']}_{preset}.png"
                _save_sample(out_path, folder / name)
                shown.append((preset, name))
        if shown:
            section += ["", f"The same line ({first['id']}) and two of its copies in the `augmented` config:", "",
                        "| Version | Image |", "|---|---|",
                        f"| original | ![original](samples/{first['id']}.png) |"]
            section += [f"| `{preset}` | ![{preset}](samples/{name}) |" for preset, name in shown]
    return "\n".join(section) + "\n"


def card(counts: dict, repo_id: str, examples: str = "") -> str:
    o, a = counts["original"], counts["augmented"]
    total = sum(o.values())
    excluded_note = "Removed from train (with their augmented copies): " + "; ".join(
        f"`{doc_id}`, {reason}" for doc_id, reason in EXCLUDED_DOCS.items())
    return f"""---
license: cc-by-nc-sa-4.0
language:
- es
- zap
task_categories:
- image-to-text
tags:
- handwritten-text-recognition
- htr
- ocr
- colonial-zapotec
- manuscripts
- kraken
pretty_name: Ticha Colonial Zapotec and Spanish manuscript lines
size_categories:
- 10K<n<100K
configs:
- config_name: original
  default: true
  data_files:
  - split: train
    path: original/train-*.parquet
  - split: validation
    path: original/validation-*.parquet
  - split: test
    path: original/test-*.parquet
- config_name: augmented
  data_files:
  - split: train
    path: augmented/train-*.parquet
  - split: test_degraded
    path: augmented/test_degraded-*.parquet
---

# Ticha manuscript lines (Colonial Zapotec and Spanish)

Line images of 17th to 19th century manuscripts from Oaxaca, Mexico, paired with their transcriptions, for
training and evaluating handwritten text recognition (HTR). The pages and transcriptions come from
[Ticha: a digital text explorer for Colonial Zapotec](https://ticha.haverford.edu) (Haverford College).

The code that downloads the pages, pairs the lines and builds this dataset is at
[github.com/Maclenn77/ticha](https://github.com/Maclenn77/ticha).

## Configs and splits

| Config | Split | Lines | Description |
|---|---|---|---|
| `original` | `train` | {o['train']:,} | Original line images |
| `original` | `validation` | {o['validation']:,} | Original line images |
| `original` | `test` | {o['test']:,} | Original line images |
| `augmented` | `train` | {a['train']:,} | Augmented copies of the train lines (half `degrade`, half `variety`) |
| `augmented` | `test_degraded` | {a['test_degraded']:,} | One `degrade` copy of each test line, no geometric changes |

The splits are made by **document**: no document appears in two splits. Within each language, documents are
assigned so that lines are close to 80/10/10 ({total:,} original lines in total). Ticha lists a few manuscripts
twice under different ids; when the copies fall into different splits, the train copy is left out (see Versions).

Augmented data is a separate config so it is never mixed in by accident. To train with it, concatenate
`original/train` and `augmented/train`; validate on `original/validation`. Evaluate on `original/test` (real
scans) and on `augmented/test_degraded` (the same lines made to look like low-quality scans) to see how much a
model degrades on poor reproductions.

```python
from datasets import load_dataset, concatenate_datasets

original = load_dataset("{repo_id}", "original")
augmented = load_dataset("{repo_id}", "augmented")
train = concatenate_datasets([original["train"], augmented["train"]])
```

{examples}
## Versions

| Version | Revision | Changes |
|---|---|---|
| 1.0 | [`b9747c9`](https://huggingface.co/datasets/{repo_id}/tree/b9747c957ff1a642d86fdb7f5a4cbc717a249e20) (2026-10-12) | First release |
| {DATASET_VERSION} | this revision | {excluded_note} |

Validation and test are unchanged, so scores on them are comparable across versions. Models trained on version
1.0 have seen the pages of Za687: leave that document out when evaluating them on the test split. Load a
specific version with `load_dataset("{repo_id}", "original", revision="<revision>")`.

## Fields

`original`:

- `id`: `<doc_id>_p<page>_l<line>`
- `image`: line image (PNG), cropped along the line's polygon with the outside painted white
- `text`: transcription of the line, as in Ticha, including editorial marks such as `q[ue]`, `(?)` and `[roto]`
- `doc_id`, `page`, `line`: Ticha document id, page number in the document, line number on the page
- `language`: `Zapotec`, `Spanish` or `Spanish with some Zapotec`
- `source`: `auto` (paired automatically) or `reviewed` (page checked line by line)
- `year`, `town`, `doc_type`, `document_url`: document metadata from Ticha

`augmented`: `id`, `image`, `text`, `doc_id`, `page`, `line`, `language`, plus `source_id` (the `id` of the
original line), `preset` (`degrade` or `variety`) and `params` (JSON with every augmentation parameter).

## How it was made

1. Page images were rebuilt from Ticha's zoom tiles; only documents with a transcription marked "Completed and
   Checked" or "Completed but not Checked" were kept (125 documents, 385 pages).
2. Pages were segmented into lines with kraken's default baseline model.
3. Each line was read by an HTR model (McCATMuS fine-tuned on Ticha lines) and the transcription was cut into
   the word spans that best match the readings, which also handles transcriptions whose line breaks do not
   follow the manuscript and words broken across lines.
4. Pairs were kept when their text fits the reading clearly better than the neighbouring lines' texts. Pages
   with too many doubtful pairs (48) were checked line by line against the images; lines that could not be
   confirmed have no text and are not in the dataset.
5. Augmentation: `degrade` imitates Ticha's untranscribed notarial scans (lower resolution, greyish paper with a
   cool cast, faded brown ink, washed-out or underexposed tone, uneven light, stains); `variety` adds
   brightness, contrast, blur, noise, JPEG compression, small rotation, slant, scale and elastic distortion.

Code: [Maclenn77/ticha](https://github.com/Maclenn77/ticha) (`ticha-scraper lines`, `scripts/build_dataset.py`,
`scripts/export_hf_dataset.py`).

## Limitations

- Transcriptions follow Ticha's editorial conventions; some are partial or uncertain (marked with `(?)`).
- Alignment is automatic. Spot checks found the large majority of pairs correct, but a few lines may contain a
  word that belongs to the neighbouring line.
- 64 lines on reviewed pages were trimmed or corrected from the images by an AI assistant (Claude); the
  Zapotec ones in particular have not been checked by a specialist.
- Three characters (½, à, ò) appear in validation but not in train.
- The fine-tuned model used for alignment was trained on lines of all splits; do not use it to report test
  results on this dataset.

## Licence

This dataset is a derivative of [Ticha](https://ticha.haverford.edu) and is shared under the same licence,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only, with
attribution, and any derivative must use the same licence.

Changes made to the Ticha material: page images rebuilt from the zoom tiles and cut into line images along each
line's polygon; transcriptions split into lines and aligned with the images, with editorial line references
(e.g. `63r-5`) and pasted Word markup removed; 64 line texts on reviewed pages trimmed to what is visible on the
line or corrected from the image; and, in the `augmented` config, synthetic copies of the line images.

## Citation

Dataset author: J. P. Pérez Tejada ([ORCID 0009-0003-9501-9918](https://orcid.org/0009-0003-9501-9918)).

When you use this dataset, cite both the dataset and Ticha, the source of the images and transcriptions. If you
use or adapt the code that built it, cite the repository as well.

```bibtex
@misc{{perez_tejada_ticha_htr_lines_2026,
  author       = {{P{{\\'e}}rez Tejada, J. P.}},
  title        = {{Ticha manuscript lines: a line-level HTR dataset of Colonial Zapotec and Spanish manuscripts}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note         = {{Derived from Ticha: a digital text explorer for Colonial Zapotec}}
}}

@misc{{ticha,
  editor       = {{Lillehaugen, Brook Danielle and Broadwell, George Aaron and Oudijk, Michel R. and
                  Allen, Laurie and Plumb, May Helena and Zarafonetis, Mike}},
  title        = {{Ticha: a digital text explorer for Colonial Zapotec}},
  publisher    = {{Haverford College Libraries}},
  howpublished = {{\\url{{https://ticha.haverford.edu}}}}
}}

@software{{perez_tejada_ticha_code_2026,
  author       = {{P{{\\'e}}rez Tejada, J. P.}},
  title        = {{ticha: scraping Ticha and building a line-level HTR dataset}},
  year         = {{2026}},
  url          = {{https://github.com/Maclenn77/ticha}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "hf_dataset")
    parser.add_argument("--repo-id", default="maclenn77/ticha-htr-lines",
                        help="Only used in the card's loading example")
    args = parser.parse_args()

    data, out = args.data.resolve(), args.output.resolve()
    shutil.rmtree(out, ignore_errors=True)
    (out / "original").mkdir(parents=True)
    (out / "augmented").mkdir(parents=True)

    texts = pd.read_csv(data / "texts_transcribed.csv", dtype=str, keep_default_na=False)
    meta = (texts.drop_duplicates("ticha_id").set_index("ticha_id")[["year", "town", "doc_type", "url"]]
            .rename(columns={"url": "document_url"}))
    meta["year"] = meta["year"].str.replace(r"\.0$", "", regex=True)

    counts = {"original": {}, "augmented": {}}
    originals = []
    for split in ("train", "validation", "test"):
        rows = original_rows(data, split, meta)
        paths = write_shards(rows, ORIGINAL_FEATURES, out / "original", split)
        counts["original"][split] = len(rows)
        originals.append(rows)
        print(f"original/{split}: {len(rows)} lines, {len(paths)} shard(s)")
    originals = pd.concat(originals, ignore_index=True)
    for name in ("train", "test_degraded"):
        rows = augmented_rows(data, name, originals)
        paths = write_shards(rows, AUGMENTED_FEATURES, out / "augmented", name)
        counts["augmented"][name] = len(rows)
        print(f"augmented/{name}: {len(rows)} lines, {len(paths)} shard(s)")
    examples = write_samples(data, out, originals)
    (out / "README.md").write_text(card(counts, args.repo_id, examples), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
