# improved_ocr_agent

`improved_ocr_agent` is a revised version of the original `ocr_agent` extractor.

It preserves the original OCR path, but changes the non-OCR path:

- OCR-routed pages: keep the original OCR / VLM flow
- non-OCR research / structured-visual pages: route through `ia_phase1`
- non-OCR text-heavy report/manual pages: keep the native `ocr_agent` extraction path

This package is used in the benchmark as the third system alongside:

- `ia_phase1`
- `ocr_agent`

## Important constraint

This package is **not standalone**.

Unlike the original `ocr_agent`, it depends on `ia_phase1` for the non-OCR scholarly / structured-visual path.

The bridge is here:

- `improved_ocr_agent/ia_phase1_bridge.py`

That bridge imports:

- `ia_phase1.markdown_export`

So if you move this package into another repository, you must also make the required `ia_phase1` subset importable there.

## Main entry points

- `improved_ocr_agent/hybrid_pdf_extractor.py`
  - primary PDF -> Markdown extractor
- `improved_ocr_agent/document_agent.py`
  - higher-level document indexing / retrieval interface
- `improved_ocr_agent/pipeline_custom.py`
  - OCR / VLM backend client

## Non-OCR routing

The non-OCR router lives in:

- `improved_ocr_agent/non_ocr_routing.py`

It assigns non-OCR pages to one of two handlers:

- `ia_phase1`
- `native`

Current intended behavior:

- research papers and structured visual documents -> `ia_phase1`
- text-heavy reports / manuals / proposals -> native `ocr_agent` path
- OCR pages -> unchanged OCR path

## Local code dependencies

### Required `ia_phase1` subset

At minimum, this package depends on the following `ia_phase1` modules:

- `modules/phase1-python/src/ia_phase1/parser.py`
- `modules/phase1-python/src/ia_phase1/figures.py`
- `modules/phase1-python/src/ia_phase1/tables.py`
- `modules/phase1-python/src/ia_phase1/equations.py`
- `modules/phase1-python/src/ia_phase1/equation_latex.py`
- `modules/phase1-python/src/ia_phase1/math_markdown.py`
- `modules/phase1-python/src/ia_phase1/sectioning.py`
- `modules/phase1-python/src/ia_phase1/markdown_export/__init__.py`
- `modules/phase1-python/src/ia_phase1/markdown_export/export.py`
- `modules/phase1-python/src/ia_phase1/markdown_export/bundle.py`
- `modules/phase1-python/src/ia_phase1/markdown_export/document_model.py`
- `modules/phase1-python/src/ia_phase1/markdown_export/models.py`
- `modules/phase1-python/src/ia_phase1/markdown_export/quality.py`

### Internal package modules

Core files inside `improved_ocr_agent`:

- `anchor.py`
- `document_agent.py`
- `front_matter.py`
- `hybrid_pdf_extractor.py`
- `ia_phase1_bridge.py`
- `image_utils.py`
- `metrics.py`
- `non_ocr_routing.py`
- `ocr_prompt.py`
- `pipeline_custom.py`
- `renderpdf.py`
- `work_queue.py`

## Python dependencies

### Required for the main extractor path

- `pymupdf` / `fitz`
- `pdfplumber`
- `pypdf`
- `Pillow`
- `httpx`
- `requests`
- `PyYAML`

### Required for anchor / rendering helpers

- `ftfy`
- `pypdfium2`
- `zstandard`

### Optional / feature-dependent

- `numpy`
- `scikit-learn`
- `sentence-transformers`
- `lingua`
- `boto3`
- `huggingface_hub`
- `tqdm`

If optional packages are missing, some workflows may degrade rather than fail, especially in `document_agent.py` and `pipeline_custom.py`.

## Usage in this repo

Run `improved_ocr_agent` directly through the benchmark helper:

```bash
PYTHONPATH=. backend/.webenv/bin/python markdown_evaluation/scripts/run_ocr_agent.py \
  --system-name improved_ocr_agent \
  --overwrite
```

With a live OCR server:

```bash
PYTHONPATH=. backend/.webenv/bin/python markdown_evaluation/scripts/run_ocr_agent.py \
  --system-name improved_ocr_agent \
  --ocr-server https://<your-server>/v1 \
  --overwrite
```

## Porting note

If you want to use this package in the original `AIserver` / `BatchAgent` codebase, do **not** treat it as a blind drop-in replacement for `ocr_agent`.

You must either:

1. vendor the required `ia_phase1` subset into that repo and patch the bridge import path, or
2. refactor the bridge so it no longer imports `ia_phase1` directly

Without that, replacing `BatchAgent/ocr_agent` with this package will likely break import/runtime behavior.
