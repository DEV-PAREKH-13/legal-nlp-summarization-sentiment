# legal-nlp-summarization-sentiment-
# Legal Text Summarization and Judgment Classification for Indian Court Judgments

An NLP pipeline that takes a raw Indian court judgment as input and returns:
1. A concise abstractive summary
2. A predicted outcome class (Accepted / Rejected / Other) with a confidence score

Built as part of Major Project-I (01CE0716), B.Tech Computer Engineering, Marwadi University.

## Team

- Dev Parekh (92301703118)
- Jhanvi Viradia (92410103092)
- Daman Kanparia (92410103042)

Guided by Prof. Krupali Gosai, Department of Computer Engineering.

## What's inside

**Classification** — three transformer models are fine-tuned and compared head-to-head on the same split, class-weighted loss, and metrics:
- `law-ai/InLegalBERT` — domain-pretrained on Indian legal text (head+tail truncation for long documents)
- `allenai/longformer-base-4096` — handles long documents natively via sparse attention
- `roberta-base` — general-purpose baseline, to quantify what domain pretraining buys

**Summarization** — abstractive summarization via Legal-Pegasus (fallback: T5-small), with optional fine-tuning on `case_info` reference summaries when available.

**Evaluation** — accuracy, precision, recall, weighted F1 for classification; ROUGE-1/2/L and BERTScore for summarization.

## Dataset

`case_files_total.csv` — 53,446 Indian court judgment records with columns: `name`, `case_category`, `case_type`, `case_info`, `judgement`, `label`.

Not included in this repo due to size. Place it locally and update `DATA_PATH` in the notebook before running.

## How to run

1. Open `notebook.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Run Section 0 to install dependencies (Colab's pre-installed numpy/pandas/torch/sklearn are left untouched to avoid version conflicts).
4. Mount Google Drive when prompted, and set `DATA_PATH` to point at your copy of `case_files_total.csv`.
5. Run all cells top to bottom. Sections are numbered and self-contained (0 → install, 1 → imports, 3 → load data, ... 22 → full pipeline, 24 → final report).

## Using the trained pipeline

```python
result = analyze_case(judgment_text)
# {
#   "original_judgment": ...,
#   "generated_summary": ...,
#   "compression_ratio": ...,
#   "predicted_class": ...,
#   "confidence_score": ...,
#   "prediction_probability": {...}
# }
```

## Status

Results reported so far are from a 360-record validation subset of the full 53,446-record corpus, used to confirm the pipeline runs end-to-end. A full-corpus run is planned before final results are treated as conclusive — see the project report's Limitations section for details.

## Report

Full project report (methodology, results, limitations): see `/report`.
