# BESSTIE Sarcasm Detection — Cross-Variety NLP

COMM061 NLP Coursework, Group PG19 — University of Surrey.

A sarcasm detection system that compares two fine-tuning approaches across three English varieties (en-UK, en-AU, en-IN), trained on the [BESSTIE-CW-26 dataset](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26):

- **RoBERTa-base** with full fine-tuning
- **TinyLlama-1.1B** with LoRA adapters (parameter-efficient fine-tuning)

**Best result:** RoBERTa en-AU → en-AU achieves Macro-F1 = 0.770 ± 0.008 (multi-seed), with sarcastic recall of 0.801 — so it isn't just predicting the majority class.

**Live demo:** https://huggingface.co/spaces/Sriram28/besstie-sarcasm

## What's in here

| Path | Description |
|------|-------------|
| `main.ipynb` | Full implementation for Q1–Q5, runnable top to bottom |
| `space_app/` | Gradio web app deployed to HuggingFace Spaces |
| `outputs/` | Figures, confusion matrices, result CSVs, and LIME explanations |
| `requirements.txt` | Python dependencies |

## The work, by section

- **Q1 — Dataset analysis:** label distributions, vocabulary overlap (Jaccard + TF-IDF cosine), and a POS-distribution comparison between sarcastic and non-sarcastic text.
- **Q2.1 — Baselines vs PTLM:** TF-IDF with Logistic Regression and Linear SVM, compared against fine-tuned RoBERTa-base (multi-seed).
- **Q2.2 — Cross-variety RoBERTa:** a 3×3 transfer matrix (train on one variety, test on all three). The en-IN model initially collapsed under class imbalance (~7% sarcastic), which motivated a custom `WeightedTrainer` with class-weighted loss. LIME is applied to interpret six representative errors.
- **Q2.3 — LoRA on TinyLlama:** three adapters (one per variety), mirroring the Q2.2 matrix for a direct comparison.
- **Q3 — Consolidated evaluation:** a 20-row master table with per-class metrics and confusion matrices.
- **Q4 — Error analysis:** regex-based error categorisation plus a few-shot prompting experiment. A sanity check exposed a failure mode where the LoRA adapter degenerates to always predicting sarcastic — something accuracy hides but Macro-F1 catches.
- **Q5 — Deployment:** a Gradio service hosting all six models, with the three LoRA adapters swapped via PEFT's `set_adapter()` to save GPU memory. Q5.2 benchmarks inference latency across batch sizes.

## Running it

\`\`\`bash
pip install -r requirements.txt
jupyter notebook main.ipynb
\`\`\`

The notebook is designed to run top to bottom. Run the shared bootstrap cell near the top first — it defines the constants, helpers, and the `WeightedTrainer` class used across the later sections.

## Models

Trained models are hosted on the HuggingFace Hub under user `Sriram28`:

- `Sriram28/besstie-roberta-en-{uk,au,in}`
- `Sriram28/besstie-lora-en-{uk,au,in}`

