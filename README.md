# ML Approaches for Ad/Tracker Blocking

Comparing TF-IDF and fine-tuned LLaMA 8B for classifying web trackers via DNS CNAME analysis.

## Problem

Ad-blockers rely on domain blacklists that break when trackers use CDNs (Cloudflare, Akamai)
shared with legitimate content. This project explores ML as a more adaptable alternative,
focusing on CNAME-based classification using the DuckDuckGo Tracker Radar dataset.

## Approach

**Dataset:** DuckDuckGo Tracker Radar — third-party domains with resource rules, DNS CNAMEs,
and fingerprint levels. Resources with fingerprint level > 2 labeled as trackers.

**Model 1 — TF-IDF + Logistic Regression (baseline)**
Features: domain, original CNAME, resolved CNAME. Fast but poor recall.

**Model 2 — LLaMA 8B fine-tuned with LoRA**
Prompt-based classification on domain + resource rule. Better generalization on unseen patterns.

## Results

| Model    | Accuracy | Precision | Recall | F1   |
|----------|----------|-----------|--------|------|
| TF-IDF   | 0.97     | 0.81      | 0.37   | 0.51 |
| LLaMA 8B | 0.84     | 0.76      | 0.84   | 0.79 |

TF-IDF's high accuracy is misleading — low recall means it misses most actual trackers.
LLaMA's balanced precision/recall makes it significantly more useful in practice.

## Contents

- `tfidf_baseline.ipynb` — TF-IDF pipeline and evaluation
- `llama_finetuning.ipynb` — LLaMA 8B LoRA fine-tuning and evaluation

## Stack

Python, PyTorch, HuggingFace Transformers, LoRA (PEFT), scikit-learn
