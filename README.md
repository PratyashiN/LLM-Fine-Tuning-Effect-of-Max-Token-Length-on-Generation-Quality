# LLM Fine-Tuning: Effect of Max Token Length on Generation Quality

Fine-tuned GPT-2 and DistilGPT-2 on instruction-response pairs and evaluated how `max_new_tokens` affects output quality.

---

## What This Project Does

- Fine-tunes small language models (GPT-2, DistilGPT-2) on the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- Compares two generation settings: `max_new_tokens=10` vs `max_new_tokens=30`
- Evaluates outputs using a custom **F1-based word overlap score** (similar to ROUGE)

---

## Experiments

| Experiment | Model | max_new_tokens | Batch Size | Context Length |
|---|---|---|---|---|
| `max_token_10.ipynb` | DistilGPT-2 | 10 | 1 | 64 |
| `max_token_30.ipynb` | GPT-2 | 30 | 2 | 128 |

---

## Results

**max_new_tokens = 10**
| Question | Model Output | Score |
|---|---|---|
| What is gravity? | Gravity is a measure of the speed of a projectile | 0.35 |
| What is binary search? | Binary search is a type of search algorithm used to | 0.42 |
| What is AI? | AI is a field of study that focuses on the | 0.32 |
| What is overfitting? | Overfitting is a type of software that allows users | 0.17 |

**max_new_tokens = 30**
| Question | Model Output | Score |
|---|---|---|
| What is gravity? | Gravity is a force which is distributed around the Earth | 0.44 |
| What is binary search? | Binary search is a type of data analysis where data is stored in a binary array | 0.27 |
| What is AI? | AI is a field of computer science that aims to develop algorithms... | 0.15 |
| What is overfitting? | Overfitting is a type of software engineering technique used in software development | 0.15 |

**Key finding:** More tokens don't always mean better answers — the model sometimes drifts off-topic with more room to generate.

---

## Evaluation Metric

Custom F1 word overlap score:

```
precision = common_words / predicted_words
recall    = common_words / reference_words
score     = 2 * precision * recall / (precision + recall)
```

---

## Setup

```bash
pip install torch transformers datasets accelerate
```

Then run either notebook:
```
jupyter notebook max_token_10.ipynb
jupyter notebook max_token_30.ipynb
```

---

## Stack

- Python, PyTorch
- HuggingFace Transformers
- Alpaca Dataset (5,000 samples)
- Jupyter Notebook
