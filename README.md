# Cross-Lingual Sentiment and Intent Classification for Indian Languages

Fine-tunes XLM-RoBERTa (base) on **English only** and applies it, zero-shot, to Hindi, Bengali, Tamil, Telugu and Marathi. The system predicts:

- **Sentiment:** negative / neutral / positive
- **Intent:** inquiry / opinion / factual
- A **probability for every class** on both outputs

No word-for-word translation is used at inference: transfer comes from XLM-R's multilingual representations. Everything runs on a free Kaggle T4 x2 GPU (fp16, max length 128).

## Contents

| File | What it is |
|---|---|
| `cross-lingual-dualhead-classifier.ipynb` | **Main notebook.** Shared encoder with two heads: training, evaluation, confidence scores, interactive prediction, confusion matrices |
| `sentimentclassifier.ipynb` | Sentiment baseline (English training, Indian-language zero-shot evaluation) |
| `intentclassifier.ipynb` | Intent classifier, v1 and v2 (with adversarial augmentation) |
| `REPORT.md` / report document | Full evaluation report and analysis |
| `results_summary.json` | All reported metrics, copied from the notebook outputs |
| `figures/` | Confusion matrices for the three models |
| `requirements.txt` | Python dependencies |

## Results

Zero-shot sentiment on IndicSentiment (binary, 998 test sentences per language), accuracy:

| Language | Standalone | Dual-head |
|---|---|---|
| Hindi | 91.28% | 91.48% |
| Bengali | 90.08% | 89.68% |
| Tamil | 90.78% | 89.28% |
| Telugu | 87.88% | 86.77% |
| Marathi | 90.08% | 89.58% |

Other results:

- Hindi 3-class sentiment (SentiHin-2500, standalone): 86.76% accuracy, 0.853 macro F1
- English sentiment: 69.77% (standalone, official test set); 73.04% (dual-head, on a 10% slice of the training split, so **not comparable**)
- English intent: 97.49% (v2), 97.07% (dual-head)
- Multilingual intent, 75 hand-written sentences: 98.67% (standalone), 100% (dual-head)

The dual-head model did not improve cross-lingual sentiment over the standalone model. See `REPORT.md` for the analysis and limitations.

## Data

| Use | Dataset |
|---|---|
| Sentiment training | `cardiffnlp/tweet_eval` (sentiment) |
| Intent: Inquiry | `jonaskoenig/Questions-vs-Statements-Classification` |
| Intent: Opinion / Factual | `SetFit/subj` (subjective / objective) |
| Cross-lingual sentiment test | `ai4bharat/IndicSentiment` (hi, bn, ta, te, mr), test only |
| Hindi 3-class test | `MLap/SentiHin-2500`, test only |
| Cross-lingual intent test | 75 hand-written sentences (15 sentences x 5 languages), included in the notebooks |

Intent labels are **proxies** built from these datasets, not human-annotated intent. Read the limitations before quoting the intent numbers.

## Running it

1. Open the notebook on Kaggle, set the accelerator to **GPU T4 x2**, and turn Internet on.
2. Run all cells. The dual-head notebook takes roughly 50 minutes.
3. **Save the models immediately.** `/kaggle/working` is cleared when a session restarts. Download the files or save them as a Kaggle Dataset.

Local setup:

```bash
pip install -r requirements.txt
```

## Predicting on new text

In the dual-head notebook, after training:

```python
predict_sentence("यह खाना बहुत स्वादिष्ट है।", model, tokenizer)
# Text: यह खाना बहुत स्वादिष्ट है।
#   Sentiment: Positive (99.8% confidence)
#   Intent: Opinion (99.3% confidence)
```

It returns a dict with the sentiment label, intent label and a confidence for each.

## Known limitations

- Intent labels come from proxy datasets, so English intent accuracy partly reflects style and domain.
- Intent confidence is overconfident (average 0.993) and is not calibrated.
- The multilingual intent test set is small and templated.
- Three of the six adversarial test sentences were later added to the augmentation data, so the 6/6 result is contaminated.
- Results are from a single seed per model; differences under about 1 point are within noise.
- A tokens-per-word (agglutination) analysis is not included.
