# ToS Clause Detector

[![Lint](https://github.com/dzoulk/ToS-Clause-Detector-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/dzoulk/ToS-Clause-Detector-ML/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project fine-tunes DistilBERT to flag unfair clauses in Terms of Service documents, things like forced arbitration, unilateral changes to the contract, or a company reserving the right to remove your content without notice. It compares the model against a classical TF-IDF plus Logistic Regression baseline, benchmarks ONNX export with INT8 quantization for CPU inference, and explains each flag with the words that actually drove it, via a gradient-based attribution method implemented from scratch.

## Dataset

The data comes from [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue), config `unfair_tos`, which is part of the [LexGLUE benchmark](https://arxiv.org/abs/2110.00976). It's built from real Terms of Service documents (Spotify, Facebook, Tinder, and others) and annotated by legal researchers. There are 5,532 training clauses and 1,607 test clauses. Unlike the sentiment datasets this project started out with, these labels come from actual legal experts instead of being scraped or auto-generated.

It's a multi-label problem. A single clause can match zero, one, or several of 8 unfair categories at once, and a clause matching none of them is just considered fair:

Limitation of liability, Unilateral termination, Unilateral change, Content removal, Contract by using, Choice of law, Jurisdiction, Arbitration

It's also heavily imbalanced. Most clauses in the dataset are fair, and within the test set, the number of examples per category ranges from 38 (Limitation of liability) down to only 7 (Arbitration).

## Results

| Model | F1 (micro) | F1 (macro) |
|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression (class_weight="balanced") | 0.593 | 0.600 |
| DistilBERT (unweighted loss) | 0.620 | 0.454 |
| DistilBERT (raw inverse-frequency pos_weight) | 0.571 | 0.555 |
| DistilBERT (sqrt-dampened pos_weight) | 0.624 | 0.602 |

Micro F1 adds up true and false positives across all 8 categories before computing a single F1 score. It's the metric LexGLUE itself reports, and it's dominated by whichever categories are most common. Macro F1 instead averages each category's F1 equally, so it's much more sensitive to how the model handles the rarest categories.

Here's what actually happened while building this, in order:

The baseline used `class_weight="balanced"` in scikit-learn's LogisticRegression from the start, because ignoring class imbalance in a linear model is a pretty basic mistake to avoid. DistilBERT's default multi-label loss, BCEWithLogitsLoss with no weighting, got no such correction, and it showed. DistilBERT beat the baseline on micro F1, but its macro F1 collapsed to 0.454, well below the baseline's 0.600. It had essentially given up on the rarest categories, since they barely move the needle in an unweighted loss.

The standard fix for this is a `pos_weight` per category, calculated as the number of negative examples divided by the number of positive examples, inside BCEWithLogitsLoss. That's the multi-label version of `class_weight="balanced"`. Applying it directly overcorrected badly. For Arbitration, which only has about 24 positive examples in training, the weight worked out to over 200, and the model started flagging clauses as Arbitration roughly five times too often, with a precision of just 0.18 against 7 true positives in the test set. Both micro and macro F1 actually got worse, dropping to 0.571 and 0.555.

Dampening that weight with a square root fixed it. DistilBERT finally beat the baseline on both metrics, 0.624 and 0.602, though only by a narrow margin on macro F1. Looking at individual categories, it clearly wins on Limitation of liability and Contract by using, holds its own elsewhere, and is still noticeably weak on Arbitration, where precision sits at 0.16. That's better than the raw-weighted version, but the model is still flagging that category more than five times too often.

The real takeaway here isn't that the transformer wins. It's that a naive multi-label setup actively loses to a properly weighted linear baseline, and getting a fair comparison took two attempts at the same fix plus an honest look at where it was still falling short.

## Project structure

- `data.py` loads the dataset and builds the multi-hot label vectors
- `baseline.py` is the TF-IDF + One-vs-Rest Logistic Regression baseline
- `train.py` fine-tunes `distilbert-base-uncased` with a class-weighted multi-label loss and saves the result to `./sentiment-model`
- `serve.py` is a FastAPI endpoint that serves the fine-tuned model
- `explain.py` computes gradient-based token attributions for why a clause was flagged
- `optimize.py` handles ONNX export, INT8 quantization, and inference benchmarking
- `requirements.txt` lists the dependencies

## Setup

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA-enabled torch build instead of the default CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
python baseline.py   # writes baseline_metrics.txt
python train.py      # writes metrics.txt, saves model to ./sentiment-model
uvicorn serve:app --reload
```

To test the API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "You agree to resolve any dispute through binding arbitration and waive your right to a jury trial."}'
```

Which returns something like this:

```json
{
  "is_unfair": true,
  "categories": [
    {"category": "Limitation of liability", "probability": 0.02, "flagged": false},
    {"category": "Unilateral termination", "probability": 0.01, "flagged": false},
    {"category": "Unilateral change", "probability": 0.03, "flagged": false},
    {"category": "Content removal", "probability": 0.01, "flagged": false},
    {"category": "Contract by using", "probability": 0.05, "flagged": false},
    {"category": "Choice of law", "probability": 0.04, "flagged": false},
    {"category": "Jurisdiction", "probability": 0.06, "flagged": false},
    {"category": "Arbitration", "probability": 0.91, "flagged": true}
  ],
  "explanations": {
    "Arbitration": [
      {"token": "you", "score": 1.0},
      {"token": "dispute", "score": -0.91},
      {"token": "arbitration", "score": 0.41},
      {"token": "agree", "score": 0.32},
      {"token": "binding", "score": 0.18}
    ]
  }
}
```

The per-category flag threshold is a runtime environment variable (`FLAG_THRESHOLD`, default 0.5) instead of a hardcoded value. How aggressively you want to flag clauses depends on whether missed unfair clauses or false alarms matter more for your use case, and this repo doesn't have a real downstream use case to optimize that against, so it's left configurable.

## Explainability

`/predict` also returns an `explanations` entry for every *flagged* category (not all 8 — most clauses flag zero or one, and the attribution pass is a full extra forward+backward through the model per category, so computing it for categories nobody flagged would be pure waste).

The method is gradient x input: backprop the flagged category's logit to the input embeddings, then take the dot product of each token's gradient with its own embedding. That's a direct measure of how much nudging a token further in the direction it's already pointing would move the score — a simple, long-standing saliency method (the input-gradient family, going back to Simonyan et al.'s saliency maps), implemented here directly on top of `transformers`/`torch` rather than pulling in Captum or SHAP for something a single extra backward pass already gives you. WordPiece subwords (`arbitra` + `##tion`) are merged back into whole words by averaging their scores, and every score is normalized by the single largest `|score|` in the clause, so 1.0 (or -1.0) always marks whichever word mattered most, positive meaning it pushed the flag up and negative meaning it pushed against it.

Full output for the example clause above, ranked by magnitude:

```
        you  +1.00
    dispute  -0.91
    resolve  -0.46
arbitration  +0.41
      agree  +0.32
         to  +0.27
        and  -0.25
    through  -0.22
    binding  +0.18
      right  -0.17
      trial  -0.09
```

Worth being honest about: this is the noisier end of attribution methods. "arbitration" and "binding" do score positive, as expected, but neither is the top-ranked token — a plain "you" is, and "dispute" scores almost as large but negative, i.e. the method claims the word most central to the whole sentence was *arguing against* the Arbitration flag. That's a known weakness of raw gradient x input on transformers; integrated gradients (averaging the gradient along a path from a baseline input instead of taking it at one point) is the standard fix, at the cost of needing dozens of forward/backward passes per explanation instead of one. Shipping the simpler, noisier method and saying so here is the same call this project made on the Arbitration precision problem: get something real in, then be specific about where it still falls short instead of implying it doesn't.

## Limitations and future work

Arbitration is still the weak category. Its precision sits at 0.16, meaning the model flags roughly five clauses as Arbitration for every one that actually belongs there, even after the square-root pos_weight dampening described above. The remaining fix probably isn't another loss-weighting pass — it's likely that Arbitration's probability distribution sits differently than the other seven categories after that correction, so a single global 0.5 cutoff doesn't suit it. Tuning a per-category threshold (or calibrating each category's probabilities separately before thresholding) is the natural next step, and it's one this repo hasn't tried yet.

The explanation method has its own known weakness, covered in detail above: raw gradient x input is noisier than integrated gradients, and it shows on real examples. That's a deliberate scope choice (one backward pass instead of dozens), not an oversight, but it's the first thing to upgrade if the explanations need to be trustworthy rather than just directional.

Test coverage is also uneven. `tests/` covers the serving layer and the explanation logic, both exercised against the real trained model, but `data.py`, `baseline.py`, `train.py`, and `optimize.py` have no tests at all, and CI only runs `ruff check`, not the test suite itself. Running the real pytest suite in CI would need the trained model artifact available to the runner, which is the main reason it isn't wired up yet.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

You'll need a trained model at `./sentiment-model` first (run `train.py`), since `serve.py` loads it as soon as it's imported.

## Docker

The image doesn't bake in the model weights. Instead, mount your trained `./sentiment-model` directory at runtime:

```bash
docker build -t unfair-tos-classifier .
docker run -p 8000:8000 -v "$(pwd)/sentiment-model:/app/sentiment-model" unfair-tos-classifier
```

## Inference optimization

```bash
pip install -r requirements-dev.txt
python optimize.py
```

This exports the fine-tuned model to ONNX and applies dynamic INT8 quantization, then benchmarks PyTorch fp32, ONNX fp32, and ONNX int8 against each other on F1, on-disk size, and CPU latency. The full results and methodology notes, including a benchmarking pitfall around ONNX Runtime's default multi-threading, are in [optimization_results.md](optimization_results.md).

## License

MIT, see [LICENSE](LICENSE).
