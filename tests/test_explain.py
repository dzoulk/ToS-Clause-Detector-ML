"""Tests for token-attribution explanations. The integration tests need a
trained model at ./sentiment-model (run train.py first), same as test_serve.py.
"""
from explain import _merge_subwords, explain_clause
from serve import device, model, tokenizer


def test_merge_subwords_averages_and_strips_hash_marks():
    # "unfairness" split as WordPiece: un + ##fair + ##ness
    merged = _merge_subwords([("un", 0.9), ("##fair", 0.3), ("##ness", 0.0)])
    assert merged == [{"token": "unfairness", "score": 0.4}]


def test_merge_subwords_leaves_whole_words_alone():
    merged = _merge_subwords([("you", 0.5), ("agree", -0.2)])
    assert merged == [{"token": "you", "score": 0.5}, {"token": "agree", "score": -0.2}]


def test_explain_clause_returns_normalized_whole_words():
    inputs = tokenizer(
        "You agree that any dispute will be resolved through binding arbitration.",
        return_tensors="pt",
    ).to(device)

    result = explain_clause(inputs, category_idx=7, model=model, tokenizer=tokenizer)  # Arbitration

    assert len(result) > 0
    tokens = [entry["token"] for entry in result]
    assert all("##" not in tok for tok in tokens)  # subwords fully merged
    assert all(tok not in tokenizer.all_special_tokens for tok in tokens)  # no [CLS]/[SEP]
    assert max(abs(entry["score"]) for entry in result) == 1.0  # normalized to its own peak
    assert all(-1.0 <= entry["score"] <= 1.0 for entry in result)


def test_explain_clause_does_not_mutate_model_eval_mode():
    inputs = tokenizer("This is a short clause.", return_tensors="pt").to(device)
    explain_clause(inputs, category_idx=0, model=model, tokenizer=tokenizer)
    assert model.training is False  # still in eval mode after a backward pass


def test_repeated_calls_do_not_accumulate_stale_gradients():
    """backward() always adds to .grad rather than replacing it, so without
    the zero_grad() at the top of each call, a second call's gradients
    would be contaminated by the first's and the result would silently
    drift depending on call history instead of being a pure function of
    (inputs, category_idx)."""
    inputs = tokenizer("This is a short clause about binding arbitration.", return_tensors="pt").to(device)
    first = explain_clause(inputs, category_idx=7, model=model, tokenizer=tokenizer)
    second = explain_clause(inputs, category_idx=7, model=model, tokenizer=tokenizer)
    assert first == second
