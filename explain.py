"""Token-level attribution for why a clause got flagged.

Uses gradient x input: backprop the target category's logit to the
input embeddings, then take the dot product of each token's gradient
with its own embedding. This measures how much nudging a token's
embedding further in the direction it's already pointing would move
that category's score - a long-standing, simple saliency method
(Simonyan et al.'s input-gradient family), not the newer feature-
attribution libraries (Captum/SHAP) which need an extra dependency for
something this single-pass computation already gives us directly.

Scores are signed: a positive score means that word pushed the flag
up, negative means it pushed back against it. They're normalized by
the single largest |score| in the clause, so 1.0 always marks
whichever word mattered most.
"""
from __future__ import annotations


def explain_clause(inputs: dict, category_idx: int, model, tokenizer) -> list[dict]:
    """inputs is the tokenizer output dict (input_ids, attention_mask)
    already moved to the model's device. Returns whole-word {token, score}
    entries, subwords merged back together, normalized so the single most
    influential word has |score| == 1.0."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    target_logit = outputs.logits[0, category_idx]
    target_logit.backward()

    token_scores = (inputs_embeds.grad[0] * inputs_embeds[0]).sum(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special = set(tokenizer.all_special_tokens)

    real = [
        (tok, score.item())
        for tok, score, mask in zip(tokens, token_scores, attention_mask[0])
        if mask.item() == 1 and tok not in special
    ]
    if not real:
        return []

    max_abs = max(abs(score) for _, score in real) or 1.0
    normalized = [(tok, score / max_abs) for tok, score in real]
    return _merge_subwords(normalized)


def _merge_subwords(token_scores: list[tuple[str, float]]) -> list[dict]:
    """WordPiece continuation pieces start with "##". Merge them back into
    the word they belong to, averaging (not summing) their scores so a
    word split into many subwords doesn't get an inflated score purely
    for having more pieces."""
    words: list[list] = []
    for tok, score in token_scores:
        if tok.startswith("##") and words:
            words[-1][0] += tok[2:]
            words[-1][1] += score
            words[-1][2] += 1
        else:
            words.append([tok, score, 1])
    return [{"token": word, "score": round(total / count, 4)} for word, total, count in words]
