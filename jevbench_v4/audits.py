"""Reject lossy native serialization before local model inference."""
from .contracts import UnsupportedInput


def audit_von(tokenizer, request, max_length=512):
    """Check every premise/hypothesis pair used by the pinned default backend."""
    lengths = []
    for _, description in request.choices:
        hypothesis = f"{request.instructions} {description}"
        ids = tokenizer(request.state, hypothesis, truncation=False)["input_ids"]
        lengths.append(len(ids))
    if max(lengths) > max_length:
        raise UnsupportedInput("von_pair_exceeds_512_tokens")
    return {"representation": "von-default-pairwise", "max_pair_tokens": max(lengths),
            "choice_count": len(lengths), "truncated": False}


def audit_laya(tokenizer, request, config, build_sequence, render_options):
    """Compare native packed tokens with the complete, untrimmed representation.

    Counting option markers alone misses truncated descriptions and state text.
    The pinned SDK additionally caps each option at 48 tokens and the whole head.
    """
    q = {"t": "choice", "ins": request.instructions, "crit": dict(request.choices)}
    if any(tokenizer.mask_token in value for value in
           [request.state, request.instructions, *dict(request.choices).keys(), *dict(request.choices).values()]):
        raise UnsupportedInput("laya_reserved_mask_token_would_be_rewritten")
    head = tokenizer(f"choice question: {request.instructions}", add_special_tokens=False)["input_ids"]
    expected = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
    expected_markers = []
    for option in render_options(q):
        expected_markers.append(len(expected))
        expected += [tokenizer.mask_token_id] + tokenizer(" " + option, add_special_tokens=False)["input_ids"]
    expected += [tokenizer.sep_token_id]
    expected += tokenizer(request.state, add_special_tokens=False)["input_ids"] + [tokenizer.sep_token_id]
    actual, markers = build_sequence(tokenizer, request.state, q, config.get("max_len", 512),
                                    config.get("head_max_len", 192))
    if actual != expected or markers != expected_markers:
        raise UnsupportedInput("laya_native_packing_would_truncate_or_rewrite_input")
    return {"representation": "laya-native", "input_tokens": len(actual),
            "choice_count": len(markers), "truncated": False}
