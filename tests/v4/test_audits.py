import unittest

from jevbench_v4.audits import audit_laya, audit_von
from jevbench_v4.contracts import DecisionRequest, UnsupportedInput
from tests.v4.helpers import request


class Tokenizer:
    mask_token = "[MASK]"
    cls_token_id, sep_token_id, mask_token_id = 1, 2, 3

    def __call__(self, first, second=None, add_special_tokens=True, **kwargs):
        ids = [ord(c) + 10 for c in first]
        if second is not None:
            ids += [self.sep_token_id] + [ord(c) + 10 for c in second]
        return {"input_ids": ([self.cls_token_id] + ids + [self.sep_token_id]) if add_special_tokens else ids}


def render(q):
    return [f"{key}: {value}" for key, value in q["crit"].items()]


def complete_sequence(tok, state, q, *limits):
    seq = [tok.cls_token_id] + tok(f"choice question: {q['ins']}", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
    markers = []
    for option in render(q):
        markers.append(len(seq))
        seq += [tok.mask_token_id] + tok(" " + option, add_special_tokens=False)["input_ids"]
    seq += [tok.sep_token_id] + tok(state, add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
    return seq, markers


class AuditTests(unittest.TestCase):
    def test_von_checks_every_pair_without_truncation(self):
        req = request(77)
        report = audit_von(Tokenizer(), req)
        self.assertEqual(report["choice_count"], 77)
        with self.assertRaises(UnsupportedInput):
            audit_von(Tokenizer(), DecisionRequest("x" * 600, req.instructions, req.choices))

    def test_laya_complete_sequence_is_accepted(self):
        report = audit_laya(Tokenizer(), request(), {}, complete_sequence, render)
        self.assertEqual(report["choice_count"], 2)
        self.assertFalse(report["truncated"])

    def test_laya_option_text_loss_detected_when_every_marker_survives(self):
        def clipped(tok, state, q, *limits):
            seq, markers = complete_sequence(tok, state, q)
            # Simulate truncation inside the last description, preserving markers.
            seq.pop(markers[-1] + 2)
            return seq, markers
        with self.assertRaises(UnsupportedInput):
            audit_laya(Tokenizer(), request(), {}, clipped, render)

    def test_laya_state_loss_and_reserved_token_rewriting_rejected(self):
        def clipped(tok, state, q, *limits):
            seq, markers = complete_sequence(tok, state, q)
            return seq[:-4] + [tok.sep_token_id], markers
        with self.assertRaises(UnsupportedInput):
            audit_laya(Tokenizer(), request(), {}, clipped, render)
        req = request()
        with self.assertRaises(UnsupportedInput):
            audit_laya(Tokenizer(), DecisionRequest("literal [MASK]", req.instructions, req.choices),
                       {}, complete_sequence, render)
