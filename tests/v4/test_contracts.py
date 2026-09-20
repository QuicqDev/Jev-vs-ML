import unittest

from jevbench_v4.contracts import DecisionRequest, digest, validate_answer
from tests.v4.helpers import body, request


class ContractTests(unittest.TestCase):
    def test_all_77_choices_survive_request_and_response(self):
        req = request(77)
        result = validate_answer(body(req, "C76"), req)
        self.assertEqual(result["prediction"], 76)
        self.assertEqual(len(result["probabilities"]), 77)
        self.assertEqual(len(req.questions["classification"]["criteria"]), 77)

    def test_choice_order_is_part_of_request_identity(self):
        req = request()
        reverse = DecisionRequest(req.state, req.instructions, tuple(reversed(req.choices)))
        self.assertNotEqual(digest(req.as_dict()), digest(reverse.as_dict()))

    def test_labels_and_confidence_are_not_used_as_probabilities(self):
        req = request()
        result = validate_answer(body(req, "C1"), req)
        self.assertEqual(result["probabilities"], [0, 1])
        self.assertNotIn("confidence", result)

    def test_missing_extra_and_invalid_probabilities_are_rejected(self):
        req = request()
        vectors = [{"C0": 1}, {"C0": 1, "C1": 0, "C2": 0},
                   {"C0": float("nan"), "C1": 0}, {"C0": -.1, "C1": 1.1},
                   {"C0": True, "C1": False}, {"C0": .1, "C1": .1}]
        for vector in vectors:
            with self.subTest(vector=vector):
                response = body(req)
                response["answers"]["classification"]["probabilities"] = vector
                with self.assertRaises(ValueError):
                    validate_answer(response, req)

    def test_small_sdk_rounding_error_is_normalized(self):
        req = request(77)
        response = body(req)
        response["answers"]["classification"]["probabilities"] = {key: .013 for key, _ in req.choices}
        self.assertAlmostEqual(sum(validate_answer(response, req)["probabilities"]), 1)

    def test_duplicate_choices_rejected(self):
        with self.assertRaises(ValueError):
            DecisionRequest("state", "question", (("x", "A"), ("x", "B")))
