import unittest

from jevbench_v4.iterative import CONDITIONS, generate, run_episode


class FirstChoiceClient:
    def call(self, request, context=None, use_cache=True):
        n = len(request.choices)
        return {"status": "ok", "prediction": 0, "probabilities": [1.0] + [0.0] * (n - 1),
                "latency_ms": 1.0, "cache_hit": False}


class IterativeTests(unittest.TestCase):
    def test_generation_is_deterministic_and_conditions_are_bounded(self):
        config = {"iterative_seed": 9, "iterative_train_episodes": 4,
                  "iterative_policy_episodes": 4, "iterative_test_episodes": 4}
        self.assertEqual(generate(config), generate(config))
        episode = {"episode_id": "case", "order_age_days": 2, "payment_captured": True,
                   "delivered": False, "initial_payment": None, "initial_delivery": None,
                   "correct_terminal": "refund"}
        for condition in CONDITIONS:
            result = run_episode(FirstChoiceClient(), episode, condition, max_steps=4)
            self.assertLessEqual(result["steps"], 4)
            self.assertIn(result["terminal"], ("refund", "deny", "escalate"))
        adaptive = run_episode(FirstChoiceClient(), episode, "adaptive_feedback", max_steps=4)
        self.assertTrue(adaptive["success"])
        self.assertEqual(adaptive["steps"], 3)
