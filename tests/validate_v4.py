"""Run the offline V4 suite with network access forbidden at the transport layer."""
from pathlib import Path
import unittest
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / "tests/v4"), top_level_dir=str(root))
    with patch("requests.sessions.Session.request", side_effect=AssertionError("V4 tests must stay offline")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
