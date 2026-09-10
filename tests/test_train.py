"""Basic check that the model still performs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import main


def test_accuracy_is_at_least_90_percent():
    accuracy = main()
    assert accuracy >= 0.9, f"accuracy dropped to {accuracy}"
