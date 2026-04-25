import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib

from server.oracle import CredLessOracle


class DummyModel:
    def __init__(self, probability: float = 0.3):
        self.probability = probability

    def predict_proba(self, _rows):
        return [[1.0 - self.probability, self.probability]]


def _build_artifact(**overrides):
    artifact = {
        "model": DummyModel(),
        "model_name": "DummyModel",
        "metrics": {},
        "feature_names": [],
    }
    artifact.update(overrides)
    return artifact


def _write_artifact(path: Path, artifact):
    joblib.dump(artifact, path)


class OracleThresholdTests(unittest.TestCase):
    def _load_oracle(self, artifact):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        model_path = Path(temp_dir.name) / "model.pkl"
        _write_artifact(model_path, artifact)
        output = io.StringIO()
        with patch("server.oracle.MODEL_PATH", model_path):
            with redirect_stdout(output):
                oracle = CredLessOracle()
        return oracle, output.getvalue()

    def test_oracle_uses_nested_risk_thresholds(self):
        oracle, _output = self._load_oracle(
            _build_artifact(
                risk_thresholds={"low_risk": 0.22, "medium_risk": 0.37},
                threshold=0.99,
            )
        )

        self.assertEqual(oracle.low_thresh, 0.22)
        self.assertEqual(oracle.high_thresh, 0.37)

    def test_oracle_derives_thresholds_from_legacy_scalar(self):
        oracle, output = self._load_oracle(_build_artifact(threshold=0.36537155886084793))

        self.assertEqual(oracle.low_thresh, 0.2192)
        self.assertEqual(oracle.high_thresh, 0.3654)
        self.assertIn("Using legacy scalar threshold", output)

    def test_oracle_uses_defaults_when_thresholds_missing(self):
        oracle, output = self._load_oracle(_build_artifact())

        self.assertEqual(oracle.low_thresh, 0.40)
        self.assertEqual(oracle.high_thresh, 0.70)
        self.assertIn("Thresholds missing from artifact", output)

    def test_predict_uses_legacy_derived_thresholds(self):
        oracle, _output = self._load_oracle(
            _build_artifact(
                model=DummyModel(probability=0.30),
                threshold=0.36537155886084793,
                feature_names=[],
            )
        )

        result = oracle.predict({})

        self.assertEqual(result["decision"], "approve")
        self.assertEqual(result["tier"], "medium_risk")
        self.assertEqual(result["thresholds"], {"low_risk": 0.2192, "medium_risk": 0.3654})


if __name__ == "__main__":
    unittest.main()
