import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class UnifiedArchitectureTests(unittest.TestCase):
    def test_root_train_delegates_to_credless_trainer(self):
        content = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("from credless_model.train import train", content)

    def test_pipeline_no_longer_references_models_saved_artifacts(self):
        content = (ROOT / "pipeline" / "main_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("models/saved", content)
        self.assertIn("credless_model", content)

    def test_parallel_supervised_artifacts_removed(self):
        self.assertFalse((ROOT / "models" / "trainer.py").exists())
        self.assertFalse((ROOT / "models" / "saved" / "finverse_model.pkl").exists())
        self.assertFalse((ROOT / "models" / "saved" / "scaler.pkl").exists())


if __name__ == "__main__":
    unittest.main()
