import unittest

import numpy as np
import pandas as pd

from utils.modeling import get_cached_model_outputs, run_bias_audit_model


def make_synthetic_df(n_rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.integers(18, 90, size=n_rows)
    hypertension = rng.integers(0, 2, size=n_rows)
    heart_disease = rng.integers(0, 2, size=n_rows)
    glucose = rng.normal(110, 25, size=n_rows).clip(50, 250)
    bmi = rng.normal(28, 6, size=n_rows).clip(14, 60)

    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.5, 0.5])
    residence = rng.choice(["Urban", "Rural"], size=n_rows, p=[0.6, 0.4])
    ever_married = rng.choice(["Yes", "No"], size=n_rows)
    work_type = rng.choice(
        ["Private", "Govt_job", "Self-employed", "children"],
        size=n_rows,
    )
    smoking_status = rng.choice(
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
        size=n_rows,
    )

    score = (
        0.03 * age
        + 0.8 * hypertension
        + 0.9 * heart_disease
        + 0.005 * glucose
        + rng.normal(0, 0.6, size=n_rows)
    )
    threshold = float(np.quantile(score, 0.75))
    stroke = (score > threshold).astype(int)

    return pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking_status,
            "stroke": stroke,
        }
    )


class TestModelingUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.df = make_synthetic_df()

    def test_run_bias_audit_model_shape_and_metrics(self) -> None:
        result = run_bias_audit_model(
            self.df,
            "Logistic Regression",
            "gender",
        )

        self.assertIn("y_true", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIn("sensitive", result)
        self.assertIn("acc", result)
        self.assertIn("prec", result)
        self.assertIn("rec", result)

        self.assertEqual(len(result["y_true"]), len(result["y_pred"]))
        self.assertEqual(len(result["y_true"]), len(result["y_proba"]))
        self.assertEqual(len(result["y_true"]), len(result["sensitive"]))
        self.assertTrue(
            np.all((result["y_proba"] >= 0.0) & (result["y_proba"] <= 1.0))
        )

        self.assertGreaterEqual(result["acc"], 0.0)
        self.assertLessEqual(result["acc"], 1.0)
        self.assertGreaterEqual(result["prec"], 0.0)
        self.assertLessEqual(result["prec"], 1.0)
        self.assertGreaterEqual(result["rec"], 0.0)
        self.assertLessEqual(result["rec"], 1.0)

    def test_get_cached_model_outputs_structure(self) -> None:
        outputs = get_cached_model_outputs(
            self.df,
            sensitive_attr="Residence_type",
            test_size=0.25,
        )

        self.assertIn("y_test", outputs)
        self.assertIn("sensitive", outputs)
        self.assertIn("y_proba_by_model", outputs)

        self.assertIn("Logistic Regression", outputs["y_proba_by_model"])
        self.assertIn("Random Forest", outputs["y_proba_by_model"])

        y_test = outputs["y_test"]
        y_lr = outputs["y_proba_by_model"]["Logistic Regression"]
        y_rf = outputs["y_proba_by_model"]["Random Forest"]

        self.assertEqual(len(y_test), len(outputs["sensitive"]))
        self.assertEqual(len(y_test), len(y_lr))
        self.assertEqual(len(y_test), len(y_rf))

        self.assertTrue(np.all((y_lr >= 0.0) & (y_lr <= 1.0)))
        self.assertTrue(np.all((y_rf >= 0.0) & (y_rf <= 1.0)))

    def test_invalid_test_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_cached_model_outputs(
                self.df,
                sensitive_attr="gender",
                test_size=1.0,
            )


if __name__ == "__main__":
    unittest.main()
