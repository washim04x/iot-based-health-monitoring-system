"""
Test suite for heart disease prediction model.
Tests model loading, signature, and performance comparison with production model.
"""
import unittest
import mlflow
import os
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

class TestModelPerformance(unittest.TestCase):
    """Test model performance and compare with production model."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and load models."""
        # Set up DagsHub credentials for MLflow tracking
        # Try both IOT_DAGSHUB_KEY and iot-dagshub-key environment variables
        dagshub_key = os.getenv("IOT_DAGSHUB_KEY")
        if not dagshub_key:
            raise ValueError("IOT_DAGSHUB_KEY environment variable not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key

        mlflow.set_tracking_uri("https://dagshub.com/washim04x/iot-based-health-monitoring-system.mlflow")
        
        cls.client = mlflow.MlflowClient()
        
        # Load test data
        cls.test_data = pd.read_csv('data/processed/test.csv')
        
        # Load staging model and transformer (new model to be tested)
        # Try local files first (for CI/CD), then fall back to MLflow registry
        import joblib
        local_model_path = 'models/model.joblib'
        local_transformer_path = 'models/build_features_transformer.joblib'
        
        if os.path.exists(local_model_path) and os.path.exists(local_transformer_path):
            # Use local models (faster and more reliable for testing)
            try:
                cls.staging_model = joblib.load(local_model_path)
                cls.staging_transformer = joblib.load(local_transformer_path)
                cls.staging_model_version = "local"
                print(f"\n✓ Loaded Staging model from local files")
            except Exception as e:
                print(f"\n✗ Error loading local Staging model: {e}")
                raise
        else:
            # Fall back to MLflow registry
            try:
                cls.staging_model_version = cls.get_latest_model_version("model", stage="Staging")
                cls.staging_transformer_version = cls.get_latest_model_version("transformer", stage="Staging")
                
                if cls.staging_model_version:
                    cls.staging_model = mlflow.sklearn.load_model(f'models:/model/{cls.staging_model_version}')
                    cls.staging_transformer = mlflow.sklearn.load_model(f'models:/transformer/{cls.staging_transformer_version}')
                    print(f"\n✓ Loaded Staging model version {cls.staging_model_version} from MLflow")
                else:
                    raise ValueError("No Staging model found")
            except Exception as e:
                print(f"\n✗ Error loading Staging model from MLflow: {e}")
                raise
        
        # Try to load production model (old model for comparison)
        try:
            cls.prod_model_version = cls.get_latest_model_version("model", stage="Production")
            cls.prod_transformer_version = cls.get_latest_model_version("transformer", stage="Production")
            
            if cls.prod_model_version:
                cls.prod_model = mlflow.sklearn.load_model(f'models:/model/{cls.prod_model_version}')
                cls.prod_transformer = mlflow.sklearn.load_model(f'models:/transformer/{cls.prod_transformer_version}')
                cls.has_production_model = True
                print(f"✓ Loaded Production model version {cls.prod_model_version}")
            else:
                cls.has_production_model = False
                print("ℹ No Production model found - this will be the first model promoted")
        except Exception as e:
            cls.has_production_model = False
            print(f"ℹ No Production model available for comparison: {e}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        """Get the latest version of a model in a specific stage."""
        client = mlflow.MlflowClient()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                latest_versions = client.get_latest_versions(model_name, stages=[stage])
                return latest_versions[0].version if latest_versions else None
        except Exception:
            return None

    def test_01_staging_model_loads(self):
        """Test that the staging model loads successfully."""
        self.assertIsNotNone(self.staging_model, "Staging model should load successfully")
        self.assertIsNotNone(self.staging_transformer, "Staging transformer should load successfully")
        print("\n✓ Test passed: Staging model and transformer loaded successfully")

    def test_02_model_signature(self):
        """Test that the model accepts correct input shape and produces valid output."""
        # Prepare test data
        X_test = self.test_data.drop(columns=['HeartDisease'])
        y_test = self.test_data['HeartDisease']
        
        # Transform features
        X_test_transformed = self.staging_transformer.transform(X_test)
        
        # Make predictions
        predictions = self.staging_model.predict(X_test_transformed)
        
        # Verify output shape
        self.assertEqual(len(predictions), len(y_test), "Prediction length should match test data length")
        self.assertTrue(all(p in [0, 1] for p in predictions), "Predictions should be binary (0 or 1)")
        print(f"\n✓ Test passed: Model signature is valid (predicted {len(predictions)} samples)")

    def test_03_model_performance_comparison(self):
        """
        Compare staging model performance with production model.
        Test passes if staging model is better OR if no production model exists.
        """
        # Prepare test data
        X_test = self.test_data.drop(columns=['HeartDisease'])
        y_test = self.test_data['HeartDisease']
        
        # Evaluate staging model
        X_test_transformed_staging = self.staging_transformer.transform(X_test)
        y_pred_staging = self.staging_model.predict(X_test_transformed_staging)
        
        staging_accuracy = accuracy_score(y_test, y_pred_staging)
        staging_roc_auc = roc_auc_score(y_test, y_pred_staging)
        
        print(f"\n{'='*60}")
        print(f"STAGING MODEL PERFORMANCE (Version {self.staging_model_version}):")
        print(f"  • Accuracy:  {staging_accuracy:.4f}")
        print(f"  • ROC AUC:   {staging_roc_auc:.4f}")
        
        # If no production model, staging model should pass
        if not self.has_production_model:
            print(f"\n✓ No production model exists - this will be the first promotion")
            print(f"{'='*60}\n")
            self.assertTrue(True, "First model - automatic pass")
            return
        
        # Evaluate production model
        X_test_transformed_prod = self.prod_transformer.transform(X_test)
        y_pred_prod = self.prod_model.predict(X_test_transformed_prod)
        
        prod_accuracy = accuracy_score(y_test, y_pred_prod)
        prod_roc_auc = roc_auc_score(y_test, y_pred_prod)
        
        print(f"\nPRODUCTION MODEL PERFORMANCE (Version {self.prod_model_version}):")
        print(f"  • Accuracy:  {prod_accuracy:.4f}")
        print(f"  • ROC AUC:   {prod_roc_auc:.4f}")
        
        # Calculate improvements
        accuracy_improvement = staging_accuracy - prod_accuracy
        roc_auc_improvement = staging_roc_auc - prod_roc_auc
        
        print(f"\nIMPROVEMENT:")
        print(f"  • Accuracy:  {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)")
        print(f"  • ROC AUC:   {roc_auc_improvement:+.4f} ({roc_auc_improvement*100:+.2f}%)")
        
        # Model passes if EITHER accuracy OR roc_auc improves (or stays the same)
        model_is_better = (staging_accuracy >= prod_accuracy) or (staging_roc_auc >= prod_roc_auc)
        
        if model_is_better:
            print(f"\n✓ RESULT: Staging model is BETTER or EQUAL - promoting to Production")
            print(f"{'='*60}\n")
        else:
            print(f"\n✗ RESULT: Production model is BETTER - keeping current Production model")
            print(f"{'='*60}\n")
        
        # Assert that staging model is better
        self.assertTrue(
            model_is_better,
            f"\nStaging model performance is not better than Production model:\n"
            f"  Staging:    Accuracy={staging_accuracy:.4f}, ROC_AUC={staging_roc_auc:.4f}\n"
            f"  Production: Accuracy={prod_accuracy:.4f}, ROC_AUC={prod_roc_auc:.4f}\n"
            f"  The current Production model will be kept."
        )

    def test_04_minimum_performance_threshold(self):
        """Test that model meets minimum performance requirements."""
        # Prepare test data
        X_test = self.test_data.drop(columns=['HeartDisease'])
        y_test = self.test_data['HeartDisease']
        
        # Evaluate staging model
        X_test_transformed = self.staging_transformer.transform(X_test)
        y_pred = self.staging_model.predict(X_test_transformed)
        
        staging_accuracy = accuracy_score(y_test, y_pred)
        staging_roc_auc = roc_auc_score(y_test, y_pred)
        
        # Minimum thresholds
        min_accuracy = 0.50  # At least 50% accuracy
        min_roc_auc = 0.50   # At least 50% ROC AUC
        
        print(f"\n{'='*60}")
        print(f"MINIMUM THRESHOLD CHECK:")
        print(f"  • Accuracy:  {staging_accuracy:.4f} (minimum: {min_accuracy:.2f}) {'✓' if staging_accuracy >= min_accuracy else '✗'}")
        print(f"  • ROC AUC:   {staging_roc_auc:.4f} (minimum: {min_roc_auc:.2f}) {'✓' if staging_roc_auc >= min_roc_auc else '✗'}")
        print(f"{'='*60}\n")
        
        self.assertGreaterEqual(
            staging_accuracy, 
            min_accuracy, 
            f"Accuracy {staging_accuracy:.4f} is below minimum threshold {min_accuracy:.2f}"
        )
        self.assertGreaterEqual(
            staging_roc_auc, 
            min_roc_auc, 
            f"ROC AUC {staging_roc_auc:.4f} is below minimum threshold {min_roc_auc:.2f}"
        )

if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2, exit=True)
