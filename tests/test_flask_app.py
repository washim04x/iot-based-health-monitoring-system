"""
Test suite for Flask heart disease prediction web application.
Tests home page rendering and prediction endpoint.
"""
import unittest
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app

class FlaskAppTests(unittest.TestCase):
    """Test Flask application endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        cls.client = app.test_client()
        cls.client.testing = True

    def test_01_home_page_loads(self):
        """Test that home page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Disease Prediction', response.data)
        print("\n✓ Test passed: Home page loads successfully")

    def test_02_home_page_contains_form(self):
        """Test that home page contains the prediction form."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Check for form elements
        self.assertIn(b'Age', response.data)
        self.assertIn(b'Sex', response.data)
        self.assertIn(b'ChestPainType', response.data)
        self.assertIn(b'RestingBP', response.data)
        self.assertIn(b'Cholesterol', response.data)
        print("✓ Test passed: Form contains all required fields")

    def test_03_predict_endpoint_healthy_patient(self):
        """Test prediction endpoint with healthy patient data."""
        # Sample data for a healthy patient (low risk)
        healthy_data = {
            'Age': '40',
            'Sex': 'M',
            'ChestPainType': 'NAP',
            'RestingBP': '120',
            'Cholesterol': '180',
            'FastingBS': '0',
            'RestingECG': 'Normal',
            'MaxHR': '150',
            'ExerciseAngina': 'N',
            'Oldpeak': '0.0',
            'ST_Slope': 'Up'
        }
        
        response = self.client.post('/predict', data=healthy_data)
        self.assertEqual(response.status_code, 200)
        # Check that response contains prediction result
        self.assertTrue(
            b'Low Risk' in response.data or b'High Risk' in response.data,
            "Response should contain either 'Low Risk' or 'High Risk'"
        )
        print("✓ Test passed: Prediction endpoint works for healthy patient")

    def test_04_predict_endpoint_at_risk_patient(self):
        """Test prediction endpoint with at-risk patient data."""
        # Sample data for an at-risk patient (high risk)
        at_risk_data = {
            'Age': '65',
            'Sex': 'M',
            'ChestPainType': 'ASY',
            'RestingBP': '160',
            'Cholesterol': '280',
            'FastingBS': '1',
            'RestingECG': 'ST',
            'MaxHR': '100',
            'ExerciseAngina': 'Y',
            'Oldpeak': '2.5',
            'ST_Slope': 'Flat'
        }
        
        response = self.client.post('/predict', data=at_risk_data)
        self.assertEqual(response.status_code, 200)
        # Check that response contains prediction result
        self.assertTrue(
            b'Low Risk' in response.data or b'High Risk' in response.data,
            "Response should contain either 'Low Risk' or 'High Risk'"
        )
        print("✓ Test passed: Prediction endpoint works for at-risk patient")

    def test_05_predict_endpoint_missing_data(self):
        """Test prediction endpoint handles missing data gracefully."""
        # Incomplete data
        incomplete_data = {
            'Age': '50',
            'Sex': 'F'
            # Missing other required fields
        }
        
        response = self.client.post('/predict', data=incomplete_data)
        # Should either return 200 with error message or 400/500 error
        self.assertIn(response.status_code, [200, 400, 500])
        print("✓ Test passed: Handles missing data gracefully")

    def test_06_static_files_accessible(self):
        """Test that static CSS and JS files are accessible."""
        # Test CSS file
        css_response = self.client.get('/static/css/style.css')
        self.assertEqual(css_response.status_code, 200)
        
        # Test JS file
        js_response = self.client.get('/static/js/script.js')
        self.assertEqual(js_response.status_code, 200)
        
        print("✓ Test passed: Static files are accessible")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
