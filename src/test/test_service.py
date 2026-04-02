import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from app import service, Request, Response

client = TestClient(service)


class TestInferEndpoint:
    """Test cases for the /infer endpoint"""

    @patch("app.model")
    @patch("app.y_scaler")
    @patch("app.x_scaler")
    def test_infer_success(self, mock_x_scaler, mock_y_scaler, mock_model):
        """Test successful inference with valid input"""
        # Setup mocks
        mock_x_scaler.transform.return_value = np.array([[0.5]])
        mock_model.predict.return_value = np.array([[0.3]])
        mock_y_scaler.inverse_transform.return_value = np.array([[10.0]])

        # Make request
        response = client.post("/infer", json={"quality": 0.5})

        # Assertions
        assert response.status_code == 200
        assert response.json() == {"speed": 10}
        mock_x_scaler.transform.assert_called_once_with([[0.5]])
        mock_model.predict.assert_called_once()
        mock_y_scaler.inverse_transform.assert_called_once()

    @patch("app.model")
    @patch("app.y_scaler")
    @patch("app.x_scaler")
    def test_infer_boundary_values(self, mock_x_scaler, mock_y_scaler, mock_model):
        """Test inference with boundary quality values"""
        # Setup mocks for minimum value
        mock_x_scaler.transform.return_value = np.array([[0.0]])
        mock_model.predict.return_value = np.array([[0.1]])
        mock_y_scaler.inverse_transform.return_value = np.array([[1.0]])

        response = client.post("/infer", json={"quality": 0.0})
        assert response.status_code == 200
        assert response.json() == {"speed": 1}

        # Setup mocks for maximum value
        mock_x_scaler.transform.return_value = np.array([[1.0]])
        mock_model.predict.return_value = np.array([[0.9]])
        mock_y_scaler.inverse_transform.return_value = np.array([[20.0]])

        response = client.post("/infer", json={"quality": 1.0})
        assert response.status_code == 200
        assert response.json() == {"speed": 20}

    def test_infer_invalid_quality_low(self):
        """Test inference with quality below minimum (0)"""
        response = client.post("/infer", json={"quality": -0.1})
        assert response.status_code == 422  # Validation error

    def test_infer_invalid_quality_high(self):
        """Test inference with quality above maximum (1)"""
        response = client.post("/infer", json={"quality": 1.1})
        assert response.status_code == 422  # Validation error

    def test_infer_missing_quality_field(self):
        """Test inference with missing quality field"""
        response = client.post("/infer", json={})
        assert response.status_code == 422  # Validation error

    def test_infer_invalid_data_type(self):
        """Test inference with invalid data type"""
        response = client.post("/infer", json={"quality": "invalid"})
        assert response.status_code == 422  # Validation error

    def test_infer_empty_request(self):
        """Test inference with empty request body"""
        response = client.post("/infer", json=None)
        assert response.status_code == 422  # Validation error

    @patch("app.model")
    @patch("app.y_scaler")
    @patch("app.x_scaler")
    def test_infer_model_prediction_rounding(
        self, mock_x_scaler, mock_y_scaler, mock_model
    ):
        """Test that speed is properly truncated to integer"""
        # Setup mocks for float result that needs truncation
        mock_x_scaler.transform.return_value = np.array([[0.5]])
        mock_model.predict.return_value = np.array([[0.3]])
        mock_y_scaler.inverse_transform.return_value = np.array(
            [[10.7]]
        )  # Should truncate to 10

        response = client.post("/infer", json={"quality": 0.5})
        assert response.status_code == 200
        assert response.json() == {"speed": 11}

    @patch("app.model")
    @patch("app.y_scaler")
    @patch("app.x_scaler")
    def test_infer_model_error_handling(self, mock_x_scaler, mock_y_scaler, mock_model):
        """Test error handling when model prediction fails"""
        # Setup mocks to raise an exception
        mock_x_scaler.transform.return_value = np.array([[0.5]])
        mock_model.predict.side_effect = Exception("Model prediction failed")

        with pytest.raises(Exception):
            client.post("/infer", json={"quality": 0.5})

    @patch("app.model")
    @patch("app.y_scaler")
    @patch("app.x_scaler")
    def test_infer_scaler_error_handling(
        self, mock_x_scaler, mock_y_scaler, mock_model
    ):
        """Test error handling when scaler transformation fails"""
        # Setup mocks to raise an exception
        mock_x_scaler.transform.side_effect = Exception("Scaler transformation failed")

        with pytest.raises(Exception):
            client.post("/infer", json={"quality": 0.5})


class TestRequestModel:
    """Test cases for the Request model validation"""

    def test_request_model_valid_values(self):
        """Test Request model with valid values"""
        request = Request(quality=0.5)
        assert request.quality == 0.5

        request = Request(quality=0.0)
        assert request.quality == 0.0

        request = Request(quality=1.0)
        assert request.quality == 1.0

    def test_request_model_invalid_values(self):
        """Test Request model with invalid values"""
        with pytest.raises(ValueError):
            Request(quality=-0.1)

        with pytest.raises(ValueError):
            Request(quality=1.1)


class TestResponseModel:
    """Test cases for the Response model validation"""

    def test_response_model_valid_values(self):
        """Test Response model with valid values"""
        response = Response(speed=10)
        assert response.speed == 10

        response = Response(speed=1)
        assert response.speed == 1

        response = Response(speed=20)
        assert response.speed == 20

    def test_response_model_invalid_values(self):
        """Test Response model with invalid values"""
        with pytest.raises(ValueError):
            Response(speed=0)

        with pytest.raises(ValueError):
            Response(speed=21)


class TestServiceMetadata:
    """Test cases for service metadata and configuration"""

    def test_service_title(self):
        """Test that service has correct title"""
        assert service.title == "Generator APIary"

    def test_service_description(self):
        """Test that service has description"""
        assert "model blueprints" in service.description
        assert "training" in service.description
        assert "inference" in service.description

    def test_endpoint_exists(self):
        """Test that /infer endpoint exists"""
        assert "/infer" in [route.path for route in service.routes]

    def test_endpoint_methods(self):
        """Test that /infer endpoint accepts POST method"""
        for route in service.routes:
            if route.path == "/infer":
                assert "POST" in route.methods
                break
