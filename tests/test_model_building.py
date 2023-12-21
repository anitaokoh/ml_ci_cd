import pytest
from src.model_building import CaliforniaHousingModel
import os
import pickle


class TestCaliforniaHousingModel:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Fixture to set up the model before each test method."""
        self.model = CaliforniaHousingModel()
        self.model.load_and_split_data()
        self.model.train()

    def test_model_predicts(self):
        """Test that the model makes predictions"""
        predictions = self.model.model.predict(self.model.X_test)
        assert len(predictions) == len(self.model.y_test)

    def test_model_saves_file(self):
        """Test that the model saves to a file"""
        filename = 'test_trained_model.pkl'
        self.model.save_model(filename=filename)
        assert os.path.exists(filename)
        os.remove(filename)  # Clean up the test file after the test




