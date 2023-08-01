

import unittest
import json
from app import app

class ModelTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        input_data = {
            "Subject Descriptor": [
            "Population"
            ],
            "values": [
                10
            ],
            "Continent": "Europe"
        }

        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)

        self.assertIn('predicted_gdp_as_at_2023', data)

        self.assertTrue(isinstance(data['predicted_gdp_as_at_2023'], (int, float)))

if __name__ == '__main__':
    unittest.main()
