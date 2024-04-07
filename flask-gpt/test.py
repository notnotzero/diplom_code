import unittest
import json
from server import app
class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_generate_route(self):
        data = {'text': 'Hello', 'max_length': 50, 'model': 'gpt2'}
        response = self.app.post(
            '/generate',
            data=json.dumps(data),
            content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/json', response.content_type)
        response_data = json.loads(response.data)
        self.assertIn('generated_text', response_data)
        self.assertIsInstance(response_data['generated_text'], str)


if __name__ == '__main__':
    unittest.main()
