import unittest
import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, auto_detect_device

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        """Test that the main index route returns a 200 OK status."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_auto_detect_device(self):
        """Test that auto detect device returns a valid device string."""
        device = auto_detect_device()
        self.assertIn(device, ['cpu', 'mps', '0', '1', '2', '3'])

if __name__ == '__main__':
    unittest.main()
