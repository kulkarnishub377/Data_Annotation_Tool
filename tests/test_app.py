import unittest
import sys
import os
import json

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, auto_detect_device, calculate_iou, get_device_info, validate_split, secure_path

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

    def test_get_device_info(self):
        """Test device info dictionary structure."""
        info = get_device_info()
        self.assertIn("cpu", info)
        self.assertIn("cuda", info)
        self.assertIn("mps", info)
        self.assertIn("selected", info)

    def test_calculate_iou(self):
        """Test IoU overlap calculation."""
        # Identical boxes -> IoU = 1.0
        box1 = [0.1, 0.1, 0.5, 0.5]
        box2 = [0.1, 0.1, 0.5, 0.5]
        self.assertAlmostEqual(calculate_iou(box1, box2), 1.0, places=4)

        # Completely disjoint boxes -> IoU = 0.0
        box3 = [0.6, 0.6, 0.9, 0.9]
        self.assertAlmostEqual(calculate_iou(box1, box3), 0.0, places=4)

        # Partial overlap
        box4 = [0.3, 0.3, 0.7, 0.7]
        iou = calculate_iou(box1, box4)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_validate_split(self):
        """Test dataset split name validation."""
        validate_split("train")
        validate_split("valid")
        validate_split("test")
        with self.assertRaises(ValueError):
            validate_split("unknown")

    def test_secure_path_traversal(self):
        """Test path traversal prevention."""
        base = "/fake/base/dir"
        
        # Valid path
        safe = secure_path(base, "images", "car.jpg")
        self.assertTrue(safe.startswith(os.path.abspath(base)))
        
        # Path traversal attack
        with self.assertRaises(ValueError):
            secure_path(base, "images", "../../../etc/passwd")

    def test_api_classes(self):
        """Test classes API returns class array."""
        res = self.app.get('/api/classes')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("classes", data)
        self.assertIsInstance(data["classes"], list)

    def test_api_models(self):
        """Test models API returns files and presets list."""
        res = self.app.get('/api/models')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("files", data)
        self.assertIn("presets", data)
        self.assertTrue(len(data["presets"]) > 0)

    def test_api_dataset_health(self):
        """Test dataset health endpoint responds with score structure."""
        res = self.app.get('/api/dataset_health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("score", data)
        self.assertIn("total_images", data)
        self.assertIn("total_boxes", data)
        self.assertIn("corrupt_images", data)
        self.assertIn("empty_images", data)
        self.assertIn("small_boxes", data)
        self.assertIn("oob_boxes", data)

if __name__ == '__main__':
    unittest.main()
