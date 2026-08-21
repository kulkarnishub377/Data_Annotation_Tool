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
    def test_resize_dataset_transforms(self):
        """Test letterbox coordinate transformation for bounding boxes."""
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from resize_dataset import transform_yolo_letterbox
        
        # Test original box at center of 1920x1080 resized to 640x640 letterbox
        lines = ["0 0.5 0.5 0.2 0.2\n"]
        orig_w, orig_h = 1920, 1080
        target_w, target_h = 640, 640
        scale = min(target_w / orig_w, target_h / orig_h) # 640/1920 = 0.3333
        pad_x = (target_w - (orig_w * scale)) / 2.0       # 0.0
        pad_y = (target_h - (orig_h * scale)) / 2.0       # 140.0
        
        transformed = transform_yolo_letterbox(
            lines, scale, pad_x, pad_y, orig_w, orig_h, target_w, target_h
        )
        self.assertEqual(len(transformed), 1)
        parts = transformed[0].split()
        self.assertEqual(parts[0], "0")
        self.assertAlmostEqual(float(parts[1]), 0.5, places=3) # Centered horizontally
        self.assertAlmostEqual(float(parts[2]), 0.5, places=3) # Centered vertically

    def test_compress_dataset(self):
        """Test multi-threaded dataset zip creation and verification."""
        import tempfile
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from compress_dataset import compress_dataset, verify_zip_archive
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "test_dataset")
            os.makedirs(os.path.join(src_dir, "train", "images"), exist_ok=True)
            with open(os.path.join(src_dir, "train", "images", "sample.txt"), "w") as f:
                f.write("test content")
                
            out_zip = os.path.join(tmpdir, "output.zip")
            success = compress_dataset(src_dir, out_zip, fmt="zip", threads=2, verify=True)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(out_zip))
            self.assertTrue(verify_zip_archive(out_zip))

if __name__ == '__main__':
    unittest.main()
