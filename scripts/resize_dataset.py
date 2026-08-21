#!/usr/bin/env python3
"""
Dataset Image Resizer for YOLO Object Detection & Instance Segmentation
========================================================================
Resizes entire YOLO datasets (train, valid, test) to target dimensions (e.g. 640x640,
1280x1280, 416x416) with accurate bounding box and polygon coordinate transformation.

Features:
  - Multi-size batch export (e.g., generate 640, 1280, and 416 versions simultaneously)
  - Letterbox padding mode (YOLO standard with gray 114 background) with bbox/polygon adjustment
  - Direct stretch / max-edge aspect preservation modes
  - Multi-threaded fast image processing
  - Auto-updates data.yaml if present

Usage:
  python scripts/resize_dataset.py --input ./dataset_default --size 640
  python scripts/resize_dataset.py --input ./dataset_default --sizes 640 1280 --mode letterbox
  python scripts/resize_dataset.py --input ./dataset_default --width 640 --height 640 --mode stretch
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Supported image formats
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def get_resample_filter():
    """Get high quality Pillow resampling filter across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    elif hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    return Image.BICUBIC


def resize_image_letterbox(img, target_w, target_h, fill_color=(114, 114, 114)):
    """
    Resize image to fit within target_w x target_h, preserving aspect ratio and
    padding with neutral gray. Returns (new_img, scale, pad_x, pad_y, orig_w, orig_h).
    """
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resample = get_resample_filter()
    resized_img = img.resize((new_w, new_h), resample=resample)

    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0

    # Create target background canvas
    mode = "RGB" if img.mode in ("RGB", "L") else "RGBA"
    fill = fill_color if mode == "RGB" else (*fill_color, 255)
    canvas = Image.new(mode, (target_w, target_h), fill)
    canvas.paste(resized_img, (int(round(pad_x)), int(round(pad_y))))

    return canvas, scale, pad_x, pad_y, orig_w, orig_h


def transform_yolo_letterbox(lines, scale, pad_x, pad_y, orig_w, orig_h, target_w, target_h):
    """
    Transform YOLO bounding box and polygon coordinates for a letterboxed image.
    All inputs and outputs are normalized (0.0 to 1.0) according to respective dimensions.
    """
    transformed = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        cls = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        if len(coords) == 4:
            # Standard YOLO Bounding Box: [cx, cy, w, h] (normalized)
            cx_orig, cy_orig, w_orig, h_orig = coords
            
            # Convert normalized original to pixel coordinates
            px_cx = cx_orig * orig_w
            px_cy = cy_orig * orig_h
            px_w  = w_orig * orig_w
            px_h  = h_orig * orig_h
            
            # Apply scale and padding
            new_px_cx = (px_cx * scale) + pad_x
            new_px_cy = (px_cy * scale) + pad_y
            new_px_w  = px_w * scale
            new_px_h  = px_h * scale
            
            # Normalize to target dimensions
            new_cx = max(0.0, min(1.0, new_px_cx / target_w))
            new_cy = max(0.0, min(1.0, new_px_cy / target_h))
            new_w  = max(0.0, min(1.0, new_px_w / target_w))
            new_h  = max(0.0, min(1.0, new_px_h / target_h))
            
            transformed.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
            
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # YOLO Polygon Segmentation: [x1, y1, x2, y2, ...] (normalized)
            new_pts = []
            for i in range(0, len(coords), 2):
                px = coords[i] * orig_w
                py = coords[i + 1] * orig_h
                
                new_px = (px * scale) + pad_x
                new_py = (py * scale) + pad_y
                
                new_norm_x = max(0.0, min(1.0, new_px / target_w))
                new_norm_y = max(0.0, min(1.0, new_py / target_h))
                new_pts.extend([f"{new_norm_x:.6f}", f"{new_norm_y:.6f}"])
                
            transformed.append(f"{cls} " + " ".join(new_pts) + "\n")
        else:
            # Unknown or classification format, keep as is
            transformed.append(line)
            
    return transformed


def process_single_image(img_path, lbl_path, out_img_path, out_lbl_path, target_w, target_h, mode="letterbox", quality=95):
    """Process and save a single image and its corresponding label file."""
    try:
        with Image.open(img_path) as img:
            img_format = img.format or "JPEG"
            
            if mode == "letterbox":
                new_img, scale, pad_x, pad_y, orig_w, orig_h = resize_image_letterbox(
                    img, target_w, target_h
                )
                
                # Save resized image
                os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
                if out_img_path.lower().endswith(('.jpg', '.jpeg')):
                    new_img.convert("RGB").save(out_img_path, format="JPEG", quality=quality)
                else:
                    new_img.save(out_img_path, quality=quality)
                
                # Transform label if exists
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    new_lines = transform_yolo_letterbox(
                        lines, scale, pad_x, pad_y, orig_w, orig_h, target_w, target_h
                    )
                    
                    os.makedirs(os.path.dirname(out_lbl_path), exist_ok=True)
                    with open(out_lbl_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                return True
                
            elif mode == "stretch":
                # Direct resize (normalized YOLO coords remain identical)
                resample = get_resample_filter()
                new_img = img.resize((target_w, target_h), resample=resample)
                
                os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
                if out_img_path.lower().endswith(('.jpg', '.jpeg')):
                    new_img.convert("RGB").save(out_img_path, format="JPEG", quality=quality)
                else:
                    new_img.save(out_img_path, quality=quality)
                
                if os.path.exists(lbl_path):
                    os.makedirs(os.path.dirname(out_lbl_path), exist_ok=True)
                    shutil.copy2(lbl_path, out_lbl_path)
                return True
                
            elif mode == "max_edge":
                orig_w, orig_h = img.size
                scale = min(target_w / orig_w, target_h / orig_h)
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                
                resample = get_resample_filter()
                new_img = img.resize((new_w, new_h), resample=resample)
                
                os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
                if out_img_path.lower().endswith(('.jpg', '.jpeg')):
                    new_img.convert("RGB").save(out_img_path, format="JPEG", quality=quality)
                else:
                    new_img.save(out_img_path, quality=quality)
                
                if os.path.exists(lbl_path):
                    os.makedirs(os.path.dirname(out_lbl_path), exist_ok=True)
                    shutil.copy2(lbl_path, out_lbl_path)
                return True
                
    except Exception as e:
        print(f"[ERROR] Failed to process {img_path}: {e}")
        return False


def resize_dataset(input_dir, output_dir, target_w, target_h, mode="letterbox", quality=95, workers=8):
    """Resize an entire YOLO dataset directory structure to target dimensions."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    if not input_path.exists():
        print(f"❌ Error: Input dataset directory '{input_path}' does not exist.")
        return False
        
    print("=" * 64)
    print(f"🚀 YOLO Dataset Resizer")
    print(f"   Input Directory : {input_path}")
    print(f"   Output Directory: {output_path}")
    print(f"   Target Size     : {target_w}x{target_h}")
    print(f"   Resize Mode     : {mode.upper()}")
    print(f"   JPEG Quality    : {quality}")
    print(f"   Worker Threads  : {workers}")
    print("=" * 64)
    
    tasks = []
    splits = ["train", "valid", "test"]
    
    # Also support flat datasets without split folders
    detected_splits = []
    for split in splits:
        if (input_path / split / "images").exists() or (input_path / split).exists():
            detected_splits.append(split)
            
    if not detected_splits:
        # Check if root contains images directly
        root_images = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if root_images:
            detected_splits = ["."]
            
    if not detected_splits:
        print(f"⚠️ No image folders (train, valid, test) found in '{input_path}'.")
        return False
        
    total_images = 0
    for split in detected_splits:
        if split == ".":
            img_dir = input_path
            lbl_dir = input_path
            out_img_dir = output_path
            out_lbl_dir = output_path
        else:
            img_dir = input_path / split / "images"
            if not img_dir.exists():
                img_dir = input_path / split
            lbl_dir = input_path / split / "labels"
            
            out_img_dir = output_path / split / "images"
            out_lbl_dir = output_path / split / "labels"
            
        if not img_dir.exists():
            continue
            
        for file in img_dir.iterdir():
            if file.is_file() and file.suffix.lower() in IMG_EXTS:
                total_images += 1
                base_name = file.stem
                lbl_file = lbl_dir / f"{base_name}.txt"
                out_img_file = out_img_dir / file.name
                out_lbl_file = out_lbl_dir / f"{base_name}.txt"
                
                tasks.append((
                    str(file), str(lbl_file),
                    str(out_img_file), str(out_lbl_file),
                    target_w, target_h, mode, quality
                ))

    print(f"📦 Found {total_images} images across {len(detected_splits)} split(s). Starting conversion...")
    
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, *t) for t in tasks]
        for future in as_completed(futures):
            if future.result():
                completed += 1
            if completed % 50 == 0 or completed == total_images:
                pct = (completed / max(1, total_images)) * 100
                print(f"   [{pct:5.1f}%] Processed {completed}/{total_images} images...", end="\r", flush=True)

    print(f"\n✅ Completed: {completed}/{total_images} images resized successfully.")

    # Copy data.yaml if present and adjust paths
    yaml_src = input_path / "data.yaml"
    if yaml_src.exists():
        yaml_dst = output_path / "data.yaml"
        shutil.copy2(yaml_src, yaml_dst)
        print(f"📄 Copied data.yaml -> {yaml_dst}")

    # Copy classes/metadata if present
    for extra in ["classes.txt", "notes.json", "state.db"]:
        extra_src = input_path / extra
        if extra_src.exists():
            shutil.copy2(extra_src, output_path / extra)
            print(f"💾 Copied {extra} -> {output_path / extra}")

    print(f"🎉 Dataset saved to: {output_path}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Resize YOLO datasets to standard dimensions (640x640, 1280x1280, etc.) with annotation scaling."
    )
    parser.add_argument(
        "-i", "--input", "--input-dir",
        dest="input_dir",
        default="./dataset_default",
        help="Path to source YOLO dataset directory (default: ./dataset_default)"
    )
    parser.add_argument(
        "-o", "--output", "--output-dir",
        dest="output_dir",
        default=None,
        help="Destination directory for resized dataset (default: <input_dir>_<size>)"
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=640,
        help="Square target size (e.g. 640, 1280, 416). Default: 640"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Generate multiple dataset versions for different sizes (e.g. --sizes 640 1280 416)"
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=None,
        help="Explicit target width (overrides --size)"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=None,
        help="Explicit target height (overrides --size)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["letterbox", "stretch", "max_edge"],
        default="letterbox",
        help="Resize mode: 'letterbox' (pads to square with 114 gray), 'stretch' (direct resize), 'max_edge' (aspect ratio preserved). Default: letterbox"
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        help="JPEG output quality (1-100). Default: 95"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads. Default: 8"
    )

    args = parser.parse_args()

    # Determine input directory
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        # Fallback check
        alt_input = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_default")
        if os.path.exists(alt_input):
            input_dir = alt_input

    # Multi-size execution
    if args.sizes:
        for s in args.sizes:
            out_dir = args.output_dir or f"{input_dir}_{s}"
            resize_dataset(
                input_dir=input_dir,
                output_dir=out_dir,
                target_w=s,
                target_h=s,
                mode=args.mode,
                quality=args.quality,
                workers=args.workers
            )
    else:
        target_w = args.width or args.size
        target_h = args.height or args.size
        out_dir = args.output_dir or f"{input_dir}_{target_w}"
        resize_dataset(
            input_dir=input_dir,
            output_dir=out_dir,
            target_w=target_w,
            target_h=target_h,
            mode=args.mode,
            quality=args.quality,
            workers=args.workers
        )


if __name__ == "__main__":
    main()
