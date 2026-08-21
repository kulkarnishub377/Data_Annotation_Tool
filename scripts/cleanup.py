#!/usr/bin/env python3
"""
Dataset Frame Cleanup Utility
=============================
Deletes specific frame ranges or corrupted entries from the SQLite state.db
and cleans up corresponding physical image and label files from disk.

Usage:
  python scripts/cleanup.py --dir ./dataset_default --start 6601 --end 8200
"""

import sqlite3
import re
import os
import argparse
from pathlib import Path

def cleanup_frames(dataset_dir, start_num, end_num):
    dataset_path = Path(dataset_dir).resolve()
    db_path = dataset_path / "state.db"
    
    if not db_path.exists():
        print(f"❌ Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("SELECT id, name, split FROM images").fetchall()

    to_delete = []
    files_deleted = 0
    for r in rows:
        name = r["name"]
        split = r["split"]
        
        m = re.search(r'(\d+)', name)
        if m:
            num = int(m.group(1))
            if start_num <= num <= end_num:
                to_delete.append(r["id"])
                
                # Delete image file
                img_path = dataset_path / split / "images" / name
                if img_path.exists():
                    try:
                        os.remove(str(img_path))
                        files_deleted += 1
                    except Exception:
                        pass
                        
                # Delete label file
                label_path = dataset_path / split / "labels" / f"{Path(name).stem}.txt"
                if label_path.exists():
                    try:
                        os.remove(str(label_path))
                        files_deleted += 1
                    except Exception:
                        pass

    if to_delete:
        chunk_size = 500
        for i in range(0, len(to_delete), chunk_size):
            chunk = to_delete[i:i + chunk_size]
            placeholders = ','.join('?' * len(chunk))
            conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", chunk)
            
        conn.commit()
        print(f"✅ Successfully deleted {len(to_delete)} records from the database.")
        print(f"🗑️ Deleted {files_deleted} physical image/label files.")
    else:
        print(f"ℹ️ No frames found matching range {start_num} to {end_num}.")
        
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Cleanup specific frame ranges from YOLO dataset and SQLite state.")
    parser.add_argument("-d", "--dir", default="./dataset_default", help="Path to dataset directory (default: ./dataset_default)")
    parser.add_argument("--start", type=int, default=6601, help="Start frame index to remove")
    parser.add_argument("--end", type=int, default=8200, help="End frame index to remove")
    args = parser.parse_args()
    
    cleanup_frames(args.dir, args.start, args.end)


if __name__ == "__main__":
    main()
