import sqlite3
import re
import os

DB_PATH = r"d:\model_train\annotated_dataset\state.db"
OUTPUT_DIR = r"d:\model_train\annotated_dataset"

def cleanup_frames():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("SELECT id, name, split FROM images").fetchall()

    to_delete = []
    files_deleted = 0
    for r in rows:
        name = r["name"]
        split = r["split"]
        
        m = re.search(r'frame_(\d+)', name)
        if m:
            num = int(m.group(1))
            if 6601 <= num <= 8200:
                to_delete.append(r["id"])
                
                # Delete image
                img_path = os.path.join(OUTPUT_DIR, split, "images", name)
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        files_deleted += 1
                    except:
                        pass
                        
                # Delete label
                basename = os.path.splitext(name)[0]
                label_path = os.path.join(OUTPUT_DIR, split, "labels", f"{basename}.txt")
                if os.path.exists(label_path):
                    try:
                        os.remove(label_path)
                        files_deleted += 1
                    except:
                        pass

    if to_delete:
        chunk_size = 500
        for i in range(0, len(to_delete), chunk_size):
            chunk = to_delete[i:i + chunk_size]
            placeholders = ','.join('?' * len(chunk))
            conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", chunk)
            
        conn.commit()
        print(f"Successfully deleted {len(to_delete)} frames from the database.")
        print(f"Also deleted {files_deleted} physical image and label files from your drive.")
    else:
        print("No frames found in the range 6601 to 8200.")
        
    conn.close()

if __name__ == "__main__":
    cleanup_frames()
    input("Press Enter to exit...")
