#!/usr/bin/env python3
"""
Automated Standalone Desktop Executable Builder for Data Annotation Studio.
Uses PyInstaller and DataAnnotationTool.spec to generate a self-contained distribution.
"""

import os
import sys
import shutil
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec_path = os.path.join(root_dir, "DataAnnotationTool.spec")
    dist_dir = os.path.join(root_dir, "dist")
    build_dir = os.path.join(root_dir, "build")
    
    print("=" * 64)
    print("[*] Data Annotation Studio - Standalone Desktop Executable Builder")
    print(f"    Root Directory : {root_dir}")
    print(f"    Spec File      : {spec_path}")
    print(f"    Output Dist    : {dist_dir}")
    print("=" * 64)
    
    if not os.path.exists(spec_path):
        print(f"[ERROR] Spec file not found at '{spec_path}'")
        sys.exit(1)
        
    pyinstaller_exe = shutil.which("pyinstaller")
    if not pyinstaller_exe:
        # Check in current python environment
        venv_pyinstaller = os.path.join(os.path.dirname(sys.executable), "pyinstaller.exe" if sys.platform == "win32" else "pyinstaller")
        if os.path.exists(venv_pyinstaller):
            pyinstaller_exe = venv_pyinstaller
        else:
            print("[ERROR] PyInstaller is not installed. Please run: pip install pyinstaller")
            sys.exit(1)
            
    print(f"[*] Using PyInstaller: {pyinstaller_exe}")
    print("[*] Compiling standalone package...")
    
    cmd = [pyinstaller_exe, "--noconfirm", spec_path]
    result = subprocess.run(cmd, cwd=root_dir)
    
    if result.returncode == 0:
        out_exe = os.path.join(dist_dir, "DataAnnotationStudio", "DataAnnotationStudio.exe" if sys.platform == "win32" else "DataAnnotationStudio")
        print("=" * 64)
        print("[OK] Standalone Desktop Package built successfully!")
        print(f"     Executable: {out_exe}")
        print("=" * 64)
    else:
        print("[ERROR] Build failed with exit code", result.returncode)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
