"""
Quick file preparation script for Google Colab upload
Run this locally before uploading to organize files
"""

import os
import shutil
from pathlib import Path

def prepare_colab_package():
    """Prepare all files needed for Colab in a single directory"""

    # Create package directory
    package_dir = Path("colab_package")
    package_dir.mkdir(exist_ok=True)

    print("Preparing Colab package...")
    print("=" * 60)

    # Required files
    required_files = [
        "table2graph_sem.py",
        "gcn_conv.py",
        "train_mimic_colab.py",
        "MIMIC_Training_Colab.ipynb",
        "COLAB_TRAINING_README.md"
    ]

    # Copy files
    for file in required_files:
        if os.path.exists(file):
            shutil.copy(file, package_dir / file)
            print(f"✓ Copied {file}")
        else:
            print(f"✗ Missing {file}")

    # Copy hosp directory
    if os.path.exists("hosp"):
        hosp_dest = package_dir / "hosp"
        if hosp_dest.exists():
            shutil.rmtree(hosp_dest)
        shutil.copytree("hosp", hosp_dest)
        csv_count = len(list(hosp_dest.glob("*.csv")))
        print(f"✓ Copied hosp/ with {csv_count} CSV files")
    else:
        print("✗ Missing hosp/ directory")

    print("\n" + "=" * 60)
    print(f"✓ Package ready in: {package_dir.absolute()}")
    print("\nNext steps:")
    print("1. Compress colab_package/ folder to ZIP")
    print("2. Upload to Google Drive")
    print("3. Extract in Colab")
    print("4. Run MIMIC_Training_Colab.ipynb")

if __name__ == "__main__":
    prepare_colab_package()
