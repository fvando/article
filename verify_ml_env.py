import sys
import importlib

packages = ["lightgbm", "pandas", "numpy"]
print("Verifying ML Dependencies:")
for p in packages:
    try:
        importlib.import_module(p)
        print(f"✅ {p} found.")
    except ImportError:
        print(f"❌ {p} NOT found.")
