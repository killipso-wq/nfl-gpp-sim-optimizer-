"""
Lightweight shim for Streamlit on Render when using app/streamlit_app.py path.

This enables Start Command to use:
    streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0

When run from the app/ subdirectory, this shim:
1. Ensures the repository root is on sys.path
2. Imports the root-level streamlit_app module as ui
3. Calls ui.main() to run the application
"""

import sys
import os
import importlib.util

# Add the repository root to sys.path so we can import from the root level
# This is needed because when running from app/, Python's module path doesn't include repo root
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import the root-level streamlit_app module directly by file path to avoid naming conflicts
try:
    root_streamlit_path = os.path.join(repo_root, "streamlit_app.py")
    spec = importlib.util.spec_from_file_location("root_streamlit_app", root_streamlit_path)
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
except Exception as e:
    raise SystemExit(f"Failed to import streamlit_app from repository root: {e}")

if hasattr(ui, "main"):
    ui.main()