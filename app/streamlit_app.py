"""
Shim to support Start Command: `streamlit run app/streamlit_app.py`.
This imports the root-level `streamlit_app` and calls `main()`.
"""
from __future__ import annotations
import os
import sys
import importlib

# Ensure repository root is on sys.path when executed from app/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    ui = importlib.import_module("streamlit_app")
except Exception as e:
    # Fail fast with a clear message for Render logs
    raise SystemExit(f"Failed to import root-level streamlit_app: {e}")

if hasattr(ui, "main") and callable(getattr(ui, "main")):
    ui.main()
else:
    # Graceful fallback so Streamlit renders an informative message
    try:
        import streamlit as st
        st.error("streamlit_app.main() not found; nothing to run.")
    except Exception:
        # If Streamlit isn't importable for some reason, just exit
        raise SystemExit("streamlit_app.main() not found; nothing to run.")