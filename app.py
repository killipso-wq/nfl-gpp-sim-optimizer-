"""
Fallback entry point for Streamlit.
Allows existing Render services using `streamlit run app.py` to work
by delegating to `streamlit_app.py`.
"""
import importlib

# Importing runs the top-level Streamlit script in streamlit_app.py
importlib.import_module("streamlit_app")