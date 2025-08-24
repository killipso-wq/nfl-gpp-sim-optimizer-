"""
Fallback entry point for Streamlit on Render.

This lets Start Command use:
    streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

It simply imports streamlit_app, which defines the UI.
"""

try:
    import streamlit_app as ui
except Exception as e:
    raise SystemExit(f"Failed to import streamlit_app: {e}")

if hasattr(ui, "main"):
    ui.main()
