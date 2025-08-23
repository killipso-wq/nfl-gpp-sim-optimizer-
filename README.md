# NFL GPP Sim Optimizer

## Streamlit UI (Render)
Deploy a simple UI where you can upload CSVs privately (not stored) and download outputs.

### One‑click deploy
If you forked this repo, Render will auto‑detect `render.yaml`.

1. Create a new Web Service at https://dashboard.render.com
2. Connect your GitHub repo and select this repository
3. Render should pick up `render.yaml` automatically
4. Click Create Web Service
5. Once deployed, open the URL to use the app

### Usage
- Upload `Players.csv` (required). Optionally upload `sims.csv`, `DEF.csv`, and `QBRBWRTE.csv`.
- Choose preset (se/mid/large) and the number of lineups.
- Click Run optimizer. When complete, download `lineups.csv`, `players_adjusted.csv`, and other outputs.

All processing happens on the server at request time. Files are not committed to the repo.

### Local run (optional)
```bash
pip install -e .
pip install -r requirements.txt
streamlit run streamlit_app.py
```