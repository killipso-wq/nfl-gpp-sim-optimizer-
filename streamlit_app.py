import os
import io
import glob
import time
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

import streamlit as st

APP_TITLE = "NFL GPP Sim Optimizer"
DESCRIPTION = (
    "Upload your CSVs (Players.csv required; optional sims.csv, DEF.csv, QBRBWRTE.csv), "
    "choose a lineup preset, and build lineups. Files are processed ephemerally and are not persisted."
)

st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Version banner (commit/branch) ---
def _version_banner():
    branch = os.getenv("RENDER_GIT_BRANCH") or os.getenv("GIT_BRANCH")
    commit = os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or os.getenv("COMMIT_SHA")
    service = os.getenv("RENDER_SERVICE_NAME")
    short = commit[:7] if commit else "unknown"
    parts = [f"branch: {branch or 'unknown'}", f"commit: {short}"]
    if service:
        parts.append(f"service: {service}")
    st.caption(" | ".join(parts))

_version_banner()
# --------------------------------------

st.title(APP_TITLE)
st.write(DESCRIPTION)

with st.sidebar:
    st.header("Inputs")
    players_file = st.file_uploader("Players.csv (required)", type=["csv"], accept_multiple_files=False)
    sims_file = st.file_uploader("sims.csv (optional)", type=["csv"], accept_multiple_files=False)
    def_file = st.file_uploader("DEF.csv (optional)", type=["csv"], accept_multiple_files=False)
    corr_file = st.file_uploader("QBRBWRTE.csv (optional)", type=["csv"], accept_multiple_files=False)

    preset = st.selectbox("Lineup preset", options=["se", "mid", "large"], index=2)
    n_lineups = st.number_input("Number of lineups", min_value=1, max_value=10000, value=150, step=1)

    run_clicked = st.button("Run optimizer", type="primary", use_container_width=True)

status_placeholder = st.empty()
results_placeholder = st.container()


def save_upload(uploaded_file, dest_path: Path):
    if uploaded_file is None:
        return None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def run_cli(workdir: Path, players: Path, sims: Path | None, defcsv: Path | None, corrcsv: Path | None, preset: str, n: int):
    outdir = workdir / "out"
    outdir.mkdir(exist_ok=True)

    # Build flags conditionally
    sims_flag = ["--sims", str(sims)] if sims and sims.exists() else []
    def_flag = ["--defcsv", str(defcsv)] if defcsv and defcsv.exists() else []
    corr_flag = ["--corrcsv", str(corrcsv)] if corrcsv and corrcsv.exists() else []

    compare_cmd = [
        "python", "-m", "nfl_gpp_sim_optimizer.cli", "compare-apply",
        "--players", str(players),
        "--outdir", str(outdir),
        *sims_flag, *def_flag, *corr_flag,
    ]

    build_cmd = [
        "python", "-m", "nfl_gpp_sim_optimizer.cli", "build-lineups",
        "--players", str(outdir / "players_adjusted.csv"),
        "--preset", preset,
        "--n", str(n),
        "--outdir", str(outdir),
        *corr_flag,
    ]

    start_time = time.time()
    # Run compare-apply
    proc1 = subprocess.run(compare_cmd, capture_output=True, text=True)
    # Streamlit-friendly logging
    st.code("$ " + " ".join(compare_cmd) + "\n\n" + proc1.stdout)
    if proc1.returncode != 0:
        raise RuntimeError(f"compare-apply failed:\nSTDOUT:\n{proc1.stdout}\n\nSTDERR:\n{proc1.stderr}")

    # Run build-lineups
    proc2 = subprocess.run(build_cmd, capture_output=True, text=True)
    st.code("$ " + " ".join(build_cmd) + "\n\n" + proc2.stdout)
    if proc2.returncode != 0:
        raise RuntimeError(f"build-lineups failed:\nSTDOUT:\n{proc2.stdout}\n\nSTDERR:\n{proc2.stderr}")

    elapsed = time.time() - start_time
    return outdir, elapsed


if run_clicked:
    if not players_file:
        st.error("Please upload Players.csv to proceed.")
    else:
        with st.spinner("Running optimizer... this can take a minute or two"):
            workdir = Path(tempfile.mkdtemp(prefix="optimizer_"))
            try:
                data_dir = workdir / "data"
                players_path = save_upload(players_file, data_dir / "Players.csv")
                sims_path = save_upload(sims_file, data_dir / "sims.csv")
                def_path = save_upload(def_file, data_dir / "DEF.csv")
                corr_path = save_upload(corr_file, data_dir / "QBRBWRTE.csv")

                status_placeholder.info("Blending projections and building lineups...")
                outdir, elapsed = run_cli(workdir, players_path, sims_path, def_path, corr_path, preset, int(n_lineups))

                results_placeholder.subheader("Results")
                csv_files = sorted(outdir.glob("*.csv"))
                if not csv_files:
                    st.warning("No output CSVs were produced. Please check your inputs.")
                else:
                    # Offer individual downloads
                    for csv_path in csv_files:
                        with open(csv_path, "rb") as f:
                            st.download_button(
                                label=f"Download {csv_path.name}",
                                data=f.read(),
                                file_name=csv_path.name,
                                mime="text/csv",
                            )

                    # Offer a zip of all outputs
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for csv_path in csv_files:
                            zf.write(csv_path, arcname=csv_path.name)
                    buf.seek(0)
                    st.download_button(
                        label="Download all results (zip)",
                        data=buf.getvalue(),
                        file_name="optimizer-results.zip",
                        mime="application/zip",
                    )

                status_placeholder.success(f"Done in {elapsed:.1f}s")
            except Exception as e:
                st.error("An error occurred while running the optimizer. See logs below.")
                st.exception(e)
            finally:
                # Best effort cleanup of temp workspace
                try:
                    shutil.rmtree(workdir, ignore_errors=True)
                except Exception:
                    pass
else:
    st.info("Upload files in the sidebar and click 'Run optimizer'.")