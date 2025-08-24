import os
import io
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st


def version_banner():
    branch = os.getenv("RENDER_GIT_BRANCH") or "local"
    commit = (os.getenv("RENDER_GIT_COMMIT") or "")[:7]
    service = os.getenv("RENDER_SERVICE_NAME") or ""
    banner = f"branch: {branch}"
    if commit:
        banner += f" • commit: {commit}"
    if service:
        banner += f" • service: {service}"
    st.caption(banner)


def main():
    st.set_page_config(page_title="NFL GPP Sim Optimizer", layout="wide")
    st.title("NFL GPP Sim Optimizer")
    version_banner()

    st.sidebar.header("Inputs")

    players_file = st.sidebar.file_uploader("Players CSV (required)", type=["csv"], key="players_csv")
    sims_file = st.sidebar.file_uploader("sims.csv (optional)", type=["csv"], key="sims_csv")
    def_file = st.sidebar.file_uploader("DEF.csv (optional)", type=["csv"], key="def_csv")
    stacks_file = st.sidebar.file_uploader("QBRBWRTE.csv (optional)", type=["csv"], key="stacks_csv")

    preset = st.sidebar.selectbox("Lineup preset", ["se", "mid", "large"], index=1)
    n_lineups = st.sidebar.number_input("Number of lineups", min_value=1, max_value=150, value=20, step=1)

    st.write("Upload files in the left sidebar, choose a preset and quantity, then click Run optimizer.")

    run = st.button("Run optimizer", use_container_width=True)

    if run:
        if not players_file:
            st.error("Please upload Players CSV to proceed.")
            st.stop()

        # Read the uploads that exist
        try:
            players_df = pd.read_csv(players_file)
        except Exception as e:
            st.error(f"Failed to read Players CSV: {e}")
            st.stop()

        st.success("Inputs loaded.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Players.csv (preview)")
            st.dataframe(players_df.head(20), use_container_width=True)
        with c2:
            st.subheader("Shapes")
            st.write(
                {
                    "players": players_df.shape,
                    "sims": pd.read_csv(sims_file).shape if sims_file else None,
                    "DEF": pd.read_csv(def_file).shape if def_file else None,
                    "QBRBWRTE": pd.read_csv(stacks_file).shape if stacks_file else None,
                }
            )

        st.divider()
        st.subheader("Optimization output (placeholder)")

        # Minimal placeholder results so the app produces downloadable files.
        results = pd.DataFrame(
            {
                "lineup_id": list(range(1, int(n_lineups) + 1)),
                "preset": [preset] * int(n_lineups),
                "generated_at_utc": [datetime.utcnow().isoformat()] * int(n_lineups),
                "note": ["placeholder lineup"] * int(n_lineups),
            }
        )

        st.write("Generated example results (placeholder). Replace with real optimizer logic later.")
        st.dataframe(results.head(20), use_container_width=True)

        # Single CSV download
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Zip download (can add more files later)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("results.csv", csv_bytes)
        zip_buf.seek(0)
        st.download_button(
            "Download all results (zip)",
            data=zip_buf.getvalue(),
            file_name="results.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
