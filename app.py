import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================
# Network Intrusion Detection – Phase 1 
# ------------------------------------------------------
# Purpose:
# Offline analysis tool for network flow data.
# Applies simple rule-based checks and statistical scoring
# to flag potentially suspicious traffic for inspection.
#
# This is an academic prototype, not a production system.
# ======================================================

ARTIFACT_DIR = "artifacts"

# ---------------------------
# Helper functions
# ---------------------------

# Columns that usually identify flows or hosts and should not
# be used as numerical features
ID_COLUMNS_LIKELY = [
    "flow_id", "source_ip", "destination_ip", "timestamp",
    "src_ip", "dst_ip", "src_address", "dst_address",
]

def canonicalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names for consistency."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]+", "", regex=True)
    )
    return df

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric where possible.
    Invalid values are handled conservatively.
    """
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    df = df.select_dtypes(include=[np.number]).copy()
    return df

def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier-like columns that should not be analysed."""
    df = df.copy()
    present = [c for c in ID_COLUMNS_LIKELY if c in df.columns]
    if present:
        df = df.drop(columns=present, errors="ignore")
    return df

# ---------------------------
# Simple signature rule
# ---------------------------

def signature_portscan(df_raw: pd.DataFrame, unique_port_threshold: int = 20) -> pd.Series:
    """
    Simple rule-based heuristic:
    Flags a source IP if it connects to many different destination ports.
    """
    if "source_ip" not in df_raw.columns or "destination_port" not in df_raw.columns:
        return pd.Series(False, index=df_raw.index)

    grp = df_raw.groupby("source_ip")["destination_port"].nunique()
    suspicious_src = set(grp[grp >= unique_port_threshold].index)
    return df_raw["source_ip"].isin(suspicious_src)

# ======================================================
# Streamlit UI configuration (light theme)
# ======================================================

st.set_page_config(
    page_title="Network Intrusion Detection – Phase 1",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# light theme (no dashboard / no cyber styling)
st.markdown("""
<style>
.main { background-color: #ffffff; }

section[data-testid="stSidebar"] {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f2933;
    font-weight: 600;
}

p, span, label {
    color: #374151;
    font-size: 0.95rem;
}

.stButton > button {
    background-color: #e5e7eb;
    color: #111827;
    border-radius: 4px;
    border: 1px solid #d1d5db;
}

[data-testid="stFileUploader"] {
    border: 1px solid #d1d5db;
    padding: 0.75rem;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Page header
# ---------------------------

st.title("Network Intrusion Detection – Phase 1")
st.write(
    "This application supports offline analysis of network flow data. "
    "It applies simple rule-based checks and statistical scoring to highlight "
    "potentially suspicious traffic for further inspection."
)

st.info(
    "Note: This prototype flags traffic patterns for analysis purposes only. "
    "It does not claim definitive identification of attacks."
)

# ---------------------------
# Sidebar: configuration only
# ---------------------------

st.sidebar.header("Configuration")

artifact_dir = st.sidebar.text_input(
    "Artifact folder",
    value="artifacts"
)

def load_artifacts(folder: str):
    """Load trained model, scaler, feature list and metadata."""
    rf = joblib.load(f"{folder}/rf_phase1_model.pkl")
    scaler = joblib.load(f"{folder}/scaler_phase1.pkl")
    with open(f"{folder}/features_phase1.json","r") as f:
        features = json.load(f)
    with open(f"{folder}/meta_phase1.json","r") as f:
        meta = json.load(f)
    threshold = float(meta["threshold"])
    return rf, scaler, features, meta, threshold

loaded = None
try:
    loaded = load_artifacts(artifact_dir)
except Exception as e:
    st.sidebar.error(f"Could not load artifacts: {e}")

if loaded:
    rf, scaler, features, meta, default_threshold = loaded
    st.sidebar.success("Artifacts loaded")

    st.sidebar.subheader("Detection parameters")

    th = st.sidebar.slider(
        "Statistical sensitivity threshold",
        0.01, 0.99, float(default_threshold), 0.01
    )

    enable_sig = st.sidebar.checkbox(
        "Enable simple port-scan rule (demonstration)",
        value=True
    )

    uniq_th = st.sidebar.slider(
        "Port-scan rule: unique destination ports",
        5, 100, 20, 1
    )

    # ---------------------------
    # 1. Data input
    # ---------------------------

    st.header("1. Data Input")
    st.write("Upload a CSV file containing network flow records.")

    up = st.file_uploader("Network traffic CSV", type=["csv"])
    demo_btn = st.button("Load example dataset (if available)")

    df = None
    if up is not None:
        df = pd.read_csv(up)
    elif demo_btn:
        try:
            df = pd.read_csv(f"{artifact_dir}/demo_traffic_anonymised.csv")
        except Exception as e:
            st.error(f"Could not load demo file: {e}")

    if df is not None:
        raw = canonicalise_cols(df)

        # Apply rule-based signature
        sig_flag = pd.Series(False, index=raw.index)
        if enable_sig:
            sig_flag = signature_portscan(raw, unique_port_threshold=uniq_th)

        # Prepare numeric feature matrix
        proc = drop_identifier_columns(raw)

        for c in list(proc.columns):
            if "label" in c or c in ["y", "attack_type"]:
                proc = proc.drop(columns=[c], errors="ignore")

        X = safe_numeric(proc)

        # Align features to training configuration
        missing = [c for c in features if c not in X.columns]
        extra   = [c for c in X.columns if c not in features]

        for c in missing:
            X[c] = 0.0

        if extra:
            X = X.drop(columns=extra, errors="ignore")

        X = X[features]
        Xs = scaler.transform(X.values)

        # Statistical scoring
        scores = rf.predict_proba(Xs)[:, 1]
        stat_flag = (scores >= th).astype(int)

        # Final decision (rule OR statistical flag)
        final_pred = ((sig_flag.values.astype(int) == 1) | (stat_flag == 1)).astype(int)

        out = raw.copy()
        out["anomaly_score"] = scores
        out["signature_flag"] = sig_flag.astype(int)
        out["final_label"] = np.where(final_pred == 1, "suspicious", "normal")

        # ---------------------------
        # 2. Analysis summary
        # ---------------------------

        st.header("2. Analysis Summary")

        st.table(pd.DataFrame({
            "Measure": [
                "Total flows analysed",
                "Suspicious flows identified",
                "Normal flows"
            ],
            "Count": [
                len(out),
                int((out["final_label"] == "suspicious").sum()),
                int((out["final_label"] == "normal").sum())
            ]
        }))

        # ---------------------------
        # 3. Detailed inspection
        # ---------------------------

        st.header("3. Detailed Inspection")
        st.write(
            "Records with the highest statistical scores. "
            "These entries may require manual review."
        )

        st.dataframe(
            out.sort_values("anomaly_score", ascending=False).head(50),
            use_container_width=True
        )

        st.download_button(
            "Download analysis results (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="nids_phase1_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a CSV file or load the example dataset to begin.")
else:
    st.info("Set the correct artifact folder in the sidebar to begin.")

