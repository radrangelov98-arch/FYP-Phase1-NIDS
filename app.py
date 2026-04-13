import json
import joblib
import numpy as np
import pandas as pd
import re
import streamlit as st

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ARTIFACT_DIR = "artifacts"

# ---------------------------
# Shared helpers
# ---------------------------
ID_COLUMNS_LIKELY = [
    "flow_id", "source_ip", "destination_ip", "timestamp",
    "src_ip", "dst_ip", "src_address", "dst_address",
]

def canonicalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]+", "", regex=True)
    )
    return df

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    df = df.select_dtypes(include=[np.number]).copy()
    return df

def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in ID_COLUMNS_LIKELY if c in df.columns]
    if present:
        df = df.drop(columns=present, errors="ignore")
    return df

# ---------------------------
# Signature layer
# ---------------------------
def signature_portscan(df_raw: pd.DataFrame, unique_port_threshold: int = 20) -> pd.Series:
    if "source_ip" not in df_raw.columns or "destination_port" not in df_raw.columns:
        return pd.Series(False, index=df_raw.index)
    grp = df_raw.groupby("source_ip")["destination_port"].nunique()
    suspicious_src = set(grp[grp >= unique_port_threshold].index)
    return df_raw["source_ip"].isin(suspicious_src)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="FYP NIDS Prototype (Phase 1)", layout="wide")
st.title("FYP NIDS Prototype — Phase 1 (Hybrid demo: signature + anomaly)")

st.sidebar.header("Artifacts")
artifact_dir = st.sidebar.text_input(
    "Artifact folder (contains rf_phase1_model.pkl etc.)",
    value="artifacts"
)

def load_artifacts(folder: str):
    rf = joblib.load(f"{folder}/rf_phase1_model.pkl")
    scaler = joblib.load(f"{folder}/scaler_phase1.pkl")
    with open(f"{folder}/features_phase1.json", "r") as f:
        features = json.load(f)
    with open(f"{folder}/meta_phase1.json", "r") as f:
        meta = json.load(f)
    threshold = float(meta["threshold"])
    return rf, scaler, features, meta, threshold

loaded = None
try:
    loaded = load_artifacts(artifact_dir)
except Exception as e:
    st.sidebar.error(f"Couldn't load artifacts: {e}")

if loaded:
    rf, scaler, features, meta, default_threshold = loaded
    st.sidebar.success("Artifacts loaded")

    st.sidebar.header("Detection settings")
    th = st.sidebar.slider("Anomaly threshold", 0.01, 0.99, float(default_threshold), 0.01)
    enable_sig = st.sidebar.checkbox("Enable PortScan signature heuristic (demo)", value=True)
    uniq_th = st.sidebar.slider("Signature: unique ports per source_ip", 5, 100, 20, 1)

    st.subheader("Upload traffic CSV (CIC-IDS2017 style)")
    up = st.file_uploader("CSV file", type=["csv"])
    demo_btn = st.button("Load demo_traffic_anonymised.csv (if present)")

    df = None
    if up is not None:
        df = pd.read_csv(up)
    elif demo_btn:
        try:
            df = pd.read_csv(f"{artifact_dir}/demo_traffic_anonymised.csv")
        except Exception as e:
            st.error(f"Couldn't load demo CSV: {e}")

    if df is not None:
        raw = canonicalise_cols(df)

        sig_flag = pd.Series(False, index=raw.index)
        if enable_sig:
            sig_flag = signature_portscan(raw, unique_port_threshold=uniq_th)

        proc = drop_identifier_columns(raw)

        # drop label-like columns before model input
        for c in list(proc.columns):
            if "label" in c or c in ["y", "attack_type"]:
                proc = proc.drop(columns=[c], errors="ignore")

        X = safe_numeric(proc)

        missing = [c for c in features if c not in X.columns]
        extra   = [c for c in X.columns if c not in features]

        if missing:
            st.warning(f"Missing {len(missing)} required features. Filling with 0. Example: {missing[:5]}")
            for c in missing:
                X[c] = 0.0

        if extra:
            X = X.drop(columns=extra, errors="ignore")

        X = X[features]
        Xs = scaler.transform(X.values)

        proba = rf.predict_proba(Xs)[:, 1]
        anomaly_pred = (proba >= th).astype(int)

        final_pred = ((sig_flag.values.astype(int) == 1) | (anomaly_pred == 1)).astype(int)

        out = raw.copy()
        out["anomaly_score"] = proba
        out["signature_flag"] = sig_flag.astype(int)
        out["final_pred"] = final_pred
        out["final_label"] = np.where(final_pred == 1, "attack", "benign")

        # ── Summary ────────────────────────────────────────────────────
        st.subheader("Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(out))
        c2.metric("Alerts (attack)", int((out["final_pred"] == 1).sum()))
        c3.metric("Benign", int((out["final_pred"] == 0).sum()))

        # ── Evaluation: Confusion Matrix + Metrics ─────────────────────
        label_col = None
        for c in raw.columns:
            if "label" in c:
                label_col = c
                break

        if label_col is not None:
            st.subheader("Model Evaluation")

            y_true = raw[label_col].astype(str).str.strip().str.lower()
            y_true_bin = (y_true != "benign").astype(int)

            cm = confusion_matrix(y_true_bin, final_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Benign", "Attack"]
            )
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix — Random Forest")
            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
            accuracy  = (tp + tn) / len(y_true_bin)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{accuracy:.3f}")
            m2.metric("Precision", f"{precision:.3f}")
            m3.metric("Recall",    f"{recall:.3f}")
            m4.metric("F1 Score",  f"{f1:.3f}")
            m5.metric("FPR",       f"{fpr:.3f}")
        else:
            st.info("Upload a labelled CSV to see the confusion matrix and metrics.")

        # ── Top Alerts ─────────────────────────────────────────────────
        st.subheader("Top alerts (highest anomaly score)")
        st.dataframe(out.sort_values("anomaly_score", ascending=False).head(50), use_container_width=True)

        st.download_button(
            "Download results CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="nids_phase1_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload a CSV or click the demo button to run the detector.")

else:
    st.info("Set the correct artifact folder path in the sidebar to begin.")
