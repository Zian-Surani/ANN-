
import os, json, io, time
import numpy as np, pandas as pd
import streamlit as st

from dfa import dfa_check
from data_utils import download_nsl_kdd, synthesize_flows, prepare_nsltkdd_like
from model_utils import train_and_save, load_model, predict_one, predict_batch, MODEL_PATH, METRICS_PATH

st.set_page_config(page_title="Hybrid IDS (DFA + ANN)", layout="wide")

st.title("🔐 AI-Driven IDS: DFA + ANN (Streamlit)")
st.caption("Real-time threat detection for cloud-based networks — with deterministic automata + neural anomaly detection.")

# Sidebar: Actions
with st.sidebar:
    st.header("⚙️ Controls")
    dataset_choice = st.selectbox("Dataset", ["Try NSL-KDD first", "Synthetic"])
    st.write("Model artifacts:", MODEL_PATH)
    if st.button("🧠 Train / Rebuild Model"):
        with st.spinner("Training in progress..."):
            df = download_nsl_kdd() if dataset_choice == "Try NSL-KDD first" else None
            used = "NSL-KDD"
            if df is None or len(df) < 5000:
                df = synthesize_flows(12000)
                used = "Synthetic"
            X, y_bin, y_multi = prepare_nsltkdd_like(df)
            metrics = train_and_save(X, y_bin)
        st.success(f"Training complete ({used}). AUC: {metrics.get('roc_auc')}")
        st.json(metrics)

    st.markdown("---")
    if st.button("🔄 Reload Model"):
        try:
            _ = load_model()
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) DFA Checker", "2) Single Prediction", "3) Batch Prediction", "4) Metrics", "5) Logs & Export"
])

# 1) DFA Checker
with tab1:
    st.subheader("Deterministic Finite Automata — Signature-like Checks")
    st.write("Enter a sequence of TCP-like events (e.g., `SYN, SYN, SYN-ACK, ACK`).")
    seq = st.text_input("Events (comma-separated)", "SYN, SYN, SYN")
    if st.button("Run DFA"):
        events = [s.strip() for s in seq.split(",") if s.strip()]
        res = dfa_check(events)
        st.write("Results:", res)
        if any(res.values()):
            st.warning("⚠️ DFA triggered a suspicious pattern.")
        else:
            st.success("✅ No DFA signatures triggered.")

# 2) Single Prediction
with tab2:
    st.subheader("ANN — Single Flow Prediction")
    st.write("Fill in basic NSL-KDD-like features for a single flow.")
    cols1 = st.columns(3)
    duration = cols1[0].number_input("duration", 0.0, 1e6, 1.2)
    protocol_type = cols1[1].selectbox("protocol_type", ["tcp","udp","icmp"])
    service = cols1[2].selectbox("service", ["http","ftp","smtp","dns","ssh","other"])

    cols2 = st.columns(3)
    flag = cols2[0].selectbox("flag", ["SF","S0","REJ","RSTR","SH"])
    src_bytes = int(cols2[1].number_input("src_bytes", 0, 1_000_000, 300))
    dst_bytes = int(cols2[2].number_input("dst_bytes", 0, 1_000_000, 450))

    cols3 = st.columns(3)
    count = int(cols3[0].number_input("count", 1, 10_000, 5))
    srv_count = int(cols3[1].number_input("srv_count", 1, 10_000, 5))
    same_srv_rate = cols3[2].number_input("same_srv_rate", 0.0, 1.0, 0.7)

    cols4 = st.columns(2)
    dst_host_count = int(cols4[0].number_input("dst_host_count", 1, 255, 20))
    dst_host_srv_count = int(cols4[1].number_input("dst_host_srv_count", 1, 255, 18))

    if st.button("Predict Flow"):
        try:
            pipe = load_model()
        except Exception as e:
            st.error(f"Model not loaded: {e}")
        else:
            row = {
                "duration": duration, "protocol_type": protocol_type, "service": service, "flag": flag,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes, "count": count, "srv_count": srv_count,
                "same_srv_rate": same_srv_rate, "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count
            }
            out = predict_one(pipe, row)
            st.write(out)
            if out["binary"] == 1:
                st.warning("⚠️ Predicted: ATTACK")
            else:
                st.success("✅ Predicted: NORMAL")

# 3) Batch Prediction
with tab3:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV with columns: `duration, protocol_type, service, flag, src_bytes, dst_bytes, count, srv_count, same_srv_rate, dst_host_count, dst_host_srv_count`")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up and st.button("Run Batch Prediction"):
        try:
            pipe = load_model()
        except Exception as e:
            st.error(f"Load failed: {e}")
        else:
            df = pd.read_csv(up)
            needed = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","count","srv_count","same_srv_rate","dst_host_count","dst_host_srv_count"]
            miss = [c for c in needed if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                out = predict_batch(pipe, df[needed])
                st.success(f"Rows: {len(out)} | Attacks: {int((out['binary_pred']==1).sum())}")
                st.dataframe(out.head(100))
                st.download_button("⬇️ Download Full Results", out.to_csv(index=False).encode("utf-8"), file_name="batch_predictions.csv", mime="text/csv")

# 4) Metrics
with tab4:
    st.subheader("Model Metrics")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            m = json.load(f)
        st.json(m)
        # Basic confusion matrix plot
        try:
            import matplotlib.pyplot as plt
            cm = np.array(m.get("confusion_matrix", [[0,0],[0,0]]))
            fig = plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.xticks([0,1], ["Normal","Attack"])
            plt.yticks([0,1], ["Normal","Attack"])
            for (i,j), val in np.ndenumerate(cm):
                plt.text(j, i, str(val), ha="center", va="center")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Plot skipped: {e}")
    else:
        st.info("No metrics yet. Train the model from the sidebar.")

# 5) Logs & Export
with tab5:
    st.subheader("Quick Start / Export")
    st.markdown("""
**How to run locally**  
```bash
pip install -r requirements.txt
streamlit run app.py
```
Artifacts are stored in `./artifacts`.
    """)
    # Offer a small sample CSV template for users
    import io
    tmpl = pd.DataFrame([{
        "duration": 1.2, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 300, "dst_bytes": 450, "count": 5, "srv_count": 5,
        "same_srv_rate": 0.7, "dst_host_count": 20, "dst_host_srv_count": 18
    }])
    st.download_button("⬇️ Download Sample CSV Template", tmpl.to_csv(index=False).encode("utf-8"),
                       file_name="sample_template.csv", mime="text/csv")

st.success("Ready. Use the sidebar to train, then try predictions.")
