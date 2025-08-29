
import io, requests, numpy as np, pandas as pd
from typing import Optional, Tuple, List

def download_nsl_kdd() -> Optional[pd.DataFrame]:
    """
    Downloads NSL-KDD Train+ and Test+ from a public mirror.
    Returns a concatenated DataFrame or None on failure.
    """
    try:
        cols_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/Field Names.csv"
        c = requests.get(cols_url, timeout=30)
        col_lines = [ln.strip() for ln in c.text.splitlines() if ln.strip()]
        feature_cols = [ln.split(",")[0] for ln in col_lines]

        tr_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
        te_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
        rt = requests.get(tr_url, timeout=60)
        re = requests.get(te_url, timeout=60)

        df_tr = pd.read_csv(io.StringIO(rt.text), header=None, names=feature_cols)
        df_te = pd.read_csv(io.StringIO(re.text), header=None, names=feature_cols)
        return pd.concat([df_tr, df_te], ignore_index=True)
    except Exception:
        return None

def synthesize_flows(n:int=12000, seed:int=7) -> pd.DataFrame:
    np.random.seed(seed)
    protos = ["tcp","udp","icmp"]
    services = ["http","ftp","smtp","dns","ssh","other"]
    flags = ["SF","S0","REJ","RSTR","SH"]
    labels = ["normal","dos","probe","r2l","u2r"]

    df = pd.DataFrame({
        "duration": np.random.exponential(2.0, n),
        "protocol_type": np.random.choice(protos, n, p=[0.7,0.25,0.05]),
        "service": np.random.choice(services, n),
        "flag": np.random.choice(flags, n),
        "src_bytes": np.random.gamma(2.0, 200, n).astype(int),
        "dst_bytes": np.random.gamma(2.0, 150, n).astype(int),
        "count": np.random.randint(1, 100, n),
        "srv_count": np.random.randint(1, 100, n),
        "same_srv_rate": np.random.rand(n),
        "dst_host_count": np.random.randint(1, 255, n),
        "dst_host_srv_count": np.random.randint(1, 255, n),
        "label": np.random.choice(labels, n, p=[0.6,0.2,0.15,0.04,0.01])
    })
    return df

def prepare_nsltkdd_like(df: pd.DataFrame):
    df = df.copy()
    if "label" not in df.columns and "class" in df.columns:
        df.rename(columns={"class":"label"}, inplace=True)

    keep = [c for c in ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
                        "count","srv_count","same_srv_rate","dst_host_count","dst_host_srv_count","label"]
            if c in df.columns]
    df = df[keep].dropna()
    df["binary"] = (df["label"].astype(str).str.lower() != "normal").astype(int)

    X = df.drop(columns=["label","binary"], errors="ignore")
    y_bin = df["binary"]
    y_multi = df["label"].astype(str)
    return X, y_bin, y_multi
