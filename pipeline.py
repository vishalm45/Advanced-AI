import argparse, json, time, os, warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================
# Dependencies:
#   pandas numpy scikit-learn pgmpy pomegranate
# ===========================

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.impute import SimpleImputer

# GP Classifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Discrete BN (pgmpy)
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Gaussian BN (pomegranate) — per your environment
from pomegranate.bayesian_network import BayesianNetwork as PomBayesNet


# ===========
# Utilities
# ===========

ID_LIKE_COLS = {
    "Transaction_ID", "User_ID", "IP_Address", "IP_Address_Flag",
    "session_id", "id", "uid", "uuid"
}

@dataclass
class Metrics:
    accuracy: float
    roc_auc: float
    brier: float
    logloss: float
    ece: float
    kl_rel: float

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        mask = inds == b
        if np.any(mask):
            conf = y_prob[mask].mean()
            acc = y_true[mask].mean()
            ece += np.abs(acc - conf) * (np.sum(mask) / N)
    return float(ece)

def kl_reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    # KL(P||Q) between observed positive rate in each bin and mean predicted prob in the bin
    y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    p_obs, p_pred = [], []
    for b in range(n_bins):
        mask = inds == b
        if np.any(mask):
            p_obs.append(np.clip(y_true[mask].mean(), 1e-6, 1-1e-6))
            p_pred.append(np.clip(y_prob[mask].mean(), 1e-6, 1-1e-6))
    if len(p_obs) < 2:
        return float("nan")
    p_obs = np.array(p_obs); p_pred = np.array(p_pred)
    return float(np.sum(p_obs * (np.log(p_obs) - np.log(p_pred))))

def ensure_binary_labels(y: pd.Series) -> pd.Series:
    if set(pd.unique(y)).issubset({0, 1}):
        return y.astype(int)
    mapping = {"No":0,"no":0,"N":0,"False":0,"F":0,"Yes":1,"yes":1,"Y":1,"True":1,"T":1}
    ym = y.map(lambda v: mapping.get(v, v))
    try:
        return ym.astype(int)
    except Exception:
        codes, uniques = pd.factorize(ym)
        if len(uniques) != 2:
            raise ValueError(f"Target is not binary; found classes: {list(uniques)}")
        return pd.Series(codes, index=y.index).astype(int)

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = ensure_binary_labels(df[target_col].copy())
    X = df.drop(columns=[target_col]).copy()
    return X, y

def infer_column_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    cat_cols, num_cols = [], []
    for c in df.columns:
        if c == target_col: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            # try coerce
            try:
                pd.to_numeric(df[c])
                num_cols.append(c)
            except Exception:
                cat_cols.append(c)
    return cat_cols, num_cols

def group_rare_categories(X: pd.DataFrame, cat_cols: List[str], min_count: int = 20) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        vc = X[c].astype("object").value_counts(dropna=False)
        rare = set(vc[vc < min_count].index)
        if rare:
            X[c] = X[c].apply(lambda v: "Other" if v in rare else v)
    return X

def drop_id_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_drop = [c for c in df.columns if c in ID_LIKE_COLS or c.lower() in ("transaction_id","user_id","id")]
    if cols_drop:
        return df.drop(columns=cols_drop)
    return df


# ==========================
#  Discrete BN - Structures
# ==========================

def mutual_info_discrete(a: pd.Series, b: pd.Series) -> float:
    # fast MI via contingency table
    xa = a.astype("int64")
    xb = b.astype("int64")
    # contingency
    tab = pd.crosstab(xa, xb)
    if tab.size == 0:
        return 0.0
    # Compute MI
    Pxy = tab.values / tab.values.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(Pxy>0, Pxy / (Px @ Py), 1.0)
        mi = np.sum(np.where(Pxy>0, Pxy * np.log(frac), 0.0))
    return float(max(mi, 0.0))

def chow_liu_tree_discrete(df_disc: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Custom Chow–Liu: maximum spanning tree over discrete vars (including 'target').
    Returns directed edges rooted at 'target' (edges point away from target).
    """
    cols = list(df_disc.columns)
    assert "target" in cols
    # compute MI for all pairs
    n = len(cols)
    mi_edges = []
    for i in range(n):
        for j in range(i+1, n):
            c1, c2 = cols[i], cols[j]
            mi = mutual_info_discrete(df_disc[c1], df_disc[c2])
            mi_edges.append((mi, c1, c2))
    # Kruskal for maximum spanning tree
    parent = {c: c for c in cols}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    edges_undirected = []
    for mi, u, v in sorted(mi_edges, key=lambda t: t[0], reverse=True):
        if union(u, v):
            edges_undirected.append((u, v))

    # Direct edges from root ('target') using BFS
    adj = {c: [] for c in cols}
    for u, v in edges_undirected:
        adj[u].append(v); adj[v].append(u)
    root = "target"
    visited = set([root])
    queue = [root]
    directed = []
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                directed.append((u, v))  # direction away from root
                queue.append(v)
    return directed


# ==========================
#  Discrete BN (DBN)
# ==========================

def preprocess_for_dbn(X: pd.DataFrame, y: pd.Series, rare_min_count: int) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Drop id-like columns, impute, group rare categories
    XY = pd.concat([X, y.rename("target")], axis=1)
    XY = drop_id_like_columns(XY)
    cat_cols, num_cols = infer_column_types(XY, "target")
    # Impute
    Xi = XY.drop(columns=["target"]).copy()
    if num_cols:
        Xi[num_cols] = Xi[num_cols].replace([np.inf, -np.inf], np.nan)
        Xi[num_cols] = Xi[num_cols].fillna(Xi[num_cols].median())
    if cat_cols:
        for c in cat_cols:
            Xi[c] = Xi[c].astype("object").fillna("Missing")
        Xi = group_rare_categories(Xi, cat_cols, min_count=rare_min_count)
    return Xi, cat_cols, num_cols

def discretize_numeric(X: pd.DataFrame, num_cols: List[str], bins: int) -> Tuple[pd.DataFrame, Optional[KBinsDiscretizer]]:
    if not num_cols:
        return X.copy(), None
    disc = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
    Xd_num = pd.DataFrame(disc.fit_transform(X[num_cols]), columns=num_cols, index=X.index).astype(int)
    Xd = pd.concat([X.drop(columns=num_cols), Xd_num], axis=1)
    return Xd, disc

def apply_discretizer_row(disc: KBinsDiscretizer, col: str, value) -> int:
    # transform a single value for a fitted single column
    val = pd.DataFrame({col: [value]})
    return int(disc.transform(val)[:, 0][0])

def train_discrete_bn(
    X: pd.DataFrame, y: pd.Series, bins: int, structure: str = "hc", max_indegree: int = 3, max_iter: int = 200
) -> Tuple[BayesianNetwork, VariableElimination, Optional[KBinsDiscretizer], pd.DataFrame]:
    Xi, cat_cols, num_cols = preprocess_for_dbn(X, y, rare_min_count=20)
    Xd, disc = discretize_numeric(Xi, num_cols, bins=bins)

    data_d = pd.concat([Xd, y.rename("target")], axis=1)

    # Structure learning
    t0 = time.time()
    if structure == "hc":
        est = HillClimbSearch(data_d, scoring_method=K2Score(data_d))
        model = est.estimate(max_indegree=max_indegree, max_iter=max_iter)
    elif structure == "chowliu":
        edges = chow_liu_tree_discrete(data_d)
        model = BayesianNetwork(edges)
    else:
        raise ValueError("structure must be one of {'hc','chowliu'}")
    t_struct = time.time() - t0

    # Parameter learning
    t0 = time.time()
    model.fit(data_d, estimator=MaximumLikelihoodEstimator)
    t_params = time.time() - t0

    infer = VariableElimination(model)
    model._fit_times = (t_struct, t_params)
    model._dbn_disc_cols = (cat_cols, num_cols)
    model._disc_transformer = disc
    return model, infer, disc, Xd

def dbn_predict_proba(infer: VariableElimination, disc: Optional[KBinsDiscretizer], cat_cols: List[str], num_cols: List[str], row: pd.Series) -> float:
    # Discretize numeric evidence and query P(target=1 | evidence=row)
    evidence = {}
    for c, v in row.items():
        if c in num_cols and disc is not None:
            evidence[c] = apply_discretizer_row(disc, c, v)
        else:
            evidence[c] = v
    q = infer.query(variables=["target"], evidence=evidence, show_progress=False)
    try:
        return float(q.values[1])
    except Exception:
        # fallback: choose the state labeled '1' if present
        states = getattr(q, "state_names", {}).get("target", [0, 1])
        if isinstance(states, list) and 1 in states:
            idx = states.index(1)
            return float(q.values[idx])
        return float(np.argmax(q.values))


# ==========================
#  Gaussian BN (GBN)
# ==========================

def preprocess_for_gbn(X: pd.DataFrame, y: pd.Series, rare_min_count: int) -> pd.DataFrame:
    df = pd.concat([X, y.rename("target")], axis=1)
    df = drop_id_like_columns(df)
    # impute
    cat_cols, num_cols = infer_column_types(df, "target")
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if cat_cols:
        for c in cat_cols:
            df[c] = df[c].astype("object").fillna("Missing")
        df = group_rare_categories(df, cat_cols, min_count=rare_min_count)
    return df

def train_gaussian_bn(X: pd.DataFrame, y: pd.Series) -> PomBayesNet:
    y = ensure_binary_labels(y).astype(int)
    df = preprocess_for_gbn(X, y, rare_min_count=20)
    # force discrete target for robust posteriors
    df["target"] = df["target"].astype(str)
    t0 = time.time()
    model = PomBayesNet.from_samples(df.values, algorithm="chow-liu", state_names=list(df.columns))
    model._fit_time = time.time() - t0
    model._columns = list(df.columns)
    return model

def gbn_predict_proba(model: PomBayesNet, x_row: Dict[str, Any]) -> float:
    dists = model.predict_proba(x_row)
    idx = {name: i for i, name in enumerate(model._columns)}["target"]
    targ = dists[idx]
    # Expect DiscreteDistribution over {"0","1"}
    try:
        params = targ.parameters[0]
        if "1" in params:
            return float(params["1"])
        if 1 in params:
            return float(params[1])
        # fallback: take the larger key as '1'
        keys = sorted(params.keys())
        return float(params[keys[-1]])
    except Exception as e:
        raise RuntimeError("GBN target is not discrete; ensure 'target' was cast to str before training.") from e


# ==========================
#  Gaussian Process (GP)
# ==========================

def make_gp_classifier() -> GaussianProcessClassifier:
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0)
    return GaussianProcessClassifier(kernel=kernel, random_state=0, n_restarts_optimizer=2, copy_X_train=False)

def fit_gp(
    X_tr: pd.DataFrame, y_tr: pd.Series
) -> Tuple[Pipeline, ColumnTransformer]:
    cat_cols, num_cols = infer_column_types(pd.concat([X_tr, y_tr.rename("target")], axis=1), "target")
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    gpc = make_gp_classifier()
    pipe = Pipeline([("pre", pre), ("clf", gpc)])
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    pipe._fit_time = time.time() - t0
    return pipe, pre

def gp_predict_proba(pipe: Pipeline, X_te: pd.DataFrame) -> Tuple[np.ndarray, float]:
    t0 = time.time()
    proba = pipe.predict_proba(X_te)[:, 1]
    return proba, time.time() - t0


# ==========================
#  Evaluation and CV
# ==========================

def evaluate_probs(y_true: np.ndarray, p1: np.ndarray) -> Metrics:
    p1 = np.clip(p1, 1e-7, 1-1e-7)
    acc = accuracy_score(y_true, (p1 >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_true, p1)
    except Exception:
        auc = float("nan")
    bri = brier_score_loss(y_true, p1)
    ll = log_loss(y_true, np.vstack([1-p1, p1]).T)
    ece = expected_calibration_error(y_true, p1)
    klr = kl_reliability(y_true, p1)
    return Metrics(acc, auc, bri, ll, ece, klr)

def cross_validate_models(
    df: pd.DataFrame, target_col: str, folds: int, discretize_bins: int, random_state: int,
    dataset_name: str, dbn_structure: str, dbn_sample_n: Optional[int], gp_max_train_n: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = split_features_target(df, target_col)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    rows, timings = [], []
    fold_idx = 0

    for tr_idx, te_idx in skf.split(X, y):
        fold_idx += 1
        X_tr_full, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr_full, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

        # Optionally sample training for BN and/or GP to ensure scalability
        def maybe_sample(Xd, yd, n):
            if n is not None and len(Xd) > n:
                rs = np.random.RandomState(random_state + fold_idx)
                idx = rs.choice(len(Xd), n, replace=False)
                return Xd.iloc[idx].copy(), yd.iloc[idx].copy()
            return Xd, yd

        # ---------- Discrete BN ----------
        try:
            X_tr_dbn, y_tr_dbn = maybe_sample(X_tr_full, y_tr_full, dbn_sample_n)
            m_dbn, inf_dbn, disc, Xd_tr = train_discrete_bn(
                X_tr_dbn, y_tr_dbn, bins=discretize_bins, structure=dbn_structure, max_indegree=3
            )
            cat_cols, num_cols = m_dbn._dbn_disc_cols
            # Prepare test rows: DO NOT discretize in advance; dbn_predict_proba will discretize each numeric value
            p1, t_inf = [], 0.0
            for _, row in X_te.iterrows():
                t0 = time.time()
                p = dbn_predict_proba(inf_dbn, m_dbn._disc_transformer, cat_cols, num_cols, row)
                t_inf += (time.time() - t0)
                p1.append(p)
            p1 = np.array(p1, dtype=float)
            m = evaluate_probs(y_te.values, p1)
            rows.append(dict(model="DBN", fold=fold_idx, **m.__dict__))
            t_struct, t_params = getattr(m_dbn, "_fit_times", (np.nan, np.nan))
            timings.append(dict(model="DBN", fold=fold_idx, train_structure_sec=t_struct, train_params_sec=t_params, infer_sec=t_inf/len(X_te)))
        except Exception as e:
            rows.append(dict(model="DBN", fold=fold_idx, accuracy=np.nan, roc_auc=np.nan, brier=np.nan, logloss=np.nan, ece=np.nan, kl_rel=np.nan))
            timings.append(dict(model="DBN", fold=fold_idx, train_structure_sec=np.nan, train_params_sec=np.nan, infer_sec=np.nan))

        # ---------- Gaussian BN ----------
        try:
            X_tr_gbn, y_tr_gbn = X_tr_full, y_tr_full  # full set is usually fine
            gbn = train_gaussian_bn(X_tr_gbn, y_tr_gbn)
            p1, t_inf = [], 0.0
            for _, row in X_te.iterrows():
                t0 = time.time()
                p = gbn_predict_proba(gbn, row.to_dict())
                t_inf += time.time() - t0
                p1.append(p)
            p1 = np.array(p1, dtype=float)
            m = evaluate_probs(y_te.values, p1)
            rows.append(dict(model="GBN", fold=fold_idx, **m.__dict__))
            timings.append(dict(model="GBN", fold=fold_idx, train_structure_sec=getattr(gbn, "_fit_time", np.nan), train_params_sec=0.0, infer_sec=t_inf/len(X_te)))
        except Exception as e:
            rows.append(dict(model="GBN", fold=fold_idx, accuracy=np.nan, roc_auc=np.nan, brier=np.nan, logloss=np.nan, ece=np.nan, kl_rel=np.nan))
            timings.append(dict(model="GBN", fold=fold_idx, train_structure_sec=np.nan, train_params_sec=np.nan, infer_sec=np.nan))

        # ---------- Gaussian Process ----------
        try:
            X_tr_gp, y_tr_gp = maybe_sample(X_tr_full, y_tr_full, gp_max_train_n)
            gp_pipe, _ = fit_gp(X_tr_gp, y_tr_gp)
            p1, t_inf = gp_predict_proba(gp_pipe, X_te)
            m = evaluate_probs(y_te.values, p1)
            rows.append(dict(model="GP", fold=fold_idx, **m.__dict__))
            timings.append(dict(model="GP", fold=fold_idx, train_structure_sec=gp_pipe._fit_time, train_params_sec=0.0, infer_sec=t_inf/len(X_te)))
        except Exception as e:
            rows.append(dict(model="GP", fold=fold_idx, accuracy=np.nan, roc_auc=np.nan, brier=np.nan, logloss=np.nan, ece=np.nan, kl_rel=np.nan))
            timings.append(dict(model="GP", fold=fold_idx, train_structure_sec=np.nan, train_params_sec=np.nan, infer_sec=np.nan))

    metrics_df = pd.DataFrame(rows)
    timings_df = pd.DataFrame(timings)
    # Summary with flattened columns
    summary_df = metrics_df.groupby("model")[["accuracy","roc_auc","brier","logloss","ece","kl_rel"]].agg(["mean","std"])
    summary_df.columns = ["_".join(c) for c in summary_df.columns]
    summary_df = summary_df.reset_index()
    return metrics_df, timings_df, summary_df


# ==========================
#  I/O helpers
# ==========================

def load_fraud_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Fraud_Label" not in df.columns:
        candidates = [c for c in df.columns if "fraud" in c.lower() and ("label" in c.lower() or "flag" in c.lower() or c.lower() in ("isfraud","is_fraud"))]
        if candidates:
            df = df.rename(columns={candidates[0]: "Fraud_Label"})
        else:
            raise ValueError("Couldn't find Fraud_Label in fraud CSV.")
    return df

def load_heart_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "heart_disease" in df.columns:
        return df
    if "target" in df.columns:
        df = df.rename(columns={"target":"heart_disease"})
    elif "HeartDisease" in df.columns:
        df = df.rename(columns={"HeartDisease":"heart_disease"})
    else:
        candidates = [c for c in df.columns if "disease" in c.lower() or c.lower() in ("num","output","target")]
        if candidates:
            df = df.rename(columns={candidates[0]:"heart_disease"})
        else:
            raise ValueError("Couldn't find heart_disease target column.")
    return df

def save_tables(outdir: str, dataset_name: str, metrics_df: pd.DataFrame, timings_df: pd.DataFrame, summary_df: pd.DataFrame):
    os.makedirs(outdir, exist_ok=True)
    metrics_path = os.path.join(outdir, f"metrics_{dataset_name}.csv")
    timings_path = os.path.join(outdir, f"timings_{dataset_name}.csv")
    summary_path = os.path.join(outdir, f"summary_{dataset_name}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    timings_df.to_csv(timings_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] {metrics_path}\n[Saved] {timings_path}\n[Saved] {summary_path}")


# ==========================
#  Posterior Query helpers
# ==========================

def query_posterior_discrete_bn(
    df: pd.DataFrame, target_col: str, evidence: Dict[str, Any], bins: int = 5, structure: str = "hc"
) -> Dict[str, float]:
    X, y = split_features_target(df, target_col)
    model, infer, disc, Xd = train_discrete_bn(X, y, bins=bins, structure=structure)
    cat_cols, num_cols = model._dbn_disc_cols
    # dbn_predict_proba discretizes numeric evidence consistently
    p1 = dbn_predict_proba(infer, disc, cat_cols, num_cols, pd.Series(evidence))
    return {f"P({target_col}=1 | evidence)": float(p1), f"P({target_col}=0 | evidence)": float(1-p1)}

def query_posterior_gaussian_bn(df: pd.DataFrame, target_col: str, evidence: Dict[str, Any]) -> Dict[str, float]:
    X, y = split_features_target(df, target_col)
    model = train_gaussian_bn(X, y)
    p1 = gbn_predict_proba(model, evidence)
    return {f"P({target_col}=1 | evidence)": float(p1), f"P({target_col}=0 | evidence)": float(1-p1)}


# ==========================
#  CLI
# ==========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraud_csv", type=str, default=None, help="Path to fraud CSV")
    ap.add_argument("--heart_csv", type=str, default=None, help="Path to heart CSV")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--discretize_bins", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)

    # BN structure choice and scalability knobs
    ap.add_argument("--dbn_structure", type=str, default="hc", choices=["hc","chowliu"], help="DBN structure learner: hill-climb or custom Chow–Liu")
    ap.add_argument("--dbn_sample_n", type=int, default=10000, help="Optional max train rows per fold for DBN to ensure finish (None to disable)")
    ap.add_argument("--gp_max_train_n", type=int, default=20000, help="Optional max train rows per fold for GP to ensure finish (None to disable)")

    # Posterior query
    ap.add_argument("--query_json", type=str, default=None, help='JSON: {"dataset":"fraud|heart","model":"dbn|gbn|gp","target":"...","evidence":{...}}')

    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    if args.query_json:
        q = json.loads(args.query_json)
        dataset = q["dataset"]
        model = q["model"]
        target = q.get("target", "Fraud_Label" if dataset=="fraud" else "heart_disease")
        evidence = q["evidence"]

        if dataset == "fraud":
            if not args.fraud_csv: raise SystemExit("--fraud_csv is required for fraud query")
            df = load_fraud_csv(args.fraud_csv)
        elif dataset == "heart":
            if not args.heart_csv: raise SystemExit("--heart_csv is required for heart query")
            df = load_heart_csv(args.heart_csv)
        else:
            raise SystemExit("dataset must be 'fraud' or 'heart'")

        if model == "dbn":
            out = query_posterior_discrete_bn(df, target, evidence, bins=args.discretize_bins, structure=args.dbn_structure)
        elif model == "gbn":
            out = query_posterior_gaussian_bn(df, target, evidence)
        elif model == "gp":
            # Fit GP on all (or capped) data, then predict for evidence row
            X, y = split_features_target(df, target)
            if args.gp_max_train_n is not None and len(X) > args.gp_max_train_n:
                rs = np.random.RandomState(args.random_state)
                idx = rs.choice(len(X), args.gp_max_train_n, replace=False)
                X = X.iloc[idx]; y = y.iloc[idx]
            pipe, _ = fit_gp(X, y)
            row = {c: evidence.get(c, np.nan) for c in X.columns}
            proba = pipe.predict_proba(pd.DataFrame([row]))[0,1]
            out = {f"P({target}=1 | evidence)": float(proba), f"P({target}=0 | evidence)": float(1-proba)}
        else:
            raise SystemExit("model must be one of: dbn, gbn, gp")

        print(json.dumps(out, indent=2))
        return

    if args.fraud_csv:
        df_fraud = load_fraud_csv(args.fraud_csv)
        metrics, timings, summary = cross_validate_models(
            df_fraud, "Fraud_Label", folds=args.folds, discretize_bins=args.discretize_bins, random_state=args.random_state,
            dataset_name="fraud", dbn_structure=args.dbn_structure, dbn_sample_n=args.dbn_sample_n, gp_max_train_n=args.gp_max_train_n
        )
        save_tables(args.outdir, "fraud", metrics, timings, summary)

    if args.heart_csv:
        df_heart = load_heart_csv(args.heart_csv)
        metrics, timings, summary = cross_validate_models(
            df_heart, "heart_disease", folds=args.folds, discretize_bins=args.discretize_bins, random_state=args.random_state,
            dataset_name="heart", dbn_structure=args.dbn_structure, dbn_sample_n=None, gp_max_train_n=None
        )
        save_tables(args.outdir, "heart", metrics, timings, summary)

    if not args.fraud_csv and not args.heart_csv and not args.query_json:
        print("Nothing to do. Provide --fraud_csv and/or --heart_csv to run CV, or --query_json to query a posterior.")

if __name__ == "__main__":
    main()
