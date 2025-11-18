# Discrete Bayesian Network - dbn
# Gaussian Bayesian Network - gbn
# Gaussian Process - gp
# terminal command examples: 
# python pipeline3.py --model dbn --fraud_csv path/to/fraud.csv --folds 5 --dbn_structure hc
# python pipeline3.py --model dbn --fraud_csv path/to/fraud.csv --folds 5 --dbn_structure chowliu
# hill-climb uses K2 score, custom built chow-liu builds max spanning tree using mutual information
# python pipeline3.py --model gbn --fraud_csv path/to/fraud.csv --folds 5
# python pipeline3.py --model gp --fraud_csv path/to/fraud.csv --folds 5 --gp_max_train_n 1000
# posterior query (uses model in json): 
# python pipeline.py --fraud_csv path/to/fraud.csv --query_file query.json


import argparse, json, time, os, warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

#sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.impute import SimpleImputer

#GP Classifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#Discrete BN
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

#Gaussian/Mixed BN (pomegranate) — per your environment
from pomegranate.bayesian_network import BayesianNetwork as PomBayesNet



#Error logging

def _log_error(outdir: str, dataset: str, model: str, fold: int, err: Exception):
    """Append a simple line-per-error to outputs/errors_<dataset>.log for debugging."""
    try:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f"errors_{dataset}.log"), "a", encoding="utf-8") as f:
            f.write(f"[{model}][fold {fold}] {type(err).__name__}: {err}\n")
    except Exception:
        pass


#Columns that are truly ID-like (remove). Keep real predictive flags.
ID_LIKE_COLS = {
    "Transaction_ID",
    "User_ID",
    "session_id", "id", "uid", "uuid"
}

@dataclass
class Metrics:
    #container for evaluation metrics on probability predictions
    accuracy: float
    roc_auc: float
    brier: float
    logloss: float
    ece: float
    kl_rel: float

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    #ece - average gap between confidence and accuracy across bins
    y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitise(y_prob, bins) - 1
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
    #KL(P||Q) between observed positive rate and mean predicted prob in each non-empty bin, return 0.0 (instead of NaN) if too few bins are populated to compute meaningful score
    y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitise(y_prob, bins) - 1
    p_obs, p_pred = [], []
    for b in range(n_bins):
        mask = inds == b
        if np.any(mask):
            p_obs.append(np.clip(y_true[mask].mean(), 1e-6, 1-1e-6))
            p_pred.append(np.clip(y_prob[mask].mean(), 1e-6, 1-1e-6))
    if len(p_obs) < 2:
        return 0.0  #prevents NaN in sparse bins
    p_obs = np.array(p_obs); p_pred = np.array(p_pred)
    return float(np.sum(p_obs * (np.log(p_obs) - np.log(p_pred))))

def ensure_binary_labels(y: pd.Series) -> pd.Series:
    #convert common binary labels to 0/1, raises value error if not binary
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
    #split dataframe into features X and binary target y
    y = ensure_binary_labels(df[target_col].copy())
    X = df.drop(columns=[target_col]).copy()
    return X, y

def infer_column_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    #split columns into categorical vs numeric, tries coercion for ambiguous object columns
    cat_cols, num_cols = [], []
    for c in df.columns:
        if c == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            #try numeric coercion, if works, treat as numeric
            try:
                pd.to_numeric(df[c])
                num_cols.append(c)
            except Exception:
                cat_cols.append(c)
    return cat_cols, num_cols

def group_rare_categories(X: pd.DataFrame, cat_cols: List[str], min_count: int = 20) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Any]]:
    #returns transformed frame, allowed category levels per column, and per column mode for fallback when neither other and missing exist
    X = X.copy()
    allowed: Dict[str, List[str]] = {}
    modes: Dict[str, Any] = {}
    for c in cat_cols:
        vc = X[c].astype("object").value_counts(dropna=False)
        rare = set(vc[vc < min_count].index)
        if rare:
            X[c] = X[c].apply(lambda v: "Other" if v in rare else v)
        X[c] = X[c].astype("object").fillna("Missing")
        allowed[c] = sorted(pd.unique(X[c].astype("object")))
        modes[c] = X[c].mode(dropna=False).iloc[0] if len(X[c]) else "Missing"
    return X, allowed, modes

def drop_id_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    #drop known ID-like columns
    cols_drop = [c for c in df.columns if c in ID_LIKE_COLS or c.lower() in ("transaction_id","user_id","id")]
    if cols_drop:
        return df.drop(columns=cols_drop)
    return df



#Discrete BN


def mutual_info_discrete(a: pd.Series, b: pd.Series) -> float:
    #compute mutual information
    xa = pd.Categorical(a.astype("object")).codes.astype("int64")
    xb = pd.Categorical(b.astype("object")).codes.astype("int64")
    
    tab = pd.crosstab(xa, xb)
    if tab.size == 0:
        return 0.0
    Pxy = tab.values / tab.values.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(Pxy>0, Pxy / (Px @ Py), 1.0)
        mi = np.sum(np.where(Pxy>0, Pxy * np.log(frac), 0.0))
    return float(max(mi, 0.0))

def chow_liu_tree_discrete(df_disc: pd.DataFrame) -> List[Tuple[str, str]]:
    #chow liu tree - maximum spanning tree, returns directed edges away from 'target'
    cols = list(df_disc.columns)
    assert "target" in cols
    #compute MI for all pairs
    n = len(cols)
    mi_edges = []
    for i in range(n):
        for j in range(i+1, n):
            c1, c2 = cols[i], cols[j]
            mi = mutual_info_discrete(df_disc[c1], df_disc[c2])
            mi_edges.append((mi, c1, c2))
    #kruskal for maximum spanning tree
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
    #direct edges from root using BFS
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
                directed.append((u, v))  #direction away from root
                queue.append(v)
    return directed



#Discrete BN (DBN)


def preprocess_for_dbn(X: pd.DataFrame, y: pd.Series, rare_min_count: int) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, List[str]], Dict[str, Any], Dict[str, float]]:
    #prepare data for discrete bn, drops id columns, imputes numerics and categoricals, groups rate categories
    #returns allowed category levels, per column nodes and numeric medians
    XY = pd.concat([X, y.rename("target")], axis=1)
    XY = drop_id_like_columns(XY)
    cat_cols, num_cols = infer_column_types(XY, "target")
    Xi = XY.drop(columns=["target"]).copy()

    num_medians: Dict[str, float] = {}
    if num_cols:
        Xi[num_cols] = Xi[num_cols].replace([np.inf, -np.inf], np.nan)
        Xi[num_cols] = Xi[num_cols].fillna(Xi[num_cols].median())
        num_medians = Xi[num_cols].median().to_dict()

    allowed_levels, col_modes = {}, {}
    if cat_cols:
        Xi, allowed_levels, col_modes = group_rare_categories(Xi, cat_cols, min_count=rare_min_count)

    return Xi, cat_cols, num_cols, allowed_levels, col_modes, num_medians


def discretise_numeric(X: pd.DataFrame, num_cols: List[str], bins: int) -> Tuple[pd.DataFrame, Optional[KBinsDiscretizer], List[str]]:
    #discretise numeric columns via KBins, returns transformed X, fitted discretiser, list of discretised columns
    if not num_cols:
        return X.copy(), None, []
    #drop constant columns before KBins
    disc_cols = [c for c in num_cols if X[c].nunique(dropna=True) > 1]
    Xw = X.drop(columns=list(set(num_cols) - set(disc_cols))) if len(disc_cols) != len(num_cols) else X.copy()
    if not disc_cols:
        return Xw.copy(), None, []
    disc = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
    Xd_num = pd.DataFrame(disc.fit_transform(Xw[disc_cols]), columns=disc_cols, index=X.index).astype(int)
    Xd = pd.concat([Xw.drop(columns=disc_cols), Xd_num], axis=1)
    return Xd, disc, disc_cols


def apply_discretiser_row(disc: KBinsDiscretizer, col: str, value) -> int:
    #transform single value with fitted KBinsDiscretizer
    val = pd.DataFrame({col: [value]})
    return int(disc.transform(val)[:, 0][0])

def _normalise_categorical_value(col: str, v: Any, allowed: Dict[str, List[str]], modes: Dict[str, Any]) -> Any:
    #map unseen categorical levels to other, or missing
    #prevents inference errors and NaNs in pgmpy
    if col not in allowed:
        return v
    levels = allowed[col]
    if v in levels:
        return v
    if "Other" in levels:
        return "Other"
    if "Missing" in levels:
        return "Missing"
    return modes.get(col, levels[0] if levels else v)

def train_discrete_bn(
    X: pd.DataFrame, y: pd.Series, bins: int, structure: str = "hc", max_indegree: int = 3, max_iter: int = 200
) -> Tuple[BayesianNetwork, VariableElimination, Optional[KBinsDiscretizer]]:
    #train pgmpy discrete bn with structure learning and MLE parameters, stores training artefacts on model for robust inference
    #  - _fit_times: (structure_sec, params_sec)
    #  - _dbn_disc_cols: (cat_cols, num_cols)
    #  - _disc_transformer: KBinsDiscretizer for numeric columns
    #  - _cat_allowed_levels: {col: [levels]}
    #  - _cat_modes: {col: mode_value}

    Xi, cat_cols, num_cols, allowed_levels, col_modes, num_medians = preprocess_for_dbn(X, y, rare_min_count=20)
    Xd, disc, disc_cols = discretise_numeric(Xi, num_cols, bins=bins)
    data_d = pd.concat([Xd, y.rename("target")], axis=1)

    #Structure learning 
    t0 = time.time()
    if structure == "hc":
        est = HillClimbSearch(data_d)
        learned = est.estimate(
            scoring_method=K2Score(data_d),
            max_indegree=max_indegree,
            max_iter=max_iter
        )
    elif structure == "chowliu":
        #build chowliu tree based on discretised data
        learned_edges = chow_liu_tree_discrete(data_d)
        learned_nodes = list(data_d.columns)
    else:
        raise ValueError("structure must be one of {'hc','chowliu'}")
    t_struct = time.time() - t0

    #build bayesian network based on structure
    if structure == "hc":
        learned_nodes = list(learned.nodes())
        learned_edges = list(learned.edges())
    model = BayesianNetwork()
    model.add_nodes_from(learned_nodes)
    model.add_edges_from(learned_edges)

    #parameter learning
    t0 = time.time()
    model.fit(data_d, estimator=MaximumLikelihoodEstimator)  #CPTs
    t_params = time.time() - t0

    infer = VariableElimination(model)
    #store artefacts
    model._fit_times = (t_struct, t_params)
    model._dbn_disc_cols = (cat_cols, num_cols)
    model._disc_transformer = disc
    model._disc_cols = disc_cols
    model._num_medians = num_medians
    model._cat_allowed_levels = allowed_levels
    model._cat_modes = col_modes
    return model, infer, disc

def dbn_predict_proba(infer: VariableElimination, model: BayesianNetwork, row: pd.Series) -> float:
    #predict P(target=1|evidence=row) for discrete bn, only pass evidence for variables that exist in learned bn
    cat_cols, num_cols = model._dbn_disc_cols
    disc = getattr(model, "_disc_transformer", None)
    disc_cols: List[str] = getattr(model, "_disc_cols", [])
    num_medians: Dict[str, float] = getattr(model, "_num_medians", {})
    allowed = getattr(model, "_cat_allowed_levels", {})
    modes = getattr(model, "_cat_modes", {})

    model_vars = set(model.nodes())            #pgmpy api
    model_feats = model_vars.difference({"target"})

    evidence: Dict[str, Any] = {}

    #handle numeric columns that were discretised together
    if disc is not None and disc_cols:
        vals = {}
        for c in disc_cols:
            v = row.get(c, np.nan)
            if pd.isna(v):
                v = num_medians.get(c, 0.0)
            vals[c] = v
        df1 = pd.DataFrame([vals], columns=disc_cols)
        bins_vec = disc.transform(df1)[0].astype(int)
        disc_map = dict(zip(disc_cols, bins_vec))
    else:
        disc_map = {}

    #build evidence dict, filtering to features that exist in learned BN
    for c, v in row.items():
        if c not in model_feats:
            continue
        if c in disc_map:
            evidence[c] = int(disc_map[c])
        elif c in cat_cols:
            evidence[c] = _normalise_categorical_value(c, v, allowed, modes)
        else:
            evidence[c] = v

    q = infer.query(variables=["target"], evidence=evidence, show_progress=False)
    try:
        return float(q.values[1])
    except Exception:
        states = getattr(q, "state_names", {}).get("target", [0, 1])
        if isinstance(states, list) and 1 in states:
            idx = states.index(1)
            return float(q.values[idx])
        return float(np.argmax(q.values))



#gaussian BN (GBN)

def preprocess_for_gbn(X: pd.DataFrame, y: pd.Series, rare_min_count: int) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Any], Dict[str, Dict[Any, int]]]:
    #prepare data for the pomegranate BN, but ensure the
    #final matrix is fully numeric (no object dtype), since older pomegranate builds require numeric arrays:
    #  - drop ID columns
    #  - Impute numeric (median)
    #  - Group rare categorical levels 'Other' and fill Missing
    #  - Encode categorical columns to integer codes and return per-column codebooks
    #returns:
    #  df_num: numeric-only DataFrame including features
    #  allowed_levels: {col: [allowed levels after grouping]}
    #  col_modes: {col: mode value used as fallback}
    #  cat_codebooks: {col: {level -> int}}

    df_all = pd.concat([X, y.rename("target")], axis=1)
    df_all = drop_id_like_columns(df_all)
    cat_cols, num_cols = infer_column_types(df_all, "target")

    df = df_all.drop(columns=["target"]).copy()

    #numeric: coerce + impute median
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    #categorical: group rare -> 'Other', fill Missing, ensure object dtype
    allowed_levels, col_modes = {}, {}
    if cat_cols:
        df[cat_cols], allowed_levels, col_modes = group_rare_categories(df[cat_cols], cat_cols, min_count=rare_min_count)
        for c in cat_cols:
            df[c] = df[c].astype("object")

    #encode categoricals to integer codes (stable order = allowed_levels)
    cat_codebooks: Dict[str, Dict[Any, int]] = {}
    if cat_cols:
        for c in cat_cols:
            levels = allowed_levels.get(c, sorted(pd.unique(df[c].astype("object"))))
            codebook = {lvl: i for i, lvl in enumerate(levels)}
            cat_codebooks[c] = codebook
            def _map_val(v):
                if v in codebook:
                    return codebook[v]
                if "Other" in codebook:
                    return codebook["Other"]
                if "Missing" in codebook:
                    return codebook["Missing"]
                return codebook.get(col_modes.get(c, levels[0]), 0)
            df[c] = df[c].map(_map_val).astype("int64")

    #ensure numeric dtypes are float64 for safety
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    return df, allowed_levels, col_modes, cat_codebooks


def train_gaussian_bn(X: pd.DataFrame, y: pd.Series) -> PomBayesNet:
    #train bn with only numeric matrix for compatibility
    # stores:
    #  - _fit_time
    #  - _columns (feature + target names)
    #  - _cat_allowed_levels (for robust mapping)
    #  - _cat_modes (fallbacks)
    #  - _cat_codebooks ({col: {level -> int}})
    #  - _is_target_int (True)
    y = ensure_binary_labels(y).astype(int)

    #df_num has only features concat target later
    df_num, allowed, modes, codebooks = preprocess_for_gbn(X, y, rare_min_count=20)

    #target is int {0,1}; keep as numeric column
    df_num = df_num.copy()
    df_num["target"] = y.astype("int64")

    state_names = list(df_num.columns)      #features + target
    data = df_num.values                    #fully numeric 2D array

    t0 = time.time()
    model = None
    #option 1 - classmethod
    try:
        model = PomBayesNet.from_samples(
            data, algorithm="chow-liu", state_names=state_names
        )
    except AttributeError:
        #option 2: instance.from_samples()
        try:
            m = PomBayesNet()
            ret = m.from_samples(data, algorithm="chow-liu", state_names=state_names)
            model = m if ret is None else ret
        except AttributeError:
            #option 3: fit() variants
            m = PomBayesNet()
            ok = False
            try:
                m.fit(data, algorithm="chow-liu", state_names=state_names)
                ok = True
            except TypeError:
                pass
            if not ok:
                m.fit(data)
            model = m

    model._fit_time = time.time() - t0
    model._columns = state_names
    model._cat_allowed_levels = allowed
    model._cat_modes = modes
    model._cat_codebooks = codebooks
    model._is_target_int = True
    return model


def _gbn_normalise_row(model: PomBayesNet, x_row: Dict[str, Any]) -> Dict[str, Any]:
    #normalise single evidence row for pomegranate bn
    #maps unseen categorical levels using allowed levels/modes
    #encodes categoricals to int codes using stored codebooks
    allowed = getattr(model, "_cat_allowed_levels", {})
    modes = getattr(model, "_cat_modes", {})
    codebooks = getattr(model, "_cat_codebooks", {})

    out: Dict[str, Any] = {}
    for k, v in x_row.items():
        if k in codebooks:
            levels = allowed.get(k, [])
            if v not in levels:
                if "Other" in levels:
                    v = "Other"
                elif "Missing" in levels:
                    v = "Missing"
                else:
                    v = modes.get(k, levels[0] if levels else v)
            cb = codebooks[k]
            out[k] = cb.get(v, cb.get("Other", cb.get("Missing", cb.get(modes.get(k, next(iter(cb.keys()))), 0))))
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = np.nan
    return out


def _gbn_predict_distributions(model: PomBayesNet, obs_dict: Dict[str, Any], state_names: List[str]):
    #robust wrapper, try dict-based call, full-width array, array without target (if width error), len(model.marginals) and len(model.states) as final fallback
    #returns flat list of distributions/values for one row, or None on failure
    #ensure baked and get sizes/orders
    try:
        if not getattr(model, "baked", False):
            model.bake()
    except Exception:
        try:
            model.bake()
        except Exception:
            pass

    #discover order/length
    try:
        baked_order = [s.name for s in model.states]
    except Exception:
        baked_order = list(getattr(model, "_columns", [])) or state_names
    n_all = len(baked_order) if baked_order else len(state_names)
    order = baked_order or state_names

    d = None #initialise to never get UnboundLocalError

    #dict-based call
    try:
        d_try = model.predict_proba({k: v for k, v in obs_dict.items() if k != "target"})
        if isinstance(d_try, list) and len(d_try) == n_all:
            return d_try
        if isinstance(d_try, list) and len(d_try) == 1 and isinstance(d_try[0], list) and len(d_try[0]) == n_all:
            return d_try[0]
    except Exception:
        pass

    def _ordered_array(cols, include_target=True):
        vals = []
        for name in cols:
            if name == "target":
                vals.append(None if include_target else None)
            else:
                vals.append(obs_dict.get(name, None))
        return np.array([vals], dtype=object)

    #full width
    try:
        arr_full = _ordered_array(order, include_target=True)
        d_try = model.predict_proba(arr_full)
        if isinstance(d_try, list) and len(d_try) == 1 and isinstance(d_try[0], list) and len(d_try[0]) == n_all:
            return d_try[0]
        if isinstance(d_try, list) and len(d_try) == n_all:
            return d_try
    except ValueError as e:
        #width mismatch, fall through to next shapes
        if "X.shape[1]" not in str(e):
            #different error, keep last error object for debugging
            d = None
    except Exception:
        pass

    #without target
    try:
        cols_no_target = [c for c in order if c != "target"]
        arr_nt = _ordered_array(cols_no_target, include_target=False)
        d_try = model.predict_proba(arr_nt)
        if isinstance(d_try, list) and len(d_try) == 1 and isinstance(d_try[0], list):
            return d_try[0]
        if isinstance(d_try, list):
            return d_try
    except Exception:
        pass

    #explicit fallbacks to len(marginals) and len(states)
    try:
        n_m = len(getattr(model, "marginals", []))
        if n_m > 0:
            cols_m = order[:n_m]
            arr_m = _ordered_array(cols_m, include_target=("target" in cols_m))
            d_try = model.predict_proba(arr_m)
            if isinstance(d_try, list) and len(d_try) == 1 and isinstance(d_try[0], list):
                return d_try[0]
            if isinstance(d_try, list):
                return d_try
    except Exception:
        pass

    try:
        n_s = len(order)
        cols_s = order[:n_s]
        arr_s = _ordered_array(cols_s, include_target=("target" in cols_s))
        d_try = model.predict_proba(arr_s)
        if isinstance(d_try, list) and len(d_try) == 1 and isinstance(d_try[0], list):
            return d_try[0]
        if isinstance(d_try, list):
            return d_try
    except Exception:
        pass

    #nothing worked — return None so the can degrade gracefully
    return d  #intentionally none



def gbn_predict_proba(model: PomBayesNet, x_row: Dict[str, Any]) -> float:
    #predict P(target=1 | evidence=x_row), fall back to 0.5 if cannot get distribution, prevents crashing
    #normalise + encode
    x_row = _gbn_normalise_row(model, x_row)

    #ensure baked & get order
    try:
        if not getattr(model, "baked", False):
            model.bake()
    except Exception:
        try:
            model.bake()
        except Exception:
            pass

    try:
        order = [s.name for s in model.states]
    except Exception:
        order = list(getattr(model, "_columns", [])) or sorted(set(list(x_row.keys()) + ["target"]))

    #ensure target present for extraction
    if "target" not in order:
        order = list(order) + ["target"]
    try:
        t_idx = order.index("target")
    except ValueError:
        t_idx = len(order) - 1

    #build observation dict (never set target)
    obs = {name: (None if name == "target" else x_row.get(name, None)) for name in order}

    #get distributions using the robust wrapper
    dists = _gbn_predict_distributions(model, obs, order)

    #if wrapper failed, degrade to prior 0.5
    if dists is None:
        return 0.5

    #flatten single-row wrappers
    if isinstance(dists, list) and len(dists) == 1 and isinstance(dists[0], list):
        dists = dists[0]

    #pick the target spot (guard length)
    targ = dists[t_idx] if (isinstance(dists, list) and t_idx < len(dists)) else dists

    #extract probability for class 1
    params = getattr(targ, "parameters", [{}])[0] if hasattr(targ, "parameters") else {}
    if 1 in params:
        return float(params[1])
    if "1" in params:
        return float(params["1"])
    if isinstance(params, dict) and params:
        k = max(params, key=params.get)
        return float(params[k])

    #last resort: coerce to float or return neutral
    try:
        return float(targ)
    except Exception:
        return 0.5


#Gaussian Process (GP)

def make_gp_classifier() -> GaussianProcessClassifier:
    #build gp classifer with RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0)
    return GaussianProcessClassifier(kernel=kernel, random_state=0, n_restarts_optimiser=2, copy_X_train=False)

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
    #batch predict probabilities, return (proba, inference_time_sec)
    t0 = time.time()
    proba = pipe.predict_proba(X_te)[:, 1]
    return proba, time.time() - t0


#evaluation and CV
def evaluate_probs(y_true: np.ndarray, p1: np.ndarray) -> Metrics:
    #accuracy, roc auc, brier score, log loss, ece, kl reliability
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
    df: pd.DataFrame, target_col: str, folds: int, discretise_bins: int, random_state: int,
    dataset_name: str, dbn_structure: str, dbn_sample_n: Optional[int], gp_max_train_n: Optional[int],
    enabled_models: List[str], outdir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #run stratified k-fold cv for selected models, return results and timings dataframes
    X, y = split_features_target(df, target_col)

    #guard: least-populated class must be >= folds
    try:
        min_class = int(y.value_counts().min())
        if min_class < folds:
            warnings.warn(f"Reducing folds from {folds} to {min_class} due to class counts.")
            folds = max(2, min_class)
    except Exception:
        pass

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    rows, timings = [], []
    fold_idx = 0

    for tr_idx, te_idx in skf.split(X, y):
        fold_idx += 1
        X_tr_full, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr_full, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

        #optional down-sampling for scalability on very large sets
        def maybe_sample(Xd, yd, n):
            if n is not None and len(Xd) > n:
                rs = np.random.RandomState(random_state + fold_idx)
                idx = rs.choice(len(Xd), n, replace=False)
                return Xd.iloc[idx].copy(), yd.iloc[idx].copy()
            return Xd, yd

        #discrete bn
        if "DBN" in enabled_models:
            try:
                X_tr_dbn, y_tr_dbn = maybe_sample(X_tr_full, y_tr_full, dbn_sample_n)
                m_dbn, inf_dbn, _ = train_discrete_bn(
                    X_tr_dbn, y_tr_dbn, bins=discretise_bins, structure=dbn_structure, max_indegree=3
                )
                #per row inference with robust evidence normalisation and key filtering
                p1, t_inf = [], 0.0
                for _, row in X_te.iterrows():
                    t0 = time.time()
                    p = dbn_predict_proba(inf_dbn, m_dbn, row)
                    t_inf += (time.time() - t0)
                    p1.append(p)
                p1 = np.array(p1, dtype=float)
                m = evaluate_probs(y_te.values, p1)
                rows.append(dict(model="DBN", fold=fold_idx, **m.__dict__))
                t_struct, t_params = getattr(m_dbn, "_fit_times", (np.nan, np.nan))
                timings.append(dict(model="DBN", fold=fold_idx, train_structure_sec=t_struct, train_params_sec=t_params, infer_sec=t_inf/len(X_te)))
            except Exception as e:
                _log_error(outdir=outdir, dataset=dataset_name, model="DBN", fold=fold_idx, err=e)
                rows.append(dict(model="DBN", fold=fold_idx, accuracy=np.nan, roc_auc=np.nan, brier=np.nan, logloss=np.nan, ece=np.nan, kl_rel=np.nan))
                timings.append(dict(model="DBN", fold=fold_idx, train_structure_sec=np.nan, train_params_sec=np.nan, infer_sec=np.nan))

        #gaussian bn
        if "GBN" in enabled_models:
            try:
                gbn = train_gaussian_bn(X_tr_full, y_tr_full)
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
                _log_error(outdir=outdir, dataset=dataset_name, model="GBN", fold=fold_idx, err=e)
                rows.append(dict(model="GBN", fold=fold_idx, accuracy=np.nan, roc_auc=np.nan, brier=np.nan, logloss=np.nan, ece=np.nan, kl_rel=np.nan))
                timings.append(dict(model="GBN", fold=fold_idx, train_structure_sec=np.nan, train_params_sec=np.nan, infer_sec=np.nan))

        #gaussian process
        if "GP" in enabled_models:
            try:
                X_tr_gp, y_tr_gp = maybe_sample(X_tr_full, y_tr_full, gp_max_train_n)
                gp_pipe, _ = fit_gp(X_tr_gp, y_tr_gp)
                p1, t_inf = gp_predict_proba(gp_pipe, X_te)
                m = evaluate_probs(y_te.values, p1)
                rows.append(dict(model="GP", fold=fold_idx, **m.__dict__))
                timings.append(dict(model="GP", fold=fold_idx, train_structure_sec=gp_pipe._fit_time, train_params_sec=0.0, infer_sec=t_inf/len(X_te)))
            except Exception as e:
                _log_error(outdir=outdir, dataset=dataset_name, model="GP", fold=fold_idx, err=e)
                rows.append(dict(model="GP", fold=fold_idx, accuracy=np.nan, roc_auc=np.nan, brier=np.nan, logloss=np.nan, ece=np.nan, kl_rel=np.nan))
                timings.append(dict(model="GP", fold=fold_idx, train_structure_sec=np.nan, train_params_sec=np.nan, infer_sec=np.nan))

    #build result tables
    metrics_df = pd.DataFrame(rows)
    timings_df = pd.DataFrame(timings)

    if len(metrics_df):
        summary_df = (metrics_df.groupby("model")[["accuracy","roc_auc","brier","logloss","ece","kl_rel"]]
                      .agg(["mean","std"]))
        summary_df.columns = ["_".join(c) for c in summary_df.columns]
        summary_df = summary_df.reset_index()
    else:
        summary_df = pd.DataFrame(columns=[
            "model",
            "accuracy_mean","accuracy_std","roc_auc_mean","roc_auc_std",
            "brier_mean","brier_std","logloss_mean","logloss_std",
            "ece_mean","ece_std","kl_rel_mean","kl_rel_std"
        ])

    return metrics_df, timings_df, summary_df



#Fraud dtype normalisation
def _parse_timestamp_features(df: pd.DataFrame, col: str = "Timestamp") -> pd.DataFrame:
    #parse timestamp colum to datetime, drop raw timestamp to avoid unseen category errors
    if col not in df.columns:
        return df
    ts = pd.to_datetime(df[col], errors="coerce", utc=False)
    df["ts_hour"] = ts.dt.hour.astype("Int64")       #0-23
    df["ts_dow"] = ts.dt.dayofweek.astype("Int64")   #0=Mon..6=Sun
    df["ts_month"] = ts.dt.month.astype("Int64")     #1..12
    df["ts_dom"] = ts.dt.day.astype("Int64")         #1..31
    df["ts_daypart"] = pd.cut(df["ts_hour"].astype("float"),
                              bins=[-0.1,6,12,18,24],
                              labels=["night","morning","afternoon","evening"]).astype("object")
    return df.drop(columns=[col])

def _coerce_binary_flag(s: pd.Series) -> pd.Series:
    #ensure common binary flags become {0,1}
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    mapping = {"no":0,"yes":1,"false":0,"true":1,"n":0,"y":1,"0":0,"1":1}
    if s.dtype == "O":
        sm = s.astype(str).str.strip().str.lower().map(mapping)
        if sm.notna().mean() > 0.95:
            return sm.fillna(0).astype(int)
    try:
        sn = pd.to_numeric(s, errors="coerce")
        if sn.dropna().nunique() <= 2:
            return sn.fillna(0).astype(int)
    except Exception:
        pass
    return s

def normalise_fraud_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    #drop id columns, parse timestamp, enforce numeric types, keep relevant flags like IP_Address_Flag
    df = df.drop(columns=[c for c in df.columns if c in ID_LIKE_COLS], errors="ignore")
    df = _parse_timestamp_features(df, col="Timestamp")

    numeric_cols = [
        "Transaction_Amount", "Account_Balance", "Avg_Transaction_Amount_7d",
        "Daily_Transaction_Count", "Failed_Transaction_Count_7d",
        "Card_Age", "Transaction_Distance", "Risk_Score"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["IP_Address_Flag", "Previous_Fraudulent_Activity", "Is_Weekend"]:
        if c in df.columns:
            df[c] = _coerce_binary_flag(df[c])

    if "Fraud_Label" in df.columns:
        df["Fraud_Label"] = _coerce_binary_flag(df["Fraud_Label"]).astype(int)

    for c in ["Transaction_Type","Device_Type","Location","Merchant_Category",
              "Card_Type","Authentication_Method","ts_daypart"]:
        if c in df.columns:
            df[c] = df[c].astype("object")

    return df



#I/O helpers
def load_fraud_csv(path: str) -> pd.DataFrame:
    #load csv and standardise, then normalise for stable training and inference
    df = pd.read_csv(path)
    if "Fraud_Label" not in df.columns:
        candidates = [c for c in df.columns if "fraud" in c.lower() and ("label" in c.lower() or "flag" in c.lower() or c.lower() in ("isfraud","is_fraud"))]
        if candidates:
            df = df.rename(columns={candidates[0]: "Fraud_Label"})
        else:
            raise ValueError("Couldn't find Fraud_Label in fraud CSV.")
    df = normalise_fraud_dtypes(df)
    return df

def load_heart_csv(path: str) -> pd.DataFrame:
    #load heart dataset and standardise, accept common target name variants
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

def save_tables(outdir: str, dataset_name: str,
                metrics_df: pd.DataFrame, timings_df: pd.DataFrame, summary_df: pd.DataFrame,
                file_suffix: str = ""):
    #write csv for metrics, timings and summary
    os.makedirs(outdir, exist_ok=True)
    metrics_path = os.path.join(outdir, f"metrics_{dataset_name}{file_suffix}.csv")
    timings_path = os.path.join(outdir, f"timings_{dataset_name}{file_suffix}.csv")
    summary_path = os.path.join(outdir, f"summary_{dataset_name}{file_suffix}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    timings_df.to_csv(timings_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] {metrics_path}\n[Saved] {timings_path}\n[Saved] {summary_path}")



#Posterior Query helpers
def query_posterior_discrete_bn(
    df: pd.DataFrame, target_col: str, evidence: Dict[str, Any], bins: int = 5, structure: str = "hc"
) -> Dict[str, float]:
    #train dbn and return posterior P(target|evidence)
    X, y = split_features_target(df, target_col)
    model, infer, _ = train_discrete_bn(X, y, bins=bins, structure=structure)
    p1 = dbn_predict_proba(infer, model, pd.Series(evidence))
    return {f"P({target_col}=1 | evidence)": float(p1), f"P({target_col}=0 | evidence)": float(1-p1)}

def query_posterior_gaussian_bn(df: pd.DataFrame, target_col: str, evidence: Dict[str, Any]) -> Dict[str, float]:
    #train gbn and return posterior P(target|evidence)
    X, y = split_features_target(df, target_col)
    model = train_gaussian_bn(X, y)
    p1 = gbn_predict_proba(model, evidence)
    return {f"P({target_col}=1 | evidence)": float(p1), f"P({target_col}=0 | evidence)": float(1-p1)}



#CLI
def _model_selection_arg_to_list(sel: str):
    sel = (sel or "all").lower()
    if sel == "dbn": return ["DBN"]
    if sel == "gbn": return ["GBN"]
    if sel == "gp":  return ["GP"]
    if sel == "all": return ["DBN","GBN","GP"]
    raise SystemExit("--model must be one of: dbn, gbn, gp, all")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraud_csv", type=str, default=None, help="Path to fraud CSV")
    ap.add_argument("--heart_csv", type=str, default=None, help="Path to heart CSV")
    ap.add_argument("--folds", type=int, default=5, help="Number of StratifiedKFold splits")
    ap.add_argument("--discretise_bins", type=int, default=5, help="KBins (quantile) for DBN numeric discretisation")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--model", type=str, default="all", choices=["dbn","gbn","gp","all"],
                    help="Select which model(s) to run: dbn | gbn | gp | all (default: all)")

    #BN structure choice and scalability knobs
    ap.add_argument("--dbn_structure", type=str, default="hc", choices=["hc","chowliu"],
                    help="DBN structure learner: hill-climb or custom Chow-Liu")
    ap.add_argument("--dbn_sample_n", type=lambda s: None if s=="None" else int(s), default=10000,
                    help="Max train rows per fold for DBN (use 'None' to disable)")
    ap.add_argument("--gp_max_train_n", type=lambda s: None if s=="None" else int(s), default=20000,
                    help="Max train rows per fold for GP (use 'None' to disable)")

    #Posterior query
    ap.add_argument("--query_file", type=str, default=None,
                help="Path to a JSON file containing the posterior query.")


    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    enabled_models = _model_selection_arg_to_list(args.model)
    model_suffix = "" if args.model == "all" else f"_{args.model.lower()}"

    # Posterior single-row query mode
    query_payload = None

    #Read from JSON file
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as fh:
            query_payload = fh.read()

    #Read inline JSON
    elif args.query_json:
        s = args.query_json.strip()
        # Strip outer quotes if shell added them
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()
        # Fix PowerShell escape sequences
        s = s.replace("`\"", '"')
        query_payload = s

    if query_payload:
        q = json.loads(query_payload)
        dataset = q["dataset"]
        model = q["model"]
        target = q.get("target", "Fraud_Label" if dataset == "fraud" else "heart_disease")
        evidence = q["evidence"]

        #load dataset based on type
        if dataset == "fraud":
            if not args.fraud_csv:
                raise SystemExit("--fraud_csv is required for fraud query")
            df = load_fraud_csv(args.fraud_csv)
        elif dataset == "heart":
            if not args.heart_csv:
                raise SystemExit("--heart_csv is required for heart query")
            df = load_heart_csv(args.heart_csv)
        else:
            raise SystemExit("dataset must be 'fraud' or 'heart'")

        #run posterior inference based on model
        if model == "dbn":
            out = query_posterior_discrete_bn(
                df, target, evidence, bins=args.discretise_bins, structure=args.dbn_structure
            )
        elif model == "gbn":
            out = query_posterior_gaussian_bn(df, target, evidence)
        elif model == "gp":
            X, y = split_features_target(df, target)
            if args.gp_max_train_n is not None and len(X) > args.gp_max_train_n:
                rs = np.random.RandomState(args.random_state)
                idx = rs.choice(len(X), args.gp_max_train_n, replace=False)
                X = X.iloc[idx]; y = y.iloc[idx]
            pipe, _ = fit_gp(X, y)
            row = {c: evidence.get(c, np.nan) for c in X.columns}
            proba = pipe.predict_proba(pd.DataFrame([row]))[0, 1]
            out = {
                f"P({target}=1 | evidence)": float(proba),
                f"P({target}=0 | evidence)": float(1 - proba),
            }
        else:
            raise SystemExit("model must be one of: dbn, gbn, gp")

        print(json.dumps(out, indent=2))
        return


    #Cross-validation mode
    if args.fraud_csv:
        df_fraud = load_fraud_csv(args.fraud_csv)
        metrics, timings, summary = cross_validate_models(
            df_fraud, "Fraud_Label", folds=args.folds, discretise_bins=args.discretise_bins, random_state=args.random_state,
            dataset_name="fraud", dbn_structure=args.dbn_structure, dbn_sample_n=args.dbn_sample_n,
            gp_max_train_n=args.gp_max_train_n, enabled_models=enabled_models, outdir=args.outdir
        )
        save_tables(args.outdir, "fraud", metrics, timings, summary, file_suffix=model_suffix)

    if args.heart_csv:
        df_heart = load_heart_csv(args.heart_csv)
        metrics, timings, summary = cross_validate_models(
            df_heart, "heart_disease", folds=args.folds, discretise_bins=args.discretise_bins, random_state=args.random_state,
            dataset_name="heart", dbn_structure=args.dbn_structure, dbn_sample_n=None, gp_max_train_n=None,
            enabled_models=enabled_models, outdir=args.outdir
        )
        save_tables(args.outdir, "heart", metrics, timings, summary, file_suffix=model_suffix)

    if not args.fraud_csv and not args.heart_csv and not args.query_json:
        print("Nothing to do. Provide --fraud_csv and/or --heart_csv to run CV, or --query_json to query a posterior.")

if __name__ == "__main__":
    main()