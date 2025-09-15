# rfbid/core.py
"""Core utilities for rfBID: data processing, feature selection, validation and metrics."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from typing import Optional, Dict

def preprocess_olink(data: pd.DataFrame,
                     panel: Optional[str] = None,
                     exclude_controls: bool = True) -> pd.DataFrame:
    if exclude_controls:
        data = data[~data["SampleID"].str.contains("control", case=False, na=False)]
    if panel is None:
        panel = data["Panel"].astype("category").cat.categories.tolist()[0]
    return data[data["Panel"] == panel]

def pivot_assays(data: pd.DataFrame) -> pd.DataFrame:
    return data.pivot_table(index="SampleID", columns=["Assay"], values="NPX")

def extract_metadata(data: pd.DataFrame, meta_cols: list) -> pd.DataFrame:
    meta_df = data[meta_cols].drop_duplicates().set_index("SampleID")
    return meta_df

def selection_frequency(X_df, y, n_iter=100, top_k=30, seed=0, rf_kwargs=None):
    rng = np.random.default_rng(seed)
    rf_kwargs = rf_kwargs or {}
    features = X_df.columns.to_list()
    freq = np.zeros(len(features), dtype=int)

    for _ in range(n_iter):
        idx = rng.choice(len(X_df), size=len(X_df), replace=True)
        Xb, yb = X_df.iloc[idx], y.iloc[idx]
        rf = RandomForestClassifier(**rf_kwargs)
        rf.fit(Xb, yb)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:top_k]
        freq[top_idx] += 1
    return pd.Series(freq / n_iter, index=features)

def validate_markers(X_df: pd.DataFrame,
                     y: pd.Series,
                     selected: list,
                     test_size: float = 0.3,
                     random_state: int = 7,
                     proba_threshold: float = 0.5,
                     positive_label: Optional[str] = None) -> Dict[str, object]:
    """Validate selected markers with Logistic Regression (AUC + predictions).

    This function ENSURES labels are encoded consistently. If `positive_label` is provided,
    the positive class is created as (y == positive_label) -> 1, else LabelEncoder is used.

    Returns a dict containing numeric encoded y_val, y_score, y_pred, auc and optionally
    the fitted LabelEncoder (if used).
    """
    # y should be a pandas Series aligned with X_df.index
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X_df.index)
    else:
        # ensure same index as X_df
        y = y.reindex(X_df.index)

    # import pdb;pdb.set_trace()
    # Encode labels
    if positive_label is None:
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y), index=y.index)
    else:
        # explicit mapping: positive_label -> 1, others -> 0
        if positive_label not in y.unique():
            raise ValueError(f"positive_label '{positive_label}' not found in target values: {y.unique()}")
        y_enc = pd.Series((y == positive_label).astype(int), index=y.index)
        le = None

    # Split using encoded labels
    X_disc, X_val, y_disc_enc, y_val_enc = train_test_split(
        X_df, y_enc, stratify=y_enc, test_size=test_size, random_state=random_state
    )

    clf = LogisticRegression(max_iter=2000).fit(X_disc[selected], y_disc_enc)

    # find index of class '1' in classifier classes_
    classes = np.array(clf.classes_)
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        # if class '1' not present (e.g. degenerate), fall back to second column if exists
        if classes.size == 1:
            raise ValueError("Only one class present in training labels; cannot compute probabilities for positive class.")
        pos_idx = 1

    proba_all = clf.predict_proba(X_val[selected])
    y_score = proba_all[:, pos_idx]
    y_pred = (y_score >= proba_threshold).astype(int)
    auc = roc_auc_score(y_val_enc, y_score)

    out = {'auc': float(auc), 'y_val': y_val_enc.values, 'y_score': y_score, 'y_pred': y_pred}
    if le is not None:
        out['label_encoder'] = le
    return out

def compute_classification_metrics(y_true, y_pred):
    """Compute accuracy, F1, MCC and confusion matrix statistics.

    Expects `y_true` and `y_pred` to be numeric encoded arrays (0/1). 
    """
    # Convert to numpy arrays
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # # If y_true is not numeric, try to encode it
    # if y_true_arr.dtype.kind in ('U', 'S', 'O'):
    #     le = LabelEncoder()
    #     y_true_arr = le.fit_transform(y_true_arr)

    # # Ensure predictions are numeric
    # if y_pred_arr.dtype.kind in ('U', 'S', 'O'):
    #     # try to encode predictions with same encoder as y_true if available
    #     try:
    #         y_pred_arr = le.transform(y_pred_arr)
    #     except Exception:
    #         # last resort: coerce to int
    #         y_pred_arr = y_pred_arr.astype(int)
    # else:
    #     y_pred_arr = y_pred_arr.astype(int)

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    fig, ax = plt.subplots(figsize = (12,8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig('confusion_matrix.png')

    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true_arr, y_pred_arr)
    f1 = f1_score(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

    return {'accuracy': float(acc), 'f1': float(f1), 'mcc': float(mcc),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def plot_selection_frequency(freq, top_n=20, colormap='viridis', figsize=(12,6)):
    freq_sorted = freq.sort_values(ascending=False).head(top_n)
    cmap = cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(freq_sorted)))

    plt.figure(figsize=figsize)
    bars = plt.bar(freq_sorted.index, freq_sorted.values, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                 f"{height:.2f}", ha='center', va='bottom', fontsize=9)
    plt.ylabel('Selection Frequency')
    plt.xlabel('Assays / Features')
    plt.title(f"Top {top_n} Most Stable Features (Random Forest)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('selection_frequency.png')
