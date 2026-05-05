"""
utils/preprocessing.py
-----------------------
Handles merging, cleaning, encoding, and normalizing the IEEE-CIS dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ─── Column Groupings ───────────────────────────────────────────────────────

CARD_COLS       = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
EMAIL_COLS      = ['P_emaildomain', 'R_emaildomain']
ADDR_COLS       = ['addr1', 'addr2']
DEVICE_COLS     = ['DeviceType', 'DeviceInfo']
NUMERICAL_COLS  = ['TransactionAmt', 'dist1', 'dist2',
                   'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10',
                   'C11','C12','C13','C14',
                   'D1','D2','D3','D4','D5','D6','D7','D8','D9',
                   'D10','D11','D12','D13','D14','D15',
                   'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']


def load_and_merge(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """
    Load raw CSVs and left-join identity onto transactions.
    ~590k rows in transactions; ~140k have identity records.
    """
    print("[1/5] Loading data…")
    tx = pd.read_csv(transaction_path)
    id_ = pd.read_csv(identity_path)
    df = tx.merge(id_, on='TransactionID', how='left')
    print(f"      Merged shape: {df.shape}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - Numerical → median imputation (robust to outliers)
    - Categorical → fill with 'UNKNOWN' sentinel value
    - Columns with >90% missing → drop entirely
    """
    print("[2/5] Handling missing values…")

    # Drop ultra-sparse columns
    thresh = 0.90
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > thresh].index.tolist()
    # Keep target & ID
    drop_cols = [c for c in drop_cols if c not in ['isFraud', 'TransactionID']]
    df.drop(columns=drop_cols, inplace=True)
    print(f"      Dropped {len(drop_cols)} columns with >{thresh*100:.0f}% missing")

    # Separate types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['TransactionID', 'isFraud']]

    # Impute
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna('UNKNOWN')

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all object columns.
    Returns df and a dict of {col: LabelEncoder} for inverse transform later.
    """
    print("[3/5] Encoding categoricals…")
    encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    print(f"      Encoded {len(cat_cols)} columns")
    return df, encoders


def normalize_numericals(df: pd.DataFrame,
                          scaler: StandardScaler = None
                          ) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale numerical columns. Pass a fitted scaler for inference.
    Log-transform TransactionAmt first (heavy right skew).
    """
    print("[4/5] Normalizing numericals…")
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt'] = np.log1p(df['TransactionAmt'])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['TransactionID', 'isFraud']]

    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        # Align columns (inference may have fewer)
        common = [c for c in scaler.feature_names_in_ if c in df.columns]
        df[common] = scaler.transform(df[common])

    return df, scaler


def get_class_weights(df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute class weights for weighted cross-entropy to handle imbalance.
    Fraud rate is ~3.5% in this dataset.
    """
    counts = df['isFraud'].value_counts()
    total = len(df)
    w0 = total / (2 * counts[0])  # weight for non-fraud
    w1 = total / (2 * counts[1])  # weight for fraud
    print(f"[5/5] Class weights → non-fraud: {w0:.3f}, fraud: {w1:.3f}")
    return w0, w1


def preprocess_pipeline(transaction_path: str,
                         identity_path: str
                         ) -> tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Full pipeline: load → merge → missing → encode → normalize.
    Returns processed DataFrame, encoders dict, scaler.
    """
    df = load_and_merge(transaction_path, identity_path)
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)
    df, scaler = normalize_numericals(df)
    w0, w1 = get_class_weights(df)
    df.attrs['class_weights'] = (w0, w1)
    print(f"\n✅ Preprocessing complete. Shape: {df.shape}")
    return df, encoders, scaler
