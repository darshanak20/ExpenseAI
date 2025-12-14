import pandas as pd
import numpy as np

def basic_cleaning(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'].astype(str).str.replace('[^0-9.-]','', regex=True), errors='coerce')
    else:
        raise ValueError('CSV must have an amount column')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT
    df = df.dropna(subset=['amount']).reset_index(drop=True)
    return df

def create_features(df):
    df = basic_cleaning(df)
    df['year'] = df['date'].dt.year.fillna(0).astype(int)
    df['month'] = df['date'].dt.month.fillna(0).astype(int)
    df['day'] = df['date'].dt.day.fillna(0).astype(int)
    df['weekday'] = df['date'].dt.weekday.fillna(0).astype(int)
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
    df['description'] = df.get('description', '').astype(str)
    df['merchant'] = df.get('merchant', '').astype(str)
    df['desc_len'] = df['description'].apply(len)
    df['merchant_len'] = df['merchant'].apply(len)
    df['amount_log'] = np.log1p(df['amount'].clip(lower=0))
    return df

def get_feature_matrix(df):
    feature_cols = ['year','month','day','weekday','is_weekend','desc_len','merchant_len','amount_log']
    X = df[feature_cols].copy()
    y_reg = df['amount'].copy()
    y_cls = df['category'] if 'category' in df.columns else None
    return X, y_reg, y_cls, df
