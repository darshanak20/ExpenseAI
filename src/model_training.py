import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from src.utils import save_model

def train_regressor(X, y, save_path=None, random_state=42):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([('num', StandardScaler(), numeric)])
    pipeline = Pipeline([('pre', pre), ('rf', RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    if save_path:
        save_model(pipeline, save_path)
    return pipeline, rmse

def train_classifier(X, y, save_path=None, min_count=30, random_state=42):
    if y is None:
        return None, None
    counts = y.value_counts()
    valid = counts[counts >= min_count].index.tolist()
    mask = y.isin(valid)
    Xv = X[mask]
    yv = y[mask]
    if yv.nunique() < 2:
        return None, None
    numeric = Xv.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([('num', StandardScaler(), numeric)])
    pipeline = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(Xv, yv, test_size=0.2, random_state=random_state, stratify=yv)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if save_path:
        save_model(pipeline, save_path)
    return pipeline, acc
