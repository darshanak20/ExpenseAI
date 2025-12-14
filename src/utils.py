import os, joblib

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path

def load_model(path):
    return joblib.load(path)
