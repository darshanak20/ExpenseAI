# run_pipeline.py - lightweight runner to process CSV and train models (no-Jupyter)
from src.data_pipeline import read_csv, write_processed
from src.feature_engineering import create_features, get_feature_matrix
from src.model_training import train_regressor, train_classifier
from src.forecasting import prepare_series, train_prophet
from src.anomaly import detect_anomalies_amounts
import os
import argparse

def run_demo(csv_path='data/raw/sample_expenses.csv'):
    if not os.path.exists(csv_path):
        print('CSV not found; please place your CSV at', csv_path)
        return
    df = read_csv(csv_path)
    df_fe = create_features(df)
    os.makedirs('data/processed', exist_ok=True)
    write_processed(df_fe, 'data/processed/processed.csv')

    X, y_reg, y_cls, df_full = get_feature_matrix(df_fe)
    print('Training regressor...')
    reg_model, rmse = train_regressor(X, y_reg, save_path='models/expense_regressor.joblib')
    print('RMSE:', rmse)

    print('Training classifier (if enough labels)...')
    cls_model, acc = train_classifier(X, y_cls, save_path='models/expense_classifier.joblib')
    if cls_model is not None:
        print('Classifier accuracy:', acc)
    else:
        print('Not enough labeled categories to train classifier.')

    print('Training Prophet forecast...')
    series = prepare_series(df_full)
    if len(series) >= 14:
        m, forecast = train_prophet(series, periods=30, save_path='models/prophet_model.joblib')
        print('Forecast generated (last rows):')
        print(forecast[['ds','yhat']].tail())
    else:
        print('Not enough time series data for Prophet (need ~14+ daily points).')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/raw/sample_expenses.csv')
    args = parser.parse_args()
    run_demo(csv_path=args.csv)
