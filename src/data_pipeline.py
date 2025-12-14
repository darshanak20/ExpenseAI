import pandas as pd
from sqlalchemy import create_engine

def get_engine(user, password, host, port, database, driver='mysql+mysqlconnector'):
    conn = f"{driver}://{user}:{password}@{host}:{port}/{database}"
    return create_engine(conn)

def read_from_db(engine, table_name, limit=None):
    sql = f"SELECT * FROM {table_name}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql(sql, engine)

def read_csv(path):
    return pd.read_csv(path)

def write_processed(df, path):
    df.to_csv(path, index=False)
