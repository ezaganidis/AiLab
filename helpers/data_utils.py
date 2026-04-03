from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class TrainContext:
    task_type: str
    target: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_cols: list[str]


def read_uploaded_data(uploaded_file: Any, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(uploaded_file)
    if file_type == "json":
        return pd.read_json(uploaded_file)
    if file_type == "xlsx":
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type")


def read_sql_data(connection_uri: str, query: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(connection_uri)
    return pd.read_sql(query, engine)


def split_context(df: pd.DataFrame, target: str, task_type: str, test_size: float, random_state: int) -> TrainContext:
    feature_cols = [c for c in df.columns if c != target]
    x = df[feature_cols]
    y = df[target]
    stratify = y if task_type == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_df = x_train.copy()
    train_df[target] = y_train
    test_df = x_test.copy()
    test_df[target] = y_test
    return TrainContext(task_type, target, train_df, test_df, y_train, y_test, feature_cols)
