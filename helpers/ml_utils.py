import importlib.util
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, explained_variance_score,
                             f1_score, get_scorer_names, jaccard_score, log_loss, matthews_corrcoef, max_error,
                             mean_absolute_error, mean_absolute_percentage_error, mean_gamma_deviance,
                             mean_pinball_loss, mean_poisson_deviance, mean_squared_error,
                             mean_squared_log_error, median_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from helpers.config import MODELS_DIR, PIPELINES_DIR

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
if HAS_XGBOOST:
    from xgboost import XGBClassifier, XGBRegressor


def build_feature_pipeline(feature_df: pd.DataFrame, numeric_scaler: str, cat_fill: str, num_fill: str) -> ColumnTransformer:
    num_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in feature_df.columns if c not in num_cols]
    scaler = StandardScaler() if numeric_scaler == "standard" else RobustScaler()

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy=num_fill)), ("scaler", scaler)])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy=cat_fill)), ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def get_feature_names(pipeline: ColumnTransformer) -> list[str]:
    names = pipeline.get_feature_names_out().tolist()
    return [n.replace("num__", "").replace("cat__", "") for n in names]


def feature_select(x_train: np.ndarray, y_train: pd.Series, task_type: str, method: str, k: int) -> tuple[np.ndarray, Any]:
    if method == "manual":
        return x_train, None

    if method == "kbest":
        score = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score, k=k)
    elif method == "mutual_info":
        score = mutual_info_classif if task_type == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=score, k=k)
    else:
        base = LogisticRegression(max_iter=2000) if task_type == "classification" else ElasticNet(alpha=0.1)
        selector = RFE(base, n_features_to_select=k, step=0.1)

    return selector.fit_transform(x_train, y_train), selector


def model_candidates(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=3000),
            "svm_svc": SVC(probability=True),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "extra_trees": ExtraTreesClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "adaboost": AdaBoostClassifier(),
            "knn": KNeighborsClassifier(),
            "neural_network_mlp": MLPClassifier(max_iter=1000),
            "gaussian_nb": GaussianNB(),
        }
        if HAS_XGBOOST:
            models["xgboost"] = XGBClassifier(eval_metric="logloss")
        return models

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "elastic_net": ElasticNet(),
        "svm_svr": SVR(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor(),
        "extra_trees": ExtraTreesRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "adaboost": AdaBoostRegressor(),
        "knn": KNeighborsRegressor(),
        "neural_network_mlp": MLPRegressor(max_iter=1000),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBRegressor()
    return models


def available_scorers(task_type: str) -> list[str]:
    scorer_names = sorted(get_scorer_names())
    if task_type == "classification":
        regression_tokens = ["neg_", "d2_", "explained_variance", "r2", "max_error"]
        return [s for s in scorer_names if not any(tok in s for tok in regression_tokens)]
    class_tokens = ["accuracy", "precision", "recall", "f1", "roc_auc", "jaccard", "balanced_accuracy", "top_k", "matthews"]
    return [s for s in scorer_names if not any(tok in s for tok in class_tokens)]


def cv_object(task_type: str, folds: int) -> Any:
    if task_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def tuning_trials(level: str) -> int:
    return {"basic": 10, "light": 25, "medium": 50, "heavy": 100, "extreme": 200}[level]


def objective_factory(model_name: str, base_model: Any, x_train: np.ndarray, y_train: pd.Series, cv: Any, metric: str):
    def objective(trial):
        model = clone(base_model)
        params = {}

        if model_name in {"logistic_regression", "ridge", "lasso", "elastic_net"} and hasattr(model, "alpha"):
            params["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        if model_name == "logistic_regression":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
        if model_name in {"svm_svc", "svm_svr"}:
            params["C"] = trial.suggest_float("C", 0.01, 20.0, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        if model_name in {"decision_tree", "random_forest", "extra_trees"}:
            params["max_depth"] = trial.suggest_int("max_depth", 2, 30)
            if hasattr(model, "min_samples_split"):
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        if model_name in {"random_forest", "extra_trees"}:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 800)
        if model_name in {"gradient_boosting", "adaboost"}:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            if hasattr(model, "learning_rate"):
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
        if model_name == "knn":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 50)
            if hasattr(model, "weights"):
                params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        if model_name == "neural_network_mlp":
            params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            params["hidden_layer_sizes"] = trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64)])
        if model_name == "xgboost":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 12)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        model.set_params(**params)
        return float(cross_val_score(model, x_train, y_train, cv=cv, scoring=metric).mean())

    return objective


def _safe_metric(name: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, float | str]:
    try:
        return {name: float(fn(*args, **kwargs))}
    except Exception as exc:
        return {name: f"N/A ({exc})"}


def evaluate_metrics(model: Any, task_type: str, x_eval: np.ndarray, y_eval: pd.Series) -> dict[str, float | str]:
    y_pred = model.predict(x_eval)
    y_true = np.array(y_eval)

    if task_type == "classification":
        metrics = {}
        metrics.update(_safe_metric("accuracy", accuracy_score, y_true, y_pred))
        metrics.update(_safe_metric("balanced_accuracy", balanced_accuracy_score, y_true, y_pred))
        metrics.update(_safe_metric("precision_macro", precision_score, y_true, y_pred, average="macro", zero_division=0))
        metrics.update(_safe_metric("precision_weighted", precision_score, y_true, y_pred, average="weighted", zero_division=0))
        metrics.update(_safe_metric("recall_macro", recall_score, y_true, y_pred, average="macro", zero_division=0))
        metrics.update(_safe_metric("recall_weighted", recall_score, y_true, y_pred, average="weighted", zero_division=0))
        metrics.update(_safe_metric("f1_macro", f1_score, y_true, y_pred, average="macro", zero_division=0))
        metrics.update(_safe_metric("f1_weighted", f1_score, y_true, y_pred, average="weighted", zero_division=0))
        metrics.update(_safe_metric("jaccard_macro", jaccard_score, y_true, y_pred, average="macro", zero_division=0))
        metrics.update(_safe_metric("matthews_corrcoef", matthews_corrcoef, y_true, y_pred))
        metrics.update(_safe_metric("cohen_kappa", cohen_kappa_score, y_true, y_pred))

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_eval)
            labels = np.unique(y_true)
            metrics.update(_safe_metric("log_loss", log_loss, y_true, y_prob, labels=labels))
            if len(labels) == 2:
                metrics.update(_safe_metric("roc_auc", roc_auc_score, y_true, y_prob[:, 1]))
            else:
                metrics.update(_safe_metric("roc_auc_ovr_weighted", roc_auc_score, y_true, y_prob, multi_class="ovr", average="weighted"))
            if y_prob.shape[1] > 1:
                metrics.update(_safe_metric("top_k_accuracy", top_k_accuracy_score, y_true, y_prob, k=min(2, y_prob.shape[1] - 1), labels=labels))
        return metrics

    y_pred = np.array(y_pred)
    metrics = {}
    metrics.update(_safe_metric("r2", r2_score, y_true, y_pred))
    metrics.update(_safe_metric("rmse", lambda a, b: np.sqrt(mean_squared_error(a, b)), y_true, y_pred))
    metrics.update(_safe_metric("mse", mean_squared_error, y_true, y_pred))
    metrics.update(_safe_metric("mae", mean_absolute_error, y_true, y_pred))
    metrics.update(_safe_metric("median_absolute_error", median_absolute_error, y_true, y_pred))
    metrics.update(_safe_metric("mape", mean_absolute_percentage_error, y_true, y_pred))
    metrics.update(_safe_metric("explained_variance", explained_variance_score, y_true, y_pred))
    metrics.update(_safe_metric("max_error", max_error, y_true, y_pred))
    metrics.update(_safe_metric("mean_pinball_loss", mean_pinball_loss, y_true, y_pred, alpha=0.5))

    if (y_true > 0).all() and (y_pred > 0).all():
        metrics.update(_safe_metric("msle", mean_squared_log_error, y_true, y_pred))
        metrics.update(_safe_metric("mean_poisson_deviance", mean_poisson_deviance, y_true, y_pred))
        metrics.update(_safe_metric("mean_gamma_deviance", mean_gamma_deviance, y_true, y_pred))
    return metrics


def bootstrap_metric(model: Any, task_type: str, x: np.ndarray, y: pd.Series, metric_name: str, repeats: int = 30) -> tuple[float, float]:
    vals = []
    for _ in range(repeats):
        idx = np.random.randint(0, len(y), len(y))
        metric = evaluate_metrics(model, task_type, x[idx], y.iloc[idx]).get(metric_name)
        if isinstance(metric, (float, int, np.floating)):
            vals.append(float(metric))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def apply_smote_if_needed(x_train: np.ndarray, y_train: pd.Series, use_smote: bool) -> tuple[np.ndarray, pd.Series]:
    if not use_smote:
        return x_train, y_train
    sampler = SMOTE(random_state=42)
    return sampler.fit_resample(x_train, y_train)


def save_pipeline(pipeline: Any, name: str) -> str:
    path = PIPELINES_DIR / name
    joblib.dump(pipeline, path)
    return str(path)


def load_pipeline(name: str) -> Any:
    return joblib.load(PIPELINES_DIR / name)


def list_saved_pipelines() -> list[str]:
    return sorted([p.name for p in PIPELINES_DIR.glob("*.joblib")])


def save_model_bundle(model: Any, pipeline: Any, selector: Any, task_type: str, feature_names: list[str], name: str) -> str:
    path = MODELS_DIR / name
    payload = {
        "model": model,
        "pipeline": pipeline,
        "selector": selector,
        "task_type": task_type,
        "feature_names": feature_names,
    }
    joblib.dump(payload, path)
    return str(path)


def load_model_bundle(name: str) -> dict[str, Any]:
    return joblib.load(MODELS_DIR / name)


def list_saved_models() -> list[str]:
    return sorted([p.name for p in MODELS_DIR.glob("*.joblib")])


def summary_metric_name(task_type: str) -> str:
    return "f1_weighted" if task_type == "classification" else "rmse"


def model_doc(task_type: str) -> dict:
    task = task_type.lower()
    if task == "classification":
        return {
            "Dummy (most frequent)": "- Baseline: predicts the most frequent class.",
            "Logistic regression": "- Linear classifier; outputs probabilities; strong baseline.",
            "Random forest": "- Bagging of trees; robust non-linear model.",
            "Extra trees": "- Like RF with extra randomness; often strong + fast.",
            "Gradient boosting": "- Sequential trees to correct errors; can overfit.",
            "Histogram gradient boosting": "- Efficient boosting with binning; good on larger data.",
            "SVM (RBF)": "- Non-linear margin classifier (RBF). Scaling matters.",
            "SVM (Linear)": "- Linear margin classifier; good for high-dimensional data.",
            "SVM (Poly)": "- Polynomial kernel; can be slower.",
            "K-nearest neighbors": "- Local voting; sensitive to scaling/noise.",
            "Neural net (MLP)": "- Feed-forward network; flexible but tune carefully.",
        }

    return {
        "Dummy (mean)": "- Baseline: always predicts the mean of the target.",
        "Linear regression": "- Linear baseline; minimizes squared error.",
        "Ridge regression": "- L2 shrinkage; helps multicollinearity.",
        "Lasso regression": "- L1 shrinkage; can zero out coefficients (feature selection).",
        "Elastic net": "- L1+L2; stable with correlated features.",
        "Random forest": "- Averaged trees; strong non-linear baseline.",
        "Extra trees": "- Randomized trees; often strong and fast.",
        "Gradient boosting": "- Sequential trees reduce residuals; tune carefully.",
        "Histogram gradient boosting": "- Efficient boosting; good on larger data.",
        "SVR (RBF)": "- Non-linear regression; scaling matters.",
        "SVR (Linear)": "- Linear SVR; simpler + faster.",
        "SVR (Poly)": "- Polynomial SVR; can model curvature.",
        "K-nearest neighbors": "- Local averaging; sensitive to scaling/noise.",
        "Neural net (MLP)": "- Feed-forward regressor; tune carefully.",
    }


def param_grids(task_type: str, model_name: str):
    if task_type.lower() == "classification":
        grids = {
            "Logistic regression": {
                "Light": {"model__C": [0.1, 1, 10]},
                "Medium": {"model__C": [0.01, 0.1, 1, 10], "model__solver": ["lbfgs", "liblinear"]},
                "Heavy": {"model__C": [0.001, 0.01, 0.1, 1, 10, 100], "model__solver": ["lbfgs", "liblinear"]},
            },
            "Random forest": {
                "Light": {"model__n_estimators": [200, 400]},
                "Medium": {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20]},
                "Heavy": {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5],
                    "model__min_samples_leaf": [1, 2],
                },
            },
            "Extra trees": {
                "Light": {"model__n_estimators": [300, 500]},
                "Medium": {"model__n_estimators": [300, 500], "model__max_depth": [None, 10, 20]},
                "Heavy": {
                    "model__n_estimators": [300, 500],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5],
                    "model__min_samples_leaf": [1, 2],
                },
            },
        }
        return grids.get(model_name, {"Light": {}, "Medium": {}, "Heavy": {}})

    grids = {
        "Ridge regression": {
            "Light": {"model__alpha": [0.1, 1, 10]},
            "Medium": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
            "Heavy": {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
        },
        "Lasso regression": {
            "Light": {"model__alpha": [0.001, 0.01, 0.1]},
            "Medium": {"model__alpha": [0.0005, 0.001, 0.01, 0.1]},
            "Heavy": {"model__alpha": [0.0001, 0.0005, 0.001, 0.01, 0.1]},
        },
    }
    return grids.get(model_name, {"Light": {}, "Medium": {}, "Heavy": {}})
