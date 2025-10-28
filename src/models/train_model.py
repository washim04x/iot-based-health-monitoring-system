import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier


def load_data(train_path):
    return pd.read_csv(train_path)


def get_estimator(model_name: str):
    model_name = (model_name or "").lower()
    if model_name in ("logistic_regression", "logreg", "lr"):
        return LogisticRegression(random_state=0)
    if model_name in ("random_forest", "rf", "randomforest"):
        return RandomForestClassifier(random_state=0)
    if model_name in ("svm", "svc"):
        # probability=True enables predict_proba for ROC AUC
        return SVC(probability=True, random_state=0)
    if model_name in ("linear_svc", "linear-svc"):
        # No predict_proba, but decision_function works for ROC AUC
        return LinearSVC(random_state=0)
    if model_name in ("knn", "kneighbors", "kneighborsclassifier"):
        return KNeighborsClassifier()
    if model_name in ("naive_bayes", "gaussian_nb", "gnb"):
        return GaussianNB()
    if model_name in ("decision_tree", "dt", "decisiontree"):
        return DecisionTreeClassifier(random_state=0)
    if model_name in ("gradient_boosting", "gb", "gbc"):
        return GradientBoostingClassifier(random_state=0)
    if model_name in ("adaboost", "ada"):
        return AdaBoostClassifier(random_state=0)
    if model_name in ("extra_trees", "extratrees", "etc"):
        return ExtraTreesClassifier(random_state=0)
    if model_name in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier  # type: ignore

            # Use a safe baseline; grid will override
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=0,
            )
        except Exception as e:
            raise ValueError(
                "xgboost is not installed. Install it or choose a different model."
            ) from e
    raise ValueError(f"Unsupported model '{model_name}'. Use one of: logistic_regression, random_forest, svm")


def to_param_grid(raw_grid: dict) -> dict:
    """Ensure the grid is a plain dict usable by GridSearchCV (raw_grid already should be)."""
    return {} if raw_grid is None else dict(raw_grid)


def train_and_tune_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    param_grid_all: dict,
    test_size: float,
    random_state: int,
    scoring: str,
    cv_cfg: dict,
    models_dir: Path,
):
    assert target_col in df.columns, f"Target column '{target_col}' not found in data."

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    estimator = get_estimator(model_name)

    # Choose CV strategy
    n_splits = int(cv_cfg.get("n_splits", 5))
    n_repeats = int(cv_cfg.get("n_repeats", 1))
    shuffle = bool(cv_cfg.get("shuffle", True))
    cv_random_state = cv_cfg.get("random_state", random_state)

    if n_repeats and n_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=cv_random_state
        )
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=cv_random_state)

    # Build param grid for selected model
    param_grid = to_param_grid((param_grid_all or {}).get(model_name, {}))
    if not param_grid:
        warnings.warn(
            f"Empty param grid for model '{model_name}'. GridSearchCV will fit with default hyperparameters."
        )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    # Evaluate on hold-out test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    roc = None
    try:
        # Prefer predict_proba, else decision_function
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)
        roc = roc_auc_score(y_test, y_score)
    except Exception:
        pass

    # Save best model
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(models_dir) / "model.joblib"
    joblib.dump(best_model, model_path)

    results = {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_accuracy": acc,
        "test_roc_auc": roc,
        "model_path": str(model_path),
        "model_name": model_name,
        "scoring": scoring,
    }
    return results


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_path = home_dir.as_posix() + "/params.yaml"

    params = yaml.safe_load(open(params_path, "r", encoding="utf-8"))['train_model']

    target_col = params["target_column"]
    test_size = float(params["test_size"])
    random_state = int(params["random_state"])
    model_name = params["model"]
    scoring = params["scoring"]
    cv_cfg = params["cv"]
    param_grid_all = params["param_grid"]

    processed_data_path = home_dir.as_posix() + "/data/processed/train_transformed.csv"
    data = load_data(processed_data_path)

    models_dir = home_dir.as_posix() + "/models"

    results = train_and_tune_model(
        df=data,
        target_col=target_col,
        model_name=model_name,
        param_grid_all=param_grid_all,
        test_size=test_size,
        random_state=random_state,
        scoring=scoring,
        cv_cfg=cv_cfg,
        models_dir=models_dir,
    )

    # Human-friendly output
    print("=== Training & Tuning Summary ===")
    print(f"Model: {results['model_name']}")
    print(f"Scoring: {results['scoring']}")
    print(f"Best CV score: {results['best_cv_score']:.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    if results["test_roc_auc"] is not None:
        print(f"Test ROC AUC: {results['test_roc_auc']:.4f}")
    print(f"Saved best model to: {results['model_path']}")





    
    


