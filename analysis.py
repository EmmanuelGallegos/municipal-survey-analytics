"""Run a reproducible analysis of the synthetic municipal survey."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "municipal_survey_synthetic.csv"
REPORTS = ROOT
TARGET = "overall_satisfaction"
FEATURES = [
    "water_reliability",
    "drainage_maintenance",
    "street_condition",
    "public_lighting",
    "waste_collection",
    "daytime_safety",
    "nighttime_safety",
    "police_trust",
    "green_area_maintenance",
]


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run src/generate_synthetic_data.py first."
        )
    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET, "respondent_id", "age_group", "zone"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    correlations = df[FEATURES + [TARGET]].corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1)
    plt.title("Municipal survey correlation matrix")
    plt.tight_layout()
    plt.savefig(REPORTS / "correlation_heatmap.png", dpi=180)
    plt.close()


def run_pca(df: pd.DataFrame) -> dict[str, float]:
    prepared = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    ).fit_transform(df[FEATURES])
    pca = PCA().fit(prepared)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(FEATURES) + 1), pca.explained_variance_ratio_.cumsum(), marker="o")
    plt.axhline(0.80, color="gray", linestyle="--", label="80% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "pca_explained_variance.png", dpi=180)
    plt.close()

    return {
        f"pc_{i + 1}_explained_variance": round(float(value), 4)
        for i, value in enumerate(pca.explained_variance_ratio_[:3])
    }


def fit_satisfaction_model(df: pd.DataFrame) -> tuple[dict[str, float], pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.25, random_state=42
    )
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    model = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.65)
    plt.plot([1, 5], [1, 5], linestyle="--", color="black")
    plt.xlabel("Actual satisfaction")
    plt.ylabel("Predicted satisfaction")
    plt.title("Predicted vs. actual satisfaction")
    plt.tight_layout()
    plt.savefig(REPORTS / "predicted_vs_actual.png", dpi=180)
    plt.close()

    fitted_pipeline = model.regressor_
    coefficients = pd.Series(
        fitted_pipeline.named_steps["model"].coef_, index=FEATURES
    ).sort_values(key=abs, ascending=False)
    metrics = {
        "test_rows": int(len(y_test)),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
    return metrics, coefficients


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_and_validate(DATA_PATH)
    save_correlation_heatmap(df)
    metrics = {"rows": int(len(df)), "synthetic_data": True}
    metrics.update(run_pca(df))
    model_metrics, coefficients = fit_satisfaction_model(df)
    metrics.update(model_metrics)
    metrics["standardized_coefficients"] = {
        key: round(float(value), 4) for key, value in coefficients.items()
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
