"""Generate fictional municipal-survey responses for reproducible analysis."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "municipal_survey_synthetic.csv"
SEED = 42
N_RESPONDENTS = 750


def clipped_rating(values: np.ndarray) -> np.ndarray:
    """Convert a continuous latent score into a 1-to-5 survey rating."""
    return np.clip(np.rint(values), 1, 5).astype(int)


def generate_survey(n: int = N_RESPONDENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    infrastructure = rng.normal(0, 1, n)
    safety_environment = 0.55 * infrastructure + rng.normal(0, 0.85, n)
    service_management = 0.45 * infrastructure + rng.normal(0, 0.9, n)

    age_group = rng.choice(
        ["18-29", "30-44", "45-59", "60+"],
        size=n,
        p=[0.29, 0.34, 0.24, 0.13],
    )
    zone = rng.choice(["North", "Central", "South", "East", "West"], size=n)

    water_reliability = clipped_rating(3.1 + 0.75 * service_management + rng.normal(0, 0.75, n))
    drainage_maintenance = clipped_rating(2.9 + 0.80 * infrastructure + rng.normal(0, 0.75, n))
    street_condition = clipped_rating(2.8 + 0.85 * infrastructure + rng.normal(0, 0.70, n))
    public_lighting = clipped_rating(3.0 + 0.65 * infrastructure + rng.normal(0, 0.80, n))
    waste_collection = clipped_rating(3.2 + 0.65 * service_management + rng.normal(0, 0.75, n))
    daytime_safety = clipped_rating(3.3 + 0.60 * safety_environment + rng.normal(0, 0.75, n))
    nighttime_safety = clipped_rating(2.7 + 0.75 * safety_environment + rng.normal(0, 0.80, n))
    police_trust = clipped_rating(2.8 + 0.55 * safety_environment + rng.normal(0, 0.90, n))
    green_area_maintenance = clipped_rating(3.0 + 0.50 * infrastructure + rng.normal(0, 0.90, n))

    satisfaction_signal = (
        0.23 * water_reliability
        + 0.17 * street_condition
        + 0.15 * waste_collection
        + 0.13 * drainage_maintenance
        + 0.10 * public_lighting
        + 0.08 * daytime_safety
        + rng.normal(0, 0.55, n)
    )
    overall_satisfaction = clipped_rating(satisfaction_signal)

    df = pd.DataFrame(
        {
            "respondent_id": [f"SYN-{i:04d}" for i in range(1, n + 1)],
            "age_group": age_group,
            "zone": zone,
            "water_reliability": water_reliability,
            "drainage_maintenance": drainage_maintenance,
            "street_condition": street_condition,
            "public_lighting": public_lighting,
            "waste_collection": waste_collection,
            "daytime_safety": daytime_safety,
            "nighttime_safety": nighttime_safety,
            "police_trust": police_trust,
            "green_area_maintenance": green_area_maintenance,
            "overall_satisfaction": overall_satisfaction,
        }
    )

    # Add a small, realistic amount of missingness for the cleaning workflow.
    for column in ["police_trust", "green_area_maintenance", "waste_collection"]:
        missing_rows = rng.choice(df.index, size=max(1, n // 50), replace=False)
        df.loc[missing_rows, column] = np.nan

    return df


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    survey = generate_survey()
    survey.to_csv(OUTPUT, index=False)
    print(f"Created {len(survey):,} synthetic responses at {OUTPUT}")


if __name__ == "__main__":
    main()
