"""Analysis on the cleaned data: exploratory plots and one demand-forecasting model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside the container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "temp_c",
    "precip_mm",
    "wind_kmh",
    "humidity_pct",
    "is_raining",
]
TARGET = "trips"
TEST_FRACTION = 0.2


def make_plots(df: pd.DataFrame, out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    profile = df.groupby(["is_weekend", "hour_of_day"])["trips"].mean().unstack(0)
    profile.columns = ["Weekday" if c == 0 else "Weekend" for c in profile.columns]
    profile.plot(ax=axes[0], marker="o", ms=3)
    axes[0].set(title="Average trips by hour of day", xlabel="Hour", ylabel="Trips per hour")

    daytime = df[df["hour_of_day"].between(7, 21)]
    rain = daytime["is_raining"] == 1
    axes[1].scatter(daytime.loc[~rain, "temp_c"], daytime.loc[~rain, "trips"], s=6, alpha=0.4, label="Dry")
    axes[1].scatter(daytime.loc[rain, "temp_c"], daytime.loc[rain, "trips"], s=6, alpha=0.6, label="Raining")
    axes[1].set(title="Trips vs temperature (7am–9pm)", xlabel="Temperature (°C)", ylabel="Trips per hour")
    axes[1].legend()

    daily = df.set_index("hour").resample("D").agg({"trips": "sum", "precip_mm": "sum"})
    axes[2].plot(daily.index, daily["trips"], color="tab:blue")
    axes[2].set(title="Daily trips and rainfall", ylabel="Trips per day")
    ax2 = axes[2].twinx()
    ax2.bar(daily.index, daily["precip_mm"], color="tab:gray", alpha=0.4, width=0.9)
    ax2.set_ylabel("Rain (mm/day)")
    fig.autofmt_xdate()

    fig.tight_layout()
    path = out_dir / "eda.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    df = df.sort_values("hour").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """Random forest vs a seasonal-average baseline, on the last 20% of hours (time-based split)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = chronological_split(df)

    # Baseline: average trips for the same (day of week, hour) in the training period.
    seasonal = train.groupby(["day_of_week", "hour_of_day"])[TARGET].mean()
    baseline_pred = (
        test.join(seasonal.rename("pred"), on=["day_of_week", "hour_of_day"])["pred"]
        .fillna(train[TARGET].mean())
    )

    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])

    rf = _metrics(test[TARGET], pred)
    base = _metrics(test[TARGET], baseline_pred)
    result = {
        "model": "RandomForestRegressor",
        "target": "trips per hour",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_period": f"{train['hour'].min():%Y-%m-%d} to {train['hour'].max():%Y-%m-%d}",
        "test_period": f"{test['hour'].min():%Y-%m-%d} to {test['hour'].max():%Y-%m-%d}",
        **{f"rf_{k}": v for k, v in rf.items()},
        **{f"baseline_{k}": v for k, v in base.items()},
    }
    importances = dict(
        sorted(zip(FEATURES, map(float, model.feature_importances_)), key=lambda kv: -kv[1])
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({**result, "feature_importances": importances}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    window = test.iloc[: 24 * 14]  # first two weeks of the test period
    ax.plot(window["hour"], window[TARGET], label="Actual", lw=1.2)
    ax.plot(window["hour"], pred[: len(window)], label="Random forest", lw=1.2)
    ax.plot(window["hour"], baseline_pred.iloc[: len(window)], label="Baseline", lw=1, ls="--")
    ax.set(title=f"Hourly trips, test period (RF R² = {rf['r2']:.2f})", ylabel="Trips per hour")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "predictions.png", dpi=120)
    plt.close(fig)
    return result
