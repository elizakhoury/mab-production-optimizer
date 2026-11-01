"""Batch analysis, grouped model comparison, and mAb production optimization.

Prediction intervals are approximate empirical residual intervals, not
independently calibrated guarantees for the optimized conditions.
"""


from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from itertools import product
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import FunctionTransformer


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_DIR / "datasets"

EXPECTED_BATCH_IDS = [
    1, 2, 10, 19, 64, 73, 82, 127, 136, 145,
    442, 451, 460, 505, 514, 515, 523, 568, 577, 586,
    883, 892, 901, 946, 955, 964, 1009, 1018, 1027, 1028,
]

TABLE_DIR = PROJECT_DIR / "tables"

DOWNTIME_DAYS = 1.0

FEATURE_COLUMNS = [
    "initial_glucose_mM",
    "initial_glutamine_mM",
    "initial_vcd_cells_mL",
    "harvest_time_d",
]

TARGET_COLUMN = "mAb_mg_L"

CONDITION_COLUMNS = [
    "initial_glucose_mM",
    "initial_glutamine_mM",
    "initial_vcd_cells_mL",
]

FIGURE_DIR = PROJECT_DIR / "figures"


def saturation_features(inputs, tau, power):
    """Create condition features multiplied by a zero-start time curve."""
    values = np.asarray(inputs, dtype=float)

    # Fixed scaling based on our experimental design.
    conditions = (
        values[:, :3] - np.array([20.0, 4.0, 800000.0])
    ) / np.array([15.0, 3.0, 600000.0])

    condition_features = PolynomialFeatures(
        degree=2,
        include_bias=True,
    ).fit_transform(conditions)

    time = values[:, 3]
    time_factor = time**power / (tau**power + time**power)

    return condition_features * time_factor[:, None]


def load_and_calculate_economics():
    """Read the selected batches and export observed economic results."""
    initial_conditions = pd.read_csv(
        DATASET_DIR / "initial_conditions.csv"
    )

    batch_files = sorted(DATASET_DIR.glob("batch_*.csv"))

    expected_names = {
        f"batch_{batch_id:04d}.csv"
        for batch_id in EXPECTED_BATCH_IDS
    }

    actual_names = {filepath.name for filepath in batch_files}

    if actual_names != expected_names:
        raise ValueError(
            f"Missing files: {sorted(expected_names - actual_names)}\n"
            f"Unexpected files: {sorted(actual_names - expected_names)}"
        )

    print("DATASET CHECK")

    print(f"Batch files found: {len(batch_files)}")

    selected_conditions = initial_conditions[
        initial_conditions["Batch ID"].isin(EXPECTED_BATCH_IDS)
    ].copy()

    if (
        len(selected_conditions) != 30
        or selected_conditions["Batch ID"].duplicated().any()
    ):
        raise ValueError("Expected one initial-conditions row per selected batch.")

    print(f"Selected initial-condition rows: {len(selected_conditions)}")

    condition_columns = [
        "Glucose [mM]",
        "Glutamine [mM]",
        "VCD [cells/mL]",
    ]

    print(
        "Unique starting conditions:",
        len(selected_conditions[condition_columns].drop_duplicates()),
    )

    required_columns = [
        "Time [h]",
        "Glucose [mM]",
        "Glutamine [mM]",
        "Lactate [mM]",
        "Ammonia [mM]",
        "VCD [cells/mL]",
        "TCD [cells/mL]",
        "mAb [mg/L]",
    ]

    print("\nBATCH FILE CHECKS")

    for filepath in batch_files:
        batch = pd.read_csv(filepath)

        missing_columns = set(required_columns) - set(batch.columns)
        if missing_columns:
            raise ValueError(
                f"{filepath.name}: missing columns {sorted(missing_columns)}"
            )

        if batch.empty:
            raise ValueError(f"{filepath.name}: file contains no observations.")

        missing_values = batch[required_columns].isna().sum().sum()

        print(
            f"{filepath.name}: "
            f"{len(batch)} rows, "
            f"{missing_values} missing values, "
            f"time {batch['Time [h]'].min():g}"
            f"–{batch['Time [h]'].max():g} h"
        )

    print("\nFIRST BATCH: FIRST FIVE ROWS")

    first_batch = pd.read_csv(batch_files[0])

    print(first_batch.head().to_string(index=False))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    conditions_by_id = selected_conditions.set_index("Batch ID")

    batch_results = []

    for filepath in batch_files:
        batch_id = int(filepath.stem.split("_")[1])
        batch = pd.read_csv(filepath).sort_values("Time [h]")
        initial = conditions_by_id.loc[batch_id]

        initial_glucose = initial["Glucose [mM]"]
        initial_glutamine = initial["Glutamine [mM]"]
        initial_vcd = initial["VCD [cells/mL]"]

        culture_cost = (
            10 * initial_glucose
            + 20 * initial_glutamine
            + 0.0005 * initial_vcd
        )

        results = pd.DataFrame({
            "batch_id": batch_id,
            "initial_glucose_mM": initial_glucose,
            "initial_glutamine_mM": initial_glutamine,
            "initial_vcd_cells_mL": initial_vcd,
            "harvest_time_d": batch["Time [h]"] / 24,
            "mAb_mg_L": batch["mAb [mg/L]"],
        })

        results["culture_cost_USD_L"] = culture_cost
        results["culture_value_USD_L"] = 1000 * results["mAb_mg_L"]
        results["profit_USD_L"] = (
            results["culture_value_USD_L"]
            - results["culture_cost_USD_L"]
        )

        cycle_time = results["harvest_time_d"] + DOWNTIME_DAYS

        results["space_time_profit_USD_L_d"] = (
            results["profit_USD_L"] / cycle_time
        )
        results["space_time_yield_mg_L_d"] = (
            results["mAb_mg_L"] / cycle_time
        )

        batch_results.append(results)

    economic_results = pd.concat(batch_results, ignore_index=True)

    economic_results.to_csv(
        TABLE_DIR / "economic_time_series.csv",
        index=False,
    )

    best_indices = economic_results.groupby("batch_id")[
        "space_time_profit_USD_L_d"
    ].idxmax()

    best_observed = economic_results.loc[best_indices].sort_values(
        "space_time_profit_USD_L_d",
        ascending=False,
    )

    best_observed.to_csv(
        TABLE_DIR / "best_observed_per_batch.csv",
        index=False,
    )

    print("\nTOP FIVE OBSERVED BATCH RESULTS")

    print(best_observed.head().to_string(index=False))

    print("\nSaved tables/economic_time_series.csv")

    print("Saved tables/best_observed_per_batch.csv")
    return economic_results, initial_conditions



def evaluate_reference_models(economic_results):
    """Evaluate baseline, linear, and polynomial models on grouped folds."""
    X = economic_results[FEATURE_COLUMNS].copy()

    y = economic_results[TARGET_COLUMN].copy()

    groups = economic_results.groupby(
        CONDITION_COLUMNS,
        sort=True,
    ).ngroup()

    print("\nMACHINE LEARNING DATA")

    print(f"Measurement rows: {len(X)}")

    print(f"Input variables: {X.shape[1]}")

    print(f"Independent starting-condition groups: {groups.nunique()}")

    print("\nINPUT COLUMNS")

    for column in FEATURE_COLUMNS:
        print(column)

    print(f"\nPrediction target: {TARGET_COLUMN}")

    group_cv = GroupKFold(n_splits=5)

    cv_splits = list(group_cv.split(X, y, groups=groups))

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    baseline_model = DummyRegressor(strategy="mean")

    baseline_scores = cross_validate(
        baseline_model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        error_score="raise",
    )

    print("\nBASELINE: FIVE-FOLD GROUPED CROSS-VALIDATION")

    print(
        "Mean validation RMSE: "
        f"{-baseline_scores['test_rmse'].mean():.4f} mg/L"
    )

    print(
        "Mean validation MAE: "
        f"{-baseline_scores['test_mae'].mean():.4f} mg/L"
    )

    print(
        "Mean validation R²: "
        f"{baseline_scores['test_r2'].mean():.4f}"
    )

    baseline_fold_results = pd.DataFrame({
        "fold": range(1, 6),
        "validation_rmse_mg_L": -baseline_scores["test_rmse"],
        "validation_mae_mg_L": -baseline_scores["test_mae"],
        "validation_r2": baseline_scores["test_r2"],
    })

    baseline_fold_results.to_csv(
        TABLE_DIR / "baseline_validation.csv",
        index=False,
    )

    print("\nSaved tables/baseline_validation.csv")

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression()),
    ])

    linear_scores = cross_validate(
        linear_model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        return_train_score=True,
        error_score="raise",
    )

    print("\nLINEAR REGRESSION: GROUPED CROSS-VALIDATION")

    print(
        "Mean training RMSE: "
        f"{-linear_scores['train_rmse'].mean():.4f} mg/L"
    )

    print(
        "Mean validation RMSE: "
        f"{-linear_scores['test_rmse'].mean():.4f} mg/L"
    )

    print(
        "Mean validation MAE: "
        f"{-linear_scores['test_mae'].mean():.4f} mg/L"
    )

    print(
        "Mean validation R²: "
        f"{linear_scores['test_r2'].mean():.4f}"
    )

    linear_fold_results = pd.DataFrame({
        "fold": range(1, 6),
        "training_rmse_mg_L": -linear_scores["train_rmse"],
        "validation_rmse_mg_L": -linear_scores["test_rmse"],
        "validation_mae_mg_L": -linear_scores["test_mae"],
        "validation_r2": linear_scores["test_r2"],
    })

    linear_fold_results.to_csv(
        TABLE_DIR / "linear_validation.csv",
        index=False,
    )

    print("\nSaved tables/linear_validation.csv")

    polynomial_model = Pipeline([
        ("input_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(
            degree=2,
            include_bias=False,
        )),
        ("regression", LinearRegression()),
    ])

    polynomial_scores = cross_validate(
        polynomial_model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        return_train_score=True,
        error_score="raise",
    )

    print("\nDEGREE-2 POLYNOMIAL: GROUPED CROSS-VALIDATION")

    print(
        "Mean training RMSE: "
        f"{-polynomial_scores['train_rmse'].mean():.4f} mg/L"
    )

    print(
        "Mean validation RMSE: "
        f"{-polynomial_scores['test_rmse'].mean():.4f} mg/L"
    )

    print(
        "Mean validation MAE: "
        f"{-polynomial_scores['test_mae'].mean():.4f} mg/L"
    )

    print(
        "Mean validation R²: "
        f"{polynomial_scores['test_r2'].mean():.4f}"
    )

    polynomial_fold_results = pd.DataFrame({
        "fold": range(1, 6),
        "training_rmse_mg_L": -polynomial_scores["train_rmse"],
        "validation_rmse_mg_L": -polynomial_scores["test_rmse"],
        "validation_mae_mg_L": -polynomial_scores["test_mae"],
        "validation_r2": polynomial_scores["test_r2"],
    })

    polynomial_fold_results.to_csv(
        TABLE_DIR / "polynomial_degree2_validation.csv",
        index=False,
    )

    print("\nSaved tables/polynomial_degree2_validation.csv")
    return X, baseline_scores, cv_splits, linear_scores, polynomial_model, polynomial_scores, scoring, y



def diagnose_polynomial_model(X, baseline_scores, cv_splits, economic_results, linear_scores, polynomial_model, polynomial_scores, y):
    """Export held-out predictions and harvest diagnostics for the polynomial."""
    polynomial_oof_predictions = cross_val_predict(
        polynomial_model,
        X,
        y,
        cv=cv_splits,
    )

    validation_predictions = economic_results.copy()

    validation_predictions["predicted_mAb_mg_L"] = (
        polynomial_oof_predictions
    )

    validation_predictions["residual_mg_L"] = (
        validation_predictions["mAb_mg_L"]
        - validation_predictions["predicted_mAb_mg_L"]
    )

    validation_predictions["predicted_space_time_profit_USD_L_d"] = (
        1000 * validation_predictions["predicted_mAb_mg_L"]
        - validation_predictions["culture_cost_USD_L"]
    ) / (
        validation_predictions["harvest_time_d"] + DOWNTIME_DAYS
    )

    validation_predictions.to_csv(
        TABLE_DIR / "polynomial_validation_predictions.csv",
        index=False,
    )

    early_harvest = validation_predictions[
        validation_predictions["harvest_time_d"].between(1, 10)
    ]

    early_rmse = (
        early_harvest["residual_mg_L"].pow(2).mean()
    ) ** 0.5

    early_mae = early_harvest["residual_mg_L"].abs().mean()

    negative_predictions = (
        validation_predictions["predicted_mAb_mg_L"] < 0
    ).sum()

    print("\nHELD-OUT PREDICTION CHECK")

    print(f"Total predictions: {len(validation_predictions)}")

    print(f"Negative concentration predictions: {negative_predictions}")

    print(f"Days 1–10 RMSE: {early_rmse:.4f} mg/L")

    print(f"Days 1–10 MAE: {early_mae:.4f} mg/L")

    print("\nSaved tables/polynomial_validation_predictions.csv")

    harvest_candidates = validation_predictions[
        validation_predictions["harvest_time_d"] > 0
    ].copy()

    recommended_indices = harvest_candidates.groupby("batch_id")[
        "predicted_space_time_profit_USD_L_d"
    ].idxmax()

    harvest_check = harvest_candidates.loc[
        recommended_indices,
        [
            "batch_id",
            "harvest_time_d",
            "predicted_space_time_profit_USD_L_d",
            "space_time_profit_USD_L_d",
        ],
    ].copy()

    harvest_check = harvest_check.rename(columns={
        "harvest_time_d": "recommended_harvest_time_d",
        "space_time_profit_USD_L_d":
            "observed_profit_at_recommended_time_USD_L_d",
    })

    observed_best = harvest_candidates.groupby("batch_id")[
        "space_time_profit_USD_L_d"
    ].max().rename("best_observed_profit_USD_L_d")

    harvest_check = harvest_check.join(
        observed_best,
        on="batch_id",
    )

    harvest_check["observed_profit_gap_USD_L_d"] = (
        harvest_check["best_observed_profit_USD_L_d"]
        - harvest_check["observed_profit_at_recommended_time_USD_L_d"]
    )

    harvest_check = harvest_check.sort_values("batch_id")

    harvest_check.to_csv(
        TABLE_DIR / "harvest_validation.csv",
        index=False,
    )

    print("\nHARVEST-TIME VALIDATION")

    print(
        "Mean observed profit gap: "
        f"{harvest_check['observed_profit_gap_USD_L_d'].mean():.2f}"
        " USD/(L·day)"
    )

    print(
        "Median observed profit gap: "
        f"{harvest_check['observed_profit_gap_USD_L_d'].median():.2f}"
        " USD/(L·day)"
    )

    print(
        "Largest observed profit gap: "
        f"{harvest_check['observed_profit_gap_USD_L_d'].max():.2f}"
        " USD/(L·day)"
    )

    print("\nCENTER-CONDITION REPEAT BATCHES")

    print(
        harvest_check[
            harvest_check["batch_id"].isin([514, 515])
        ].to_string(index=False)
    )

    print("\nSaved tables/harvest_validation.csv")

    model_comparison_rows = []

    for model_name, scores in [
        ("Mean baseline", baseline_scores),
        ("Linear regression", linear_scores),
        ("Polynomial degree 2", polynomial_scores),
    ]:
        model_comparison_rows.append({
            "model": model_name,
            "mean_validation_rmse_mg_L":
                -scores["test_rmse"].mean(),
            "std_validation_rmse_mg_L":
                scores["test_rmse"].std(ddof=1),
            "mean_validation_mae_mg_L":
                -scores["test_mae"].mean(),
            "mean_validation_r2":
                scores["test_r2"].mean(),
        })

    model_comparison = pd.DataFrame(
        model_comparison_rows
    ).sort_values("mean_validation_rmse_mg_L")

    model_comparison.to_csv(
        TABLE_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")

    print(model_comparison.round(4).to_string(index=False))

    print("\nSaved tables/model_comparison.csv")
    return model_comparison



def inspect_polynomial_optimization(X, economic_results, initial_conditions, polynomial_model, y):
    """Keep the polynomial search and early-time checks as comparison evidence."""
    final_model = clone(polynomial_model)

    final_model.fit(X, y)

    glucose_levels = sorted(initial_conditions["Glucose [mM]"].unique())

    glutamine_levels = sorted(initial_conditions["Glutamine [mM]"].unique())

    vcd_levels = sorted(initial_conditions["VCD [cells/mL]"].unique())

    harvest_times = sorted(
        economic_results.loc[
            economic_results["harvest_time_d"] > 0,
            "harvest_time_d",
        ].unique()
    )

    search_results = pd.DataFrame(
        product(
            glucose_levels,
            glutamine_levels,
            vcd_levels,
            harvest_times,
        ),
        columns=FEATURE_COLUMNS,
    )

    search_results["predicted_mAb_mg_L"] = final_model.predict(
        search_results[FEATURE_COLUMNS]
    )

    search_results["culture_cost_USD_L"] = (
        10 * search_results["initial_glucose_mM"]
        + 20 * search_results["initial_glutamine_mM"]
        + 0.0005 * search_results["initial_vcd_cells_mL"]
    )

    search_results["predicted_culture_value_USD_L"] = (
        1000 * search_results["predicted_mAb_mg_L"]
    )

    cycle_time = search_results["harvest_time_d"] + DOWNTIME_DAYS

    search_results["predicted_space_time_profit_USD_L_d"] = (
        search_results["predicted_culture_value_USD_L"]
        - search_results["culture_cost_USD_L"]
    ) / cycle_time

    search_results["predicted_space_time_yield_mg_L_d"] = (
        search_results["predicted_mAb_mg_L"] / cycle_time
    )

    search_results.to_csv(
        TABLE_DIR / "optimization_grid.csv",
        index=False,
    )

    feasible_results = search_results[
        search_results["predicted_mAb_mg_L"] >= 0
    ].copy()

    if feasible_results.empty:
        raise ValueError("No candidates have nonnegative predicted mAb.")

    ranked_candidates = feasible_results.sort_values(
        "predicted_space_time_profit_USD_L_d",
        ascending=False,
    )

    recommendation = ranked_candidates.iloc[0].copy()

    ranked_candidates.head(10).to_csv(
        TABLE_DIR / "top_predicted_candidates.csv",
        index=False,
    )

    print("\nMODEL-BASED GRID SEARCH")

    print(f"Candidate combinations evaluated: {len(search_results)}")

    print("\nRECOMMENDED CONDITIONS AND PREDICTED RESULTS")

    print(recommendation.to_string())

    print("\nSaved tables/optimization_grid.csv")

    print("Saved tables/top_predicted_candidates.csv")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    same_conditions = (
        (search_results["initial_glucose_mM"]
         == recommendation["initial_glucose_mM"])
        & (search_results["initial_glutamine_mM"]
           == recommendation["initial_glutamine_mM"])
        & (search_results["initial_vcd_cells_mL"]
           == recommendation["initial_vcd_cells_mL"])
    )

    recommended_curve = search_results.loc[
        same_conditions
    ].sort_values("harvest_time_d")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        recommended_curve["harvest_time_d"],
        recommended_curve["predicted_space_time_profit_USD_L_d"],
        color="tab:blue",
        label="Predicted: recommended starting conditions",
    )

    ax.scatter(
        recommendation["harvest_time_d"],
        recommendation["predicted_space_time_profit_USD_L_d"],
        color="black",
        marker="*",
        s=180,
        zorder=5,
        label="Best predicted harvest time on grid",
    )

    for batch_id, color in [(514, "tab:orange"), (515, "tab:green")]:
        observed = economic_results[
            economic_results["batch_id"] == batch_id
        ]

        ax.scatter(
            observed["harvest_time_d"],
            observed["space_time_profit_USD_L_d"],
            s=15,
            alpha=0.7,
            color=color,
            label=f"Observed batch {batch_id}: 20 mM, 4 mM, 800,000 cells/mL",
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Harvest time [days]")

    ax.set_ylabel("Space-time profit [USD/(L·day)]")

    ax.set_title("Predicted recommendation and observed center-condition batches")

    ax.grid(alpha=0.25)

    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "recommended_harvest_curve.png",
        dpi=300,
    )

    plt.close(fig)

    print("\nSaved figures/recommended_harvest_curve.png")

    early_check = pd.DataFrame({
        "initial_glucose_mM": [recommendation["initial_glucose_mM"]] * 5,
        "initial_glutamine_mM": [
            recommendation["initial_glutamine_mM"]
        ] * 5,
        "initial_vcd_cells_mL": [
            recommendation["initial_vcd_cells_mL"]
        ] * 5,
        "harvest_time_d": [0, 8 / 24, 1, 2, 3],
    })

    early_check["predicted_mAb_mg_L"] = final_model.predict(
        early_check[FEATURE_COLUMNS]
    )

    print("\nEARLY-TIME MODEL CHECK")

    print(
        early_check[
            ["harvest_time_d", "predicted_mAb_mg_L"]
        ].round(4).to_string(index=False)
    )
    return search_results



def select_saturation_model(X, cv_splits, economic_results, scoring, y):
    """Compare saturation settings and inspect held-out harvest decisions."""
    saturation_rows = []

    saturation_models = {}

    for tau in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for power in [1, 2, 3]:
            candidate = Pipeline([
                ("features", FunctionTransformer(
                    saturation_features,
                    kw_args={"tau": tau, "power": power},
                )),
                ("regression", LinearRegression(fit_intercept=False)),
            ])

            scores = cross_validate(
                candidate,
                X,
                y,
                cv=cv_splits,
                scoring=scoring,
                return_train_score=True,
                error_score="raise",
            )

            saturation_models[(tau, power)] = candidate

            saturation_rows.append({
                "tau_days": tau,
                "power": power,
                "training_rmse_mg_L": -scores["train_rmse"].mean(),
                "validation_rmse_mg_L": -scores["test_rmse"].mean(),
                "validation_mae_mg_L": -scores["test_mae"].mean(),
                "validation_r2": scores["test_r2"].mean(),
            })

    saturation_comparison = pd.DataFrame(
        saturation_rows
    ).sort_values("validation_rmse_mg_L")

    saturation_comparison.to_csv(
        TABLE_DIR / "saturation_model_comparison.csv",
        index=False,
    )

    print("\nSATURATION MODEL COMPARISON")

    print(saturation_comparison.round(4).to_string(index=False))

    best_settings = saturation_comparison.iloc[0]

    best_tau = float(best_settings["tau_days"])

    best_power = int(best_settings["power"])

    selected_saturation_model = clone(
        saturation_models[(best_tau, best_power)]
    )

    saturation_oof = cross_val_predict(
        selected_saturation_model,
        X,
        y,
        cv=cv_splits,
    )

    saturation_validation = economic_results.copy()

    saturation_validation["predicted_mAb_mg_L"] = saturation_oof

    saturation_validation["residual_mg_L"] = (
        y.to_numpy() - saturation_oof
    )

    saturation_validation["predicted_space_time_profit_USD_L_d"] = (
        1000 * saturation_validation["predicted_mAb_mg_L"]
        - saturation_validation["culture_cost_USD_L"]
    ) / (
        saturation_validation["harvest_time_d"] + DOWNTIME_DAYS
    )

    early_rows = saturation_validation[
        saturation_validation["harvest_time_d"].between(1, 10)
    ]

    early_rmse = (
        early_rows["residual_mg_L"].pow(2).mean()
    ) ** 0.5

    candidates = saturation_validation[
        saturation_validation["harvest_time_d"] > 0
    ].copy()

    chosen_indices = candidates.groupby("batch_id")[
        "predicted_space_time_profit_USD_L_d"
    ].idxmax()

    saturation_harvest = candidates.loc[chosen_indices].copy()

    best_observed_profit = candidates.groupby("batch_id")[
        "space_time_profit_USD_L_d"
    ].max()

    saturation_harvest["observed_profit_gap_USD_L_d"] = (
        saturation_harvest["batch_id"].map(best_observed_profit)
        - saturation_harvest["space_time_profit_USD_L_d"]
    )

    saturation_validation.to_csv(
        TABLE_DIR / "saturation_validation_predictions.csv",
        index=False,
    )

    saturation_harvest.to_csv(
        TABLE_DIR / "saturation_harvest_validation.csv",
        index=False,
    )

    print("\nSATURATION MODEL CHECK")

    print(f"Selected tau: {best_tau:g} days")

    print(f"Selected power: {best_power}")

    print(f"Days 1–10 RMSE: {early_rmse:.4f} mg/L")

    print(
        "Negative held-out concentration predictions:",
        int((saturation_oof < 0).sum()),
    )

    print(
        "Mean observed profit gap: "
        f"{saturation_harvest['observed_profit_gap_USD_L_d'].mean():.2f}"
        " USD/(L·day)"
    )

    print(
        "Largest observed profit gap: "
        f"{saturation_harvest['observed_profit_gap_USD_L_d'].max():.2f}"
        " USD/(L·day)"
    )
    return best_power, best_tau, saturation_validation, selected_saturation_model



def optimize_saturation_model(X, search_results, selected_saturation_model, y):
    """Fit the selected model and search the starting-condition and time grid."""
    final_saturation_model = clone(selected_saturation_model)

    final_saturation_model.fit(X, y)

    saturation_grid = search_results[FEATURE_COLUMNS].copy()

    saturation_grid["predicted_mAb_mg_L"] = (
        final_saturation_model.predict(saturation_grid[FEATURE_COLUMNS])
    )

    saturation_grid["culture_cost_USD_L"] = (
        10 * saturation_grid["initial_glucose_mM"]
        + 20 * saturation_grid["initial_glutamine_mM"]
        + 0.0005 * saturation_grid["initial_vcd_cells_mL"]
    )

    saturation_grid["predicted_culture_value_USD_L"] = (
        1000 * saturation_grid["predicted_mAb_mg_L"]
    )

    cycle_days = saturation_grid["harvest_time_d"] + DOWNTIME_DAYS

    saturation_grid["predicted_space_time_profit_USD_L_d"] = (
        saturation_grid["predicted_culture_value_USD_L"]
        - saturation_grid["culture_cost_USD_L"]
    ) / cycle_days

    saturation_grid["predicted_space_time_yield_mg_L_d"] = (
        saturation_grid["predicted_mAb_mg_L"] / cycle_days
    )

    saturation_grid.to_csv(
        TABLE_DIR / "saturation_optimization_grid.csv",
        index=False,
    )

    feasible_saturation = saturation_grid[
        saturation_grid["predicted_mAb_mg_L"] >= 0
    ].sort_values(
        "predicted_space_time_profit_USD_L_d",
        ascending=False,
    )

    if feasible_saturation.empty:
        raise ValueError("No feasible saturation-model candidates.")

    saturation_recommendation = feasible_saturation.iloc[0].copy()

    feasible_saturation.head(10).to_csv(
        TABLE_DIR / "saturation_top_candidates.csv",
        index=False,
    )

    zero_time_input = feasible_saturation[FEATURE_COLUMNS].head(1).copy()

    zero_time_input["harvest_time_d"] = 0.0

    zero_prediction = final_saturation_model.predict(zero_time_input)[0]

    print("\nSATURATION MODEL: PROPOSED CONDITIONS")

    print(saturation_recommendation.to_string())

    print(f"\nPredicted mAb at time zero: {zero_prediction:.6f} mg/L")
    return saturation_grid, saturation_recommendation



def estimate_prediction_intervals(saturation_recommendation, saturation_validation):
    """Estimate approximate residual-based intervals at the proposed time."""
    recommended_time = float(
        saturation_recommendation["harvest_time_d"]
    )

    interval_data = saturation_validation.loc[
        np.isclose(
            saturation_validation["harvest_time_d"],
            recommended_time,
        )
    ].copy()

    if (
        len(interval_data) != 30
        or interval_data["batch_id"].duplicated().any()
    ):
        raise ValueError("Expected one error per batch at the harvest time.")

    residual_lower, residual_upper = np.quantile(
        interval_data["residual_mg_L"],
        [0.025, 0.975],
    )

    predicted_mab = float(
        saturation_recommendation["predicted_mAb_mg_L"]
    )

    mab_lower = predicted_mab + residual_lower

    mab_upper = predicted_mab + residual_upper

    cost = float(saturation_recommendation["culture_cost_USD_L"])

    cycle_days = recommended_time + DOWNTIME_DAYS

    prediction_intervals = pd.DataFrame([
        {
            "metric": "mAb concentration",
            "units": "mg/L",
            "prediction": predicted_mab,
            "approximate_95_PI_lower": mab_lower,
            "approximate_95_PI_upper": mab_upper,
        },
        {
            "metric": "Culture value",
            "units": "USD/L",
            "prediction": 1000 * predicted_mab,
            "approximate_95_PI_lower": 1000 * mab_lower,
            "approximate_95_PI_upper": 1000 * mab_upper,
        },
        {
            "metric": "Space-time profit",
            "units": "USD/(L*day)",
            "prediction": (1000 * predicted_mab - cost) / cycle_days,
            "approximate_95_PI_lower":
                (1000 * mab_lower - cost) / cycle_days,
            "approximate_95_PI_upper":
                (1000 * mab_upper - cost) / cycle_days,
        },
        {
            "metric": "Space-time yield",
            "units": "mg/(L*day)",
            "prediction": predicted_mab / cycle_days,
            "approximate_95_PI_lower": mab_lower / cycle_days,
            "approximate_95_PI_upper": mab_upper / cycle_days,
        },
    ])

    prediction_intervals.to_csv(
        TABLE_DIR / "recommendation_prediction_intervals.csv",
        index=False,
    )

    interval_data.to_csv(
        TABLE_DIR / "prediction_interval_residuals.csv",
        index=False,
    )

    print("\nAPPROXIMATE 95% PREDICTION INTERVALS")

    print(f"Harvest time: {recommended_time:.4f} days")

    print(f"Batch residuals used: {len(interval_data)}")

    print(prediction_intervals.round(4).to_string(index=False))
    return prediction_intervals, recommended_time



def export_recommendation_figures(prediction_intervals, recommended_time, saturation_grid, saturation_recommendation):
    """Export the recommendation and fixed-time input response figures."""
    recommendation_summary = prediction_intervals.copy()

    recommendation_summary.insert(
        0, "initial_glucose_mM",
        saturation_recommendation["initial_glucose_mM"],
    )

    recommendation_summary.insert(
        1, "initial_glutamine_mM",
        saturation_recommendation["initial_glutamine_mM"],
    )

    recommendation_summary.insert(
        2, "initial_vcd_cells_mL",
        saturation_recommendation["initial_vcd_cells_mL"],
    )

    recommendation_summary.insert(
        3, "harvest_time_d",
        saturation_recommendation["harvest_time_d"],
    )

    recommendation_summary.insert(
        4, "culture_cost_USD_L",
        saturation_recommendation["culture_cost_USD_L"],
    )

    recommendation_summary.to_csv(
        TABLE_DIR / "recommendation_summary.csv",
        index=False,
    )

    same_saturation_conditions = (
        (saturation_grid["initial_glucose_mM"]
         == saturation_recommendation["initial_glucose_mM"])
        & (saturation_grid["initial_glutamine_mM"]
           == saturation_recommendation["initial_glutamine_mM"])
        & (saturation_grid["initial_vcd_cells_mL"]
           == saturation_recommendation["initial_vcd_cells_mL"])
    )

    saturation_curve = saturation_grid.loc[
        same_saturation_conditions
    ].sort_values("harvest_time_d")

    profit_interval = prediction_intervals.loc[
        prediction_intervals["metric"] == "Space-time profit"
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        saturation_curve["harvest_time_d"],
        saturation_curve["predicted_space_time_profit_USD_L_d"],
        color="tab:blue",
        label="Saturation-model prediction",
    )

    ax.vlines(
        recommended_time,
        profit_interval["approximate_95_PI_lower"],
        profit_interval["approximate_95_PI_upper"],
        color="black",
        linewidth=2,
        label="Approximate 95% prediction interval at selected time",
    )

    ax.scatter(
        recommended_time,
        profit_interval["prediction"],
        marker="*",
        s=180,
        color="black",
        zorder=5,
        label="Best predicted point on search grid",
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Harvest time [days]")

    ax.set_ylabel("Space-time profit [USD/(L·day)]")

    ax.set_title(
        "Proposed conditions: glucose 20 mM, glutamine 2 mM,\n"
        "initial VCD 400,000 cells/mL"
    )

    ax.grid(alpha=0.25)

    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "saturation_recommendation.png",
        dpi=300,
    )

    plt.close(fig)

    print("\nSaved tables/recommendation_summary.csv")

    print("Saved figures/saturation_recommendation.png")

    factor_settings = [
        ("initial_glucose_mM", "Initial glucose [mM]"),
        ("initial_glutamine_mM", "Initial glutamine [mM]"),
        ("initial_vcd_cells_mL", "Initial VCD [million cells/mL]"),
    ]

    response_tables = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for column_index, (factor, label) in enumerate(factor_settings):
        # Select grid rows where only this factor varies.
        keep = np.isclose(
            saturation_grid["harvest_time_d"],
            saturation_recommendation["harvest_time_d"],
        )

        for other_factor in CONDITION_COLUMNS:
            if other_factor != factor:
                keep &= np.isclose(
                    saturation_grid[other_factor],
                    saturation_recommendation[other_factor],
                )

        response = saturation_grid.loc[keep].sort_values(factor).copy()
        response["varied_factor"] = factor
        response_tables.append(response)

        x_values = response[factor].to_numpy()
        selected_value = float(saturation_recommendation[factor])

        if factor == "initial_vcd_cells_mL":
            x_values = x_values / 1_000_000
            selected_value = selected_value / 1_000_000

        axes[0, column_index].plot(
            x_values,
            response["predicted_space_time_profit_USD_L_d"],
            marker="o",
            color="tab:blue",
        )

        axes[1, column_index].plot(
            x_values,
            response["predicted_space_time_yield_mg_L_d"],
            marker="o",
            color="tab:green",
        )

        for row_index in [0, 1]:
            ax = axes[row_index, column_index]
            ax.axvline(
                selected_value,
                color="gray",
                linestyle="--",
                label="Proposed input value",
            )
            ax.set_xlabel(label)
            ax.grid(alpha=0.25)

        axes[0, column_index].set_ylabel(
            "Predicted profit [USD/(L·day)]"
        )
        axes[1, column_index].set_ylabel(
            "Predicted yield [mg/(L·day)]"
        )

    axes[0, 0].legend(fontsize=8)

    fig.suptitle(
        "Starting-condition effects at a fixed harvest time of "
        f"{recommended_time:.2f} days\n"
        "Other inputs held at the proposed conditions"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(
        FIGURE_DIR / "initial_condition_effects.png",
        dpi=300,
    )

    plt.close(fig)

    condition_responses = pd.concat(response_tables, ignore_index=True)

    condition_responses.to_csv(
        TABLE_DIR / "initial_condition_effects.csv",
        index=False,
    )

    print("\nSaved figures/initial_condition_effects.png")

    print("Saved tables/initial_condition_effects.csv")



def export_model_comparison(X, best_power, best_tau, cv_splits, model_comparison, scoring, selected_saturation_model, y):
    """Include the selected saturation model in the comparison table."""
    selected_saturation_scores = cross_validate(
        selected_saturation_model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        return_train_score=True,
        error_score="raise",
    )

    saturation_comparison_row = pd.DataFrame([{
        "model": (
            f"Saturation: tau={best_tau:g} days, power={best_power}"
        ),
        "mean_validation_rmse_mg_L":
            -selected_saturation_scores["test_rmse"].mean(),
        "std_validation_rmse_mg_L":
            selected_saturation_scores["test_rmse"].std(ddof=1),
        "mean_validation_mae_mg_L":
            -selected_saturation_scores["test_mae"].mean(),
        "mean_validation_r2":
            selected_saturation_scores["test_r2"].mean(),
    }])

    final_model_comparison = pd.concat(
        [model_comparison, saturation_comparison_row],
        ignore_index=True,
    ).sort_values("mean_validation_rmse_mg_L")

    final_model_comparison.to_csv(
        TABLE_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nUPDATED MODEL COMPARISON")

    print(final_model_comparison.round(4).to_string(index=False))



def run_workflow():
    """Run the complete analysis and regenerate the CSV and PNG outputs."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    economic_results, initial_conditions = load_and_calculate_economics()
    X, baseline_scores, cv_splits, linear_scores, polynomial_model, polynomial_scores, scoring, y = evaluate_reference_models(economic_results)
    model_comparison = diagnose_polynomial_model(X, baseline_scores, cv_splits, economic_results, linear_scores, polynomial_model, polynomial_scores, y)
    search_results = inspect_polynomial_optimization(X, economic_results, initial_conditions, polynomial_model, y)
    best_power, best_tau, saturation_validation, selected_saturation_model = select_saturation_model(X, cv_splits, economic_results, scoring, y)
    saturation_grid, saturation_recommendation = optimize_saturation_model(X, search_results, selected_saturation_model, y)
    prediction_intervals, recommended_time = estimate_prediction_intervals(saturation_recommendation, saturation_validation)
    export_recommendation_figures(prediction_intervals, recommended_time, saturation_grid, saturation_recommendation)
    export_model_comparison(X, best_power, best_tau, cv_splits, model_comparison, scoring, selected_saturation_model, y)
    from .coverage_validation import validate_interval_coverage
    validate_interval_coverage(economic_results, initial_conditions)

