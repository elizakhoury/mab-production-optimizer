"""Retrospective nested grouped validation of approximate prediction intervals.

No conformal or conditional 95% coverage guarantee is claimed. The saturation
model family was chosen during earlier exploration of these same data.
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .functions import (
    CONDITION_COLUMNS, FEATURE_COLUMNS, TABLE_DIR, DOWNTIME_DAYS,
    saturation_features,
)


def validate_interval_coverage(economic_results, initial_conditions):
    """Keep outer test condition groups out of tuning and error calibration."""
    X = economic_results[FEATURE_COLUMNS].reset_index(drop=True)
    y = economic_results['mAb_mg_L'].reset_index(drop=True)
    groups = economic_results.groupby(CONDITION_COLUMNS, sort=True).ngroup()
    groups = groups.reset_index(drop=True)
    data = economic_results.reset_index(drop=True)
    all_predictions = []
    fold_records = []
    tuning_records = []

    # Planned measurement times and known input catalogue, not batch outcomes.
    grid = pd.DataFrame(product(
        sorted(initial_conditions['Glucose [mM]'].unique()),
        sorted(initial_conditions['Glutamine [mM]'].unique()),
        sorted(initial_conditions['VCD [cells/mL]'].unique()),
        np.arange(8, 673, 8) / 24,
    ), columns=FEATURE_COLUMNS)
    costs = (10 * grid.iloc[:, 0] + 20 * grid.iloc[:, 1]
             + 0.0005 * grid.iloc[:, 2])

    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=5).split(X, y, groups), start=1
    ):
        print(f'Interval validation: outer fold {fold}/5', flush=True)
        X_train, y_train = X.iloc[train], y.iloc[train]
        inner_groups = groups.iloc[train]
        assert set(inner_groups).isdisjoint(set(groups.iloc[test]))
        inner_splits = list(GroupKFold(n_splits=5).split(
            X_train, y_train, inner_groups))
        best_score, best_model, best_parameters = np.inf, None, None
        for tau, power in product([0.5, 1., 2., 4., 8.], [1, 2, 3]):
            candidate = Pipeline([
                ('features', FunctionTransformer(saturation_features,
                    kw_args={'tau': tau, 'power': power})),
                ('regression', LinearRegression(fit_intercept=False)),
            ])
            scores = cross_validate(candidate, X_train, y_train,
                cv=inner_splits, scoring='neg_root_mean_squared_error',
                error_score='raise')
            rmse = -scores['test_score'].mean()
            tuning_records.append(dict(outer_fold=fold, tau_days=tau,
                                       power=power, inner_rmse_mg_L=rmse))
            if rmse < best_score:
                best_score, best_model = rmse, candidate
                best_parameters = (tau, power)

        # Reproduce the current empirical interval recipe inside outer training.
        # These inner errors are reused after tuning, but outer outcomes remain
        # untouched until assessment. Quantiles are approximate, not conformal.
        inner_predictions = cross_val_predict(best_model, X_train, y_train,
                                               cv=inner_splits)
        errors = pd.DataFrame({
            'harvest_time_d': X_train['harvest_time_d'].to_numpy(),
            'residual': y_train.to_numpy() - inner_predictions,
        })
        quantiles = errors.groupby('harvest_time_d')['residual'].quantile(
            [0.025, 0.975]).unstack()
        fitted = clone(best_model).fit(X_train, y_train)
        test_predictions = fitted.predict(X.iloc[test])
        result = data.iloc[test].copy()
        result['outer_fold'] = fold
        result['condition_group'] = groups.iloc[test].to_numpy()
        result['prediction_mg_L'] = test_predictions
        result['lower_mg_L'] = test_predictions + result['harvest_time_d'].map(
            quantiles[0.025]).to_numpy()
        result['upper_mg_L'] = test_predictions + result['harvest_time_d'].map(
            quantiles[0.975]).to_numpy()
        result['covered'] = ((result['mAb_mg_L'] >= result['lower_mg_L'])
                             & (result['mAb_mg_L'] <= result['upper_mg_L']))
        result['width_mg_L'] = result['upper_mg_L'] - result['lower_mg_L']
        result['profit_width_USD_L_d'] = (
            1000 * result['width_mg_L'] /
            (result['harvest_time_d'] + DOWNTIME_DAYS))
        result['predicted_profit_USD_L_d'] = (
            1000 * result['prediction_mg_L'] - result['culture_cost_USD_L']
        ) / (result['harvest_time_d'] + DOWNTIME_DAYS)

        # Select the global candidate using only this fold's fitted model.
        grid_prediction = fitted.predict(grid)
        grid_profit = (1000 * grid_prediction - costs) / (
            grid['harvest_time_d'] + DOWNTIME_DAYS)
        feasible = np.where(grid_prediction >= 0, grid_profit, -np.inf)
        best_index = int(np.argmax(feasible))
        assert np.isfinite(feasible[best_index])
        proposal = grid.iloc[best_index]
        result['at_fold_selected_time'] = np.isclose(
            result['harvest_time_d'], proposal['harvest_time_d'])
        # This time is evaluated at held-out conditions, not necessarily the
        # global proposed condition, which may have no measured batch.
        result['at_batch_selected_time'] = False
        positive = result[result['harvest_time_d'] > 0]
        selected = positive.groupby('batch_id')['predicted_profit_USD_L_d'].idxmax()
        result.loc[selected, 'at_batch_selected_time'] = True
        all_predictions.append(result)
        fold_records.append(dict(outer_fold=fold,
            training_groups=inner_groups.nunique(),
            test_groups=groups.iloc[test].nunique(),
            selected_tau_days=best_parameters[0], selected_power=best_parameters[1],
            inner_rmse_mg_L=best_score,
            proposed_glucose_mM=proposal.iloc[0],
            proposed_glutamine_mM=proposal.iloc[1],
            proposed_vcd_cells_mL=proposal.iloc[2],
            proposed_harvest_time_d=proposal.iloc[3]))

    predictions = pd.concat(all_predictions, ignore_index=True)
    assert len(predictions) == len(data)
    assert not predictions.duplicated(['batch_id', 'harvest_time_d']).any()
    assert predictions[['lower_mg_L', 'upper_mg_L']].notna().all().all()
    scopes = {
        'All positive times': predictions['harvest_time_d'] > 0,
        'Days 1 to 10': predictions['harvest_time_d'].between(1, 10),
        'Fold-selected time at held-out conditions': predictions['at_fold_selected_time'],
        'Model-selected harvest for each held-out batch': predictions['at_batch_selected_time'],
    }
    summaries, group_summaries = [], []
    for label, mask in scopes.items():
        subset = predictions.loc[mask]
        group = subset.groupby('condition_group').agg(
            coverage=('covered', 'mean'), mean_width_mg_L=('width_mg_L', 'mean'))
        group['scope'] = label
        group_summaries.append(group.reset_index())
        summaries.append(dict(scope=label, observations=len(subset),
            condition_groups=group.shape[0], covered_observations=int(subset['covered'].sum()),
            pooled_coverage_percent=100 * subset['covered'].mean(),
            equal_group_coverage_percent=100 * group['coverage'].mean(),
            mean_width_mg_L=subset['width_mg_L'].mean(),
            mean_profit_width_USD_L_d=subset['profit_width_USD_L_d'].mean()))
    summary = pd.DataFrame(summaries)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(TABLE_DIR/'nested_interval_predictions.csv', index=False)
    pd.DataFrame(fold_records).to_csv(TABLE_DIR/'nested_interval_folds.csv', index=False)
    pd.DataFrame(tuning_records).to_csv(TABLE_DIR/'nested_interval_tuning.csv', index=False)
    pd.concat(group_summaries).to_csv(TABLE_DIR/'nested_interval_group_coverage.csv', index=False)
    summary.to_csv(TABLE_DIR/'nested_interval_coverage.csv', index=False)
    print('\nRETROSPECTIVE NESTED INTERVAL VALIDATION')
    print(summary.round(4).to_string(index=False))
    print('Coverage at the final unmeasured optimized conditions remains unverified.')
    return summary
