# soft_sensor_lib.py
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, make_scorer
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor as _CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class CatBoostRegressorSK(_CatBoostRegressor, BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around CatBoostRegressor."""
    pass

class SoftSensorPipeline:
    """
    Collection of utilities extracted from the notebook for reuse/import.
    Use functions as: SoftSensorPipeline.train_85_test_15_split(df)
    """

    # -----------------------
    # Data helpers
    # -----------------------
    @staticmethod
    def load_csv_with_date(path, date_col='date', decimal=','):
        df = pd.read_csv(path, decimal=decimal)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        return df

    @staticmethod
    def pad_incomplete_hour_blocks(df, rows_per_hour=180):
        df_copy = df.copy()
        idx_counts = df_copy.index.value_counts().sort_index()
        missing = idx_counts[idx_counts != rows_per_hour]
        for time_stamp, count in missing.items():
            num_pads = rows_per_hour - count
            padding = df_copy.loc[time_stamp].tail(1)
            padding = pd.concat([padding] * num_pads)
            df_copy = pd.concat([df_copy, padding]).sort_index()
        return df_copy

    @staticmethod
    def reconstruct_20s_and_1min(df_mining):
        df_temp = df_mining.copy()
        df_temp['hour'] = df_temp.index
        df_temp['idx_in_hour'] = df_temp.groupby('hour').cumcount()
        df_temp.index = df_temp['hour'] + pd.to_timedelta(df_temp['idx_in_hour'] * 20, unit='s')
        df_temp = df_temp.drop(columns=['hour', 'idx_in_hour'])

        full_20s_index = pd.date_range(df_temp.index.min(), df_temp.index.max(), freq='20s')
        df_20s = df_temp.reindex(full_20s_index)
        df_20s.index.name = 'date'

        df_1min = df_20s.resample('1min').mean(numeric_only=True)
        return df_20s, df_1min

    # -----------------------
    # Train/Test splits
    # -----------------------
    @staticmethod
    def train_85_test_15_split(df, debug=False):
        df_ml = df.copy().dropna()
        X = df_ml.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])
        y = df_ml['% Silica Concentrate']

        split_idx = int(len(df_ml) * 0.85)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if debug:
            with pd.option_context('display.max_columns', None):
                print('X_train ', X_train.shape, ':', sep='')
                display(X_train)
                print('y_train ', y_train.shape, ':', sep='')
                display(y_train.to_frame().T)
                print('\\nX_test ', X_test.shape, ':', sep='')
                display(X_test)
                print('y_test ', y_test.shape, ':', sep='')
                display(y_test.to_frame().T)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def train_1_test_59_split(df, debug=False):
        df_ml = df.copy().dropna()
        X = df_ml.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])
        y = df_ml['% Silica Concentrate']

        if len(df_ml) % 60 != 0:
            print(f"⚠️ Warning: Dataframe length ({len(df_ml)}) is not divisible by 60. Last incomplete block will still be handled.")

        train_idx = df_ml.iloc[::60].index
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        test_mask = ~df_ml.index.isin(train_idx)
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        if debug:
            with pd.option_context('display.max_columns', None):
                print('X_train ', X_train.shape, ':', sep='')
                display(X_train)
                print('y_train ', y_train.shape, ':', sep='')
                display(y_train.to_frame().T)
                print('\\nX_test ', X_test.shape, ':', sep='')
                display(X_test)
                print('y_test ', y_test.shape, ':', sep='')
                display(y_test.to_frame().T)

        return X_train, y_train, X_test, y_test

    # -----------------------
    # Feature Engineering
    # -----------------------
    @staticmethod
    def make_features(df_1min):
        df = df_1min.copy()

        for i in range(1, 25):
            df[f'% Iron Concentrate {i}h ago'] = df['% Iron Concentrate'].shift(i*60)
            df[f'% Silica Concentrate {i}h ago'] = df['% Silica Concentrate'].shift(i*60)

        df['% Iron Concentrate 7d ago'] = df['% Iron Concentrate'].shift(7*24*60)
        df['% Silica Concentrate 7d ago'] = df['% Silica Concentrate'].shift(7*24*60)

        iron_hourly   = df['% Iron Concentrate'].groupby(df.index.floor('h')).first()
        silica_hourly = df['% Silica Concentrate'].groupby(df.index.floor('h')).first()
        hour_index = df.index.floor('h')

        for hr in [6, 24]:
            roll_mean = iron_hourly.rolling(window=hr, min_periods=hr).mean().shift(1)
            roll_std  = iron_hourly.rolling(window=hr, min_periods=hr).std().shift(1)
            df[f'% Iron Conc rolling mean {hr}h'] = hour_index.map(roll_mean)
            df[f'% Iron Conc rolling std {hr}h']  = hour_index.map(roll_std)

        for hr in [6, 12, 24]:
            roll_mean = silica_hourly.rolling(window=hr, min_periods=hr).mean().shift(1)
            roll_std  = silica_hourly.rolling(window=hr, min_periods=hr).std().shift(1)
            df[f'% Silica Conc rolling mean {hr}h'] = hour_index.map(roll_mean)
            df[f'% Silica Conc rolling std {hr}h']  = hour_index.map(roll_std)

        df['pH_diff_1min'] = df['Ore Pulp pH'].diff(1)
        df['AirFlow07_diff_1min'] = df['Flotation Column 07 Air Flow'].diff(1)

        df['Amina_to_pulp'] = df['Amina Flow'] / df['Ore Pulp Flow']
        df['Starch_to_pulp'] = df['Starch Flow'] / df['Ore Pulp Flow']

        df['AirFlow_total'] = df.filter(like='Air Flow').sum(axis=1)
        df['Level_total'] = df.filter(like='Level').sum(axis=1)

        for col in ['AirFlow_total', 'Level_total']:
            df[f'{col}_roll_mean_10m'] = df[col].rolling(window=10, min_periods=10).mean().shift(1)
            df[f'{col}_roll_std_10m'] = df[col].rolling(window=10, min_periods=10).std().shift(1)

        df['day_of_week'] = df.index.dayofweek
        df['hour_of_day'] = df.index.hour

        return df

    # -----------------------
    # Modeling helpers
    # -----------------------
    @staticmethod
    def make_models():
        models = []

        elasticnet_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=5000, random_state=42)),
        ])
        elasticnet_params = {
            "model__alpha":   [0.01, 0.05, 0.1, 0.5, 1.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        }
        models.append(("ElasticNet", elasticnet_pipeline, elasticnet_params))

        cat = CatBoostRegressorSK(
            loss_function="MAE",
            random_seed=42,
            verbose=False,
            task_type="GPU",
        )
        cat_params = {
            "depth":        [6, 8],
            "learning_rate":[0.03, 0.1],
            "n_estimators": [300, 700],
            "l2_leaf_reg":  [1, 3],
        }
        # Below are the parameters for the best performing CatBoost model after conducting CV grid search on
        # both 85-15 and 1-59 train/test splits using cat_params dictionary
        cat_params_best = {"depth": [6], "learning_rate": [0.03], "n_estimators": [300], "l2_leaf_reg": [3]}
        models.append(("CatBoost", cat, cat_params_best))

        lgbm = LGBMRegressor(objective="regression", random_state=42, n_jobs=-1)
        lgbm_params = {
            "num_leaves":    [31, 63],
            "learning_rate": [0.05],
            "n_estimators":  [300, 500],
        }
        # Below are the parameters for the best performing CatBoost model after conducting CV grid search on
        # both 85-15 and 1-59 train/test splits using lgbm_params dictionary
        lgbm_params_best = {"num_leaves":[31],"learning_rate":[0.05],"n_estimators":[500]}
        models.append(("LightGBM", lgbm, lgbm_params_best))

        return models

    @staticmethod
    def tune_and_evaluate_models(models, X_train, y_train, X_test, y_test, folds=3, n_jobs=2, verbose=1):
        results = []
        tscv = TimeSeriesSplit(n_splits=folds)
        mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

        for name, estimator, param_grid in models:
            print(f"\n===== Tuning {name} =====")
            timer = time.time()

            search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                                  scoring=mape_scorer, cv=tscv, n_jobs=n_jobs, verbose=verbose)
            search.fit(X_train, y_train)
            timer = time.time() - timer
            tune_train_time = time.strftime("%M:%S", time.gmtime(timer))

            best_model = search.best_estimator_
            print(f"Best params for {name}: {search.best_params_}")

            y_pred = best_model.predict(X_test)

            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)

            print(f"{name} Test MAPE: {mape:.4f}")
            print(f"{name} Test R²:   {r2:.4f}")
            print(f"{name} Test MAE:  {mae:.4f}")
            print(f"{name} Time:      {tune_train_time}")

            results.append({
                "model": name,
                "best_params": search.best_params_,
                "test_MAPE": mape,
                "test_R2": r2,
                "test_MAE": mae,
                "best_estimator": best_model,
                "tune_train_time": tune_train_time,
                "folds": folds,
            })

        summary = pd.DataFrame([{k: v for k, v in r.items() if k != "best_estimator"} for r in results])
        return results, summary

    @staticmethod
    def train_test_on_baseline_model(df, splitter, debug=False):
        X_train, y_train, X_test, y_test = splitter(df, debug)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape  = mean_absolute_percentage_error(y_test, y_pred)
        r2    = r2_score(y_test, y_pred)
        mae   = mean_absolute_error(y_test, y_pred)

        print("\n===================================")
        print("Baseline Linear Regression Results:")
        print(f"MAPE: {mape * 100:.2f}%")
        print(f"R²:   {r2:.4f}")
        print(f"MAE:  {mae:.4f}")
        print("===================================")

        return model, (mape, r2, mae), (X_train, y_train, X_test, y_test)

    @staticmethod
    def run_model_comparison(df_features, split_func, folds=3, n_jobs=2, debug=True):
        X_train, y_train, X_test, y_test = split_func(df_features)
        if debug:
            print("X_train shape:", X_train.shape, " X_test shape:", X_test.shape)
        models = SoftSensorPipeline.make_models()
        results, summary = SoftSensorPipeline.tune_and_evaluate_models(
            models=models, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            folds=folds, n_jobs=n_jobs, verbose=1
        )
        return results, summary