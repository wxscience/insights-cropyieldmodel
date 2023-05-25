import numpy as np
import pandas as pd
import datetime as dt
import logging
from itertools import product
from dateutil.relativedelta import relativedelta
from pprint import pprint

# sklearn libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR, NuSVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# shared functions
import custom_libraries.shared_functions as sf
import realtime_predictions.run_crop_model as cm

import warnings


def _init_logger():
    logger = logging.getLogger("ul-crop_model")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


_init_logger()
_logger = logging.getLogger("ul-crop_model.train")


def train_regression(x_model, y_model, best_params=None):
    def evaluate(model, x_vals, y_vals):
        predictions = model.predict(x_vals)
        errors = abs(predictions - y_vals)
        mape = 100 * np.mean(errors / y_vals)
        accuracy = 100 - mape

        bias = np.mean(predictions - y_vals)
        mae = np.mean(errors)

        print("--------------------------------------")
        print("Model Performance")
        print("MAE: {:0.4f}".format(mae))
        print("Average Bias: {:0.4f}".format(bias))
        print("Average Stdev: {:0.4f}".format(np.std(predictions - y_vals)))
        print("Accuracy = {:0.2f}%.".format(accuracy))
        print("--------------------------------------")

        return accuracy, bias, mae

    if best_params is None:
        # hyperparameter tuning for random forest regression
        n_estimators = [int(x) for x in np.linspace(start=2000, stop=8000, num=10)]
        max_features = ["sqrt", "log2"]
        max_depth = [int(x) for x in np.linspace(start=40, stop=400, num=11)]
        max_depth.append(None)
        min_samples_split = [int(x) for x in np.linspace(start=4, stop=20, num=5)]
        min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=5, num=3)]

        random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }

        # use the random grid to search for best hyperparameters
        if len(y_model) < 20:
            cv_num = 3
        else:
            cv_num = 5

        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100,
            cv=cv_num,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )

        wts = np.arange(10, 95, 85 / len(x_model.index))
        rf_random.fit(x_model, y_model, sample_weight=wts)
        best_params = rf_random.best_params_
        best_estimator = rf_random.best_estimator_
    else:
        best_estimator = RandomForestRegressor(**best_params)
        best_estimator.fit(x_model, y_model)

    # accuracy test
    accuracy, bias, mae = evaluate(best_estimator, x_model, y_model)

    # pprint(best_params)
    # pprint(accuracy)

    # permutation feature importances
    r = permutation_importance(best_estimator, x_model, y_model)
    feature_names = []
    for i in r.importances_mean.argsort()[::-1]:
        """
        print(
            f"{x_model.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}"
        )
        """
        feature_names.append(x_model.columns[i])

    # 80% cutoff
    x_subset_names = feature_names[0:10]

    return best_estimator, best_params, accuracy, bias, mae, x_subset_names


def main(config, country, state):
    warnings.filterwarnings("ignore")

    ver = config["ver"]
    _logger.info(f"Beginning training: {country} -- {state}")

    # set training years
    if state == "new-york":
        ty1 = 1999
    else:
        ty1 = int(config["training"]["train_years"][country][0])
    ty2 = int(config["training"]["train_years"][country][1])

    # Load in the truth file
    crop_truth_df = sf.load_crop_truth(config, country, state).loc[:2021]
    y_xv = crop_truth_df.copy()["yield"]

    # create feature for below; clip for training period
    yield_feature_df = crop_truth_df.copy(deep=True)
    yield_feature_df.index = [i + 1 for i in crop_truth_df.index]

    yield_feature_df = yield_feature_df.loc[
        [i for i in yield_feature_df.index if i >= ty1], "yield"
    ]

    crop_truth_df = crop_truth_df.loc[[i for i in crop_truth_df.index if i >= ty1], :]

    print("---------------")
    print(crop_truth_df)

    # load in the monthly weather data (already observed)
    tp_data_df = sf.load_tp(
        config,
        np.nan,
        state,
        country=country,
        training=True,
    )

    # load in the terraclimate vpd and soil moisture data
    vpd_df = sf.load_terraclimate(
        config,
        "vpd",
        state,
        country=country,
    )

    soil_df = sf.load_terraclimate(
        config,
        "soil",
        state,
        country=country,
    )

    # load in the weather data: constructed analog forecast data
    ca_fcst_df = sf.load_ca(config, state, country=country)

    # get the trend, which is usually the stat to beat
    trend_df = sf.detrend_df(crop_truth_df, win=5, return_trend=True)

    """
    Add ENSO, MJO, etc. data as predictors
    Especially important for Brazil,
    but also helpful for U.S.
    """
    enso_df = sf.load_enso(config)
    mjo_df = sf.load_mjo(config)

    """
    Train the model!
    model will run monthly and make production output forecast based on these features:
    - vpd and soil moisture during current month
    - constructed analog forecast of tmax, tmin, and precip for the next 12 months
    - last year's yield values
    """

    # set growing season information
    months_list = config["growing-season"][country]

    # set up date list
    date_list = pd.date_range(
        start=f"{ty1}-01-01",
        end=f"2021-12-31",
        freq="MS",
    )

    results_df = pd.DataFrame(
        index=months_list,
        columns=["state", "month", "mae", "accuracy", "bias", "important_features"],
    )

    # loop through each month in the growing season
    # must do in order b/c most recent forecasts are used as inputs
    for month in months_list:
        """
        REMEMBER: crop_truth_df years are the FIRST year in the couple.
        So, 1995 means 1995/1996 season. Planted in 1995, harvested in 1996.
        """
        # we need to load in the training info so we can figure out
        # which features to use, and add bias

        _logger.info(f"Starting: month - {month}")

        months_passed = months_list[0 : months_list.index(month) + 1]

        # use this to label forecast columns correctly, even though it might be outside of the growing season
        months_future = pd.date_range(
            start=dt.datetime(2000, month, 1),
            end=dt.datetime(2000, month, 1) + relativedelta(months=4),
            freq="MS",
        ).month.tolist()[1:]

        monthly_date_list = [d for d in date_list if d.month in months_passed]

        # Load in the latest USDA planting season data
        # pp_df = sf.load_crop_truth_usda(config, state, month)

        # make them relative to the season's starting date
        # e.g. for brazil, the model in Dec must include data from Sep and Oct.
        # set up features, must clip everything to fit the above date_list:
        x_past_df = pd.DataFrame(index=monthly_date_list)

        # put the features into x_past_df
        x_past_df.loc[:, "vpd"] = vpd_df.loc[monthly_date_list, "vpd"]
        x_past_df.loc[:, "soil"] = soil_df.loc[monthly_date_list, "soil"]
        x_past_df.loc[:, "enso"] = enso_df.loc[monthly_date_list, "enso"]
        x_past_df.loc[:, ("rmm1", "rmm2", "phase", "amplitude")] = mjo_df.loc[
            monthly_date_list, :
        ]
        x_past_df.loc[:, ("precip", "tmin", "tmax", "frost")] = tp_data_df.loc[
            monthly_date_list, :
        ]

        # reshape and clean up this thing so that col names are meaningful:
        # feature_x, where x is the valid month
        x_past_df = x_past_df.set_index(x_past_df.index.month)

        x_df = pd.DataFrame(
            x_past_df.values.reshape(
                int(len(x_past_df) / len(months_passed)),
                len(x_past_df.columns) * len(months_passed),
            )
        )

        x_df.columns = [
            "{}_{}".format(j, i) for i, j in product(months_passed, x_past_df.columns)
        ]

        # deal with constructed analog, only need fcsts from the CURRENT month
        ca_fcst_x_df = ca_fcst_df.loc[monthly_date_list, :].copy(deep=True)

        # fix the ca_fcst_x_df names so instead of f values we just have the actual month
        ca_fcst_cols = [
            ca_fcst_x_df.columns.str.split("_")[i][0]
            for i in range(len(ca_fcst_x_df.columns))
        ]
        ca_fcst_cols = [
            f"{ca_fcst_cols[i]}_{np.tile(months_future[0:3], int(len(ca_fcst_cols)/3))[i]}"
            for i in range(len(ca_fcst_cols))
        ]
        ca_fcst_x_df.columns = ca_fcst_cols

        x_df.loc[:, ca_fcst_x_df.columns] = ca_fcst_x_df.loc[
            ca_fcst_x_df.index.month == month, :
        ].values

        x_df.loc[:, "trend"] = trend_df.loc[:, "yield_trend"].values

        x = x_df.set_index(trend_df.index, drop=True)

        # lastly, use either previous year yield (if first month) or prev month
        # yield forecast as a feature
        if len(months_passed) == 1:
            # first month, use last year's number
            x.loc[:, "prev_yield"] = yield_feature_df
        else:
            x.loc[:, "prev_yield"] = model_fcsts

        # set up targets
        y = crop_truth_df.loc[:, "yield"]

        """
        Train the model. No need to normalize for RF.
        """
        x_train = x.dropna().loc[:2019]
        y_train = y.loc[x.index].loc[:2019]

        # train model only thru 2019
        model, model_params, accuracy, bias, mae, feature_names = train_regression(
            x_train, y_train
        )

        # save model
        model_file = f'{config["bucket_out_path"]}trained_models/weather_model/v{ver}/{country}/{country}-{state}-{month}.joblib'

        sf.save_model(model, model_file)

        # Grab the forecasts for each month and add to the df
        model_fcsts = pd.Series(index=y.index)
        model_fcsts[:] = model.predict(x)

        results_df.loc[
            month, ["state", "month", "mae", "accuracy", "bias", "important_features"]
        ] = [state, month, mae, accuracy, bias, feature_names]

        # cross validate model; get predictions for 2020, 2021, 2022
        predictions_df = pd.DataFrame(
            index=np.arange(x.index[-1] - 15, x.index[-1] + 1),
            columns=[
                "state",
                "year",
                "month",
                "model_mape",
                "prediction",
                "top_features",
                "acreage",
                "yield",
                "APE",
            ],
        )

        for yyyy in np.arange(x.index[-1] - 10, x.index[-1] + 1):
            print(f"Starting year {yyyy} for {month}...")

            # drop the prediction year for proper cross validation (xv)
            x_xv = x.copy(deep=True).drop(labels=yyyy)
            y_xv = y.copy(deep=True).drop(labels=yyyy)

            # add in the best params so we don't have to do a grid search each time
            (
                model_xv,
                model_params,
                accuracy_xv,
                _,
                _,
                _,
            ) = train_regression(
                x_xv.reset_index(drop=True),
                y_xv.reset_index(drop=True),
                best_params=model_params,
            )

            fcst_df = pd.DataFrame(index=x_xv.index, columns=["fcst", "obs"])
            fcst_df["fcst"] = model_xv.predict(x_xv)
            fcst_df["obs"] = y_xv

            obs = y[yyyy]
            fcst = model_xv.predict(x.loc[yyyy, :].values.reshape(1, -1)).item()

            ape = np.abs(obs - fcst) / obs * 100

            predictions_df.loc[yyyy, :] = (
                state,
                yyyy,
                month,
                100 - mae,
                fcst,
                feature_names[0:10],
                crop_truth_df.loc[yyyy, "area_harvested"],
                obs,
                ape,
            )

        print(f"{yyyy}-{month}")
        predictions_df.to_csv(
            f'{config["bucket_out_path"]}results_{country}/backtest_predictions/v{ver}/{country}_{state}_{month}.csv'
        )

        results_df.to_csv(
            f'{config["bucket_out_path"]}validation_{country}/v{ver}/{country}_{state}.csv',
        )

    return True


if __name__ == "__main__":
    main()
