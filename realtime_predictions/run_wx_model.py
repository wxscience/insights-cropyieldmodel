import numpy as np

import pandas as pd
import xarray as xr
import joblib
import logging
import datetime as dt
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
import realtime_predictions.run_crop_model as run_crop_model
from dateutil.relativedelta import relativedelta

# shared functions
import custom_libraries.shared_functions as sf


def load_ca_forecast(config, run_date, state, country):
    """
    Load in the latest CA forecast
    """

    # the GEFS will be used for the first month, then we'll use CA after
    # we always want the GEFS to be based on the latest forecast
    # the latest forecast will be 00z from yesterday
    gefs_run_date = run_date - dt.timedelta(days=1)

    print(f"GEFS Run Date: {gefs_run_date:%Y%m%d}")

    gefs = sf.load_gefs(config, gefs_run_date, state, country)

    if country == "argentina":
        # argentina and brazil share the file b/c it covers all of SA
        ca_path = f'{config["realtime"]["ca_path"]}predictions_brazil_ca_{run_date:%Y%m%d}.zarr'
    else:
        ca_path = f'{config["realtime"]["ca_path"]}predictions_{country.lower()}_ca_{run_date:%Y%m%d}.zarr'

    ca_forecast = xr.open_dataset(ca_path, engine="zarr")

    ca_forecast_masked = sf.mask_data(ca_forecast, state, country)
    ca_forecast_masked = ca_forecast_masked.sel(fcst_time=np.arange(0, 4))

    ca_forecast = ca_forecast_masked.mean(["lat", "lon"]).to_dataframe()[
        ["precip", "tmax", "tmin"]
    ]

    """
    since the first row is for the current month and will be combined with month-to-date data
    we should weight it by number of days that it represents relative to the length of the month
    """
    wgt = sf.month_weight(gefs_run_date)

    # gefs is already weighted
    ca_forecast.loc[0, ["tmax", "tmin"]] = (
        ca_forecast.loc[0, ["tmax", "tmin"]] * (1 - wgt)
        + gefs.loc[gefs_run_date.month, ["tmax", "tmin"]]
    )
    ca_forecast.loc[0, ["precip"]] = (
        ca_forecast.loc[0, ["precip"]] + gefs.loc[gefs_run_date.month, ["precip"]]
    )

    # frost is hard b/c the constructed analog model can't predict daily frost values
    ca_forecast["frost"] = np.nan

    ca_forecast.loc[0, "frost"] = gefs.loc[gefs_run_date.month, "frost"]

    # extrapolate frost forward
    ca_forecast["frost"] = ca_forecast.loc[:, "frost"].fillna(method="ffill")

    # for now, only the first frost entry (cur month) can be picked since the model isn't trained on CA frost fcsts
    ca_forecast.index = pd.date_range(
        start=run_date.replace(day=1),
        end=run_date + relativedelta(months=ca_forecast.index[-1]),
        freq="MS",
    )

    return ca_forecast


def load_realtime_tc(config, var, state, country, months_list):
    zarr_path = config["realtime"]["gdas_path"]

    # use the pre-processed zarr data to get soil and vpd numbers
    ds = xr.open_zarr(f"{zarr_path}{var}.zarr")

    if var == "soil":
        # soil source file names this soilw instead of soil
        ds = ds.rename_vars({"soilw": "soil"})

    if len(months_list == 1):
        # if it's the first month, ds will be empty; use prev month
        ds = ds.isel(time=len(ds.time) - 1)
        # add time dim
        ds = ds.expand_dims("time")
    else:
        # sometimes it takes a few days for the new month to be available
        ds = ds.where(ds.time.isin(months_list), drop=True)

    ds_masked = sf.mask_data(ds, state, country)
    df_masked = ds_masked.mean(["lat", "lon"]).to_dataframe()

    climo_list = sf.load_terraclimate(
        config,
        var,
        state,
        years=np.arange(1991, 2020),
        country=country,
        return_climo=True,
    ).loc[:, var]

    climo = climo_list[months_list.month]

    # if the ends don't align then the VPD data is lagging (which is common for the first few days each month)
    if (
        df_masked.index[-1] + relativedelta(months=1)
    ).month == months_list.month.values[-1]:
        # do the same for the current month, which is never here
        df_masked.loc[df_masked.index[-1] + relativedelta(months=1), var] = np.nan
    else:
        temp_df = pd.DataFrame(index=months_list, columns=[var])
        temp_df[var] = df_masked.iloc[-1, 1]
        df_masked = temp_df.copy()

    # interpolate downward since these values shouldn't change drastically month to month
    df_masked = df_masked.interpolate(method="linear")

    df_masked = df_masked[var] - climo.values

    col_names = [f"{var}_{x}" for x in months_list.month.to_list()]
    df = pd.DataFrame.from_dict(
        dict(zip(col_names, df_masked.values)), orient="index"
    ).T

    return df


def get_crop_data(config, state, run_date, wx_pred, country):
    df_cols = ["state", "year", "month", "prediction", "acreage", "yield"]

    backtest_path = f"{config['bucket_in_path']}output/results_{country}/backtest_predictions/v{config['ver']}/{country}_{state}_{run_date.month}.csv"
    backtest_data_df = pd.read_csv(backtest_path).iloc[:, 1:].dropna()

    # remove multiple header rows
    backtest_data_df = backtest_data_df.loc[
        backtest_data_df.ne(backtest_data_df.columns).any(axis=1), df_cols
    ]

    # merge the two
    backtest_data_df["year"] = backtest_data_df["year"].astype(int)
    backtest_data_df["month"] = backtest_data_df["month"].astype(int)
    backtest_data_df["prediction"] = backtest_data_df["prediction"].astype(float)

    mi = pd.MultiIndex.from_frame(backtest_data_df[["year", "month", "state"]])
    backtest_data_df = backtest_data_df.set_index(mi, drop=True).iloc[:, 3:]

    # run the crop model now
    crop_df = run_crop_model.main(
        config,
        run_date,
        country,
        state,
        backtest_data_df[["prediction", "yield"]],
        wx_pred,
    )

    return crop_df


def get_growing_season(run_date, country):
    # get the growing season complete with years
    if (country == "brazil") | (country == "argentina"):
        if run_date.month in [8, 9, 10, 11, 12]:
            growing_year = run_date.year
        else:
            growing_year = run_date.year - 1

        growing_season = pd.date_range(
            start=dt.datetime(growing_year, 8, 1),
            end=dt.datetime(growing_year + 1, 4, 1),
            freq="MS",
        )

    elif country == "usa":
        growing_year = run_date.year

        # u.s. doesn't straddle jan 1
        growing_season = pd.date_range(
            start=dt.datetime(growing_year, 4, 1),
            end=dt.datetime(growing_year, 11, 1),
            freq="MS",
        )

    return growing_season


def main(config, country, state, run_date):
    print(f"State: {state} // Run Date: {run_date:%Y-%m-%d}")
    ver = config["ver"]
    
    if config["prod_status"] == "test":
        dynamo_db_name = "insights-soy-fcst-test"
    else:
        dynamo_db_name = "insights-soy-fcst"

    run_month = run_date.month

    growing_season = get_growing_season(run_date, country)

    # months_passed INCLUDES THE CURRENT MONTH
    growth_date = run_date - dt.timedelta(days=1)

    months_passed = growing_season[
        growing_season <= dt.datetime(run_date.year, run_date.month, 1)
    ]

    soy_truth_df = sf.load_soy_truth(config, country, state)
    soy_truth_df = soy_truth_df.rename(columns={"yield": "prev_yield"})

    # get the trend, which is usually the stat to beat
    x_trend = sf.detrend_df(soy_truth_df, win=5, return_trend=True)
    x_trend = x_trend.rename(columns={"prev_yield_trend": "trend"})

    # load in the terraclimate vpd and soil moisture data
    x_vpd = load_realtime_tc(
        config, "vpd", state, country=country, months_list=months_passed
    )

    x_soil = load_realtime_tc(
        config, "soil", state, country=country, months_list=months_passed
    )

    # load in the monthly weather data (already observed)
    x_past_df = sf.load_tp(
        config, run_date, state, country=country, training=False
    ).loc[months_passed, :]

    # load in the weather data: constructed analog forecast data
    ca_fcst_df = load_ca_forecast(config, run_date, state, country)

    # we need to load in the training info so we can figure out
    # which features to use, and add bias
    bias_file = (
        f'{config["bucket_out_path"]}validation_{country}/v{ver}/{country}_{state}.csv'
    )

    biases_df = pd.read_csv(bias_file)
    biases_df = biases_df.set_index(biases_df.columns[0], drop=True)

    bias = biases_df.loc[run_month, "bias"]
    mape = 100 - biases_df.loc[run_month, "accuracy"]

    """
    # only for testing, load in the pickles
    ca_fcst_df = pd.read_pickle("ca_fcst_df.pickle")
    x_past_df = pd.read_pickle("x_past_df.pickle")
    """

    # add together past and future: ca_fcst_df + x_past_df (already been weighted)
    x_past_df.iloc[-1, :] = x_past_df.iloc[-1, :] + ca_fcst_df.iloc[0, :]
    tp_df = pd.concat([x_past_df, ca_fcst_df.iloc[1:, :]], axis=0)

    """
    Add ENSO, MJO, etc. data as predictors
    Especially important for Brazil,
    but also helpful for U.S.
    """
    enso_df = sf.load_enso(config)[-11:]

    if len(months_passed) == enso_df.index.isin(months_passed).sum():
        enso_df = enso_df.loc[months_passed, :]
    else:
        # at the first of the month we often don't have the most recent values
        # we can extract forward since ENSO values don't change dramatically w/time
        enso_df.loc[months_passed[1], :] = enso_df.loc[months_passed[0], :].values

    x_enso = enso_df.loc[enso_df.index.month.isin(months_passed.month), ["enso"]].T
    x_enso.columns = ["enso_{}".format(i) for i in x_enso.columns.month]
    x_enso = x_enso.reset_index(drop=True)

    mjo_df = sf.load_mjo(config)[-11:]

    if len(months_passed) == mjo_df.index.isin(months_passed).sum():
        mjo_df = mjo_df.loc[months_passed, :]
    else:
        # at the first of the month we often don't have the most recent values
        # we can extract forward since ENSO values don't change dramatically w/time
        mjo_df.loc[months_passed[1], :] = mjo_df.loc[months_passed[0], :].values

    x_mjo = mjo_df.loc[
        mjo_df.index.month.isin(months_passed.month),
        ["rmm1", "rmm2", "phase", "amplitude"],
    ]
    x_mjo = x_mjo.stack().reset_index(level=0).T
    new_cols = [
        "{}_{}".format(x_mjo.columns[i], int(x_mjo.iloc[0, i].month))
        for i in range(len(x_mjo.columns))
    ]
    x_mjo.columns = new_cols
    x_mjo = pd.DataFrame(x_mjo.iloc[1, :]).T

    # using the column names helps with following the features through the model
    # concat and drop years without data

    x_past = pd.DataFrame()

    for i in tp_df.index.month:
        temp_df = pd.DataFrame(
            tp_df.loc[tp_df.index.month == i, :].values,
            columns=[x + "_" + str(i) for x in tp_df.columns],
        )
        x_past = pd.concat([x_past, temp_df], axis=1)

    x_soy_truth = pd.DataFrame(soy_truth_df.iloc[-1, :]).T.reset_index(drop=True)
    x_yield_trend = pd.DataFrame(x_trend.iloc[-1, :]).T.reset_index(drop=True)[
        ["trend"]
    ]

    x_df = pd.concat(
        [x_vpd, x_soil, x_enso, x_mjo, x_past, x_soy_truth, x_yield_trend], axis=1
    )

    # load model
    model_file = f'simplecache::{config["bucket_out_path"]}trained_models/weather_model/v{ver}/{country}/{country}-{state}-{run_month}.joblib'
    model = sf.open_model(model_file)

    x_important = x_df.loc[:, model.feature_names_in_]

    y = model.predict(x_important).item()

    if (country == "argentina") | (state in ["PR", "RS", "GO", "MS", "MT", "DF"]):
        # run the crop model now
        crop_y = get_crop_data(config, state, run_date, y, country)

        # Load optimal windows, min splits are weights applied to CROP not wx model.
        min_splits = pd.read_pickle("local_stats/min_splits_brazil.pickle")
    else:
        # for some reason we don't have crop data for KY
        if state == "kentucky":
            crop_y = y
            wx_wt = 1
        else:
            crop_y = get_crop_data(config, state, run_date, y, country)

            # get the wts
            wx_wts = pd.read_pickle("local_stats/usa_wx_wts.pkl")
            wx_wt = wx_wts[
                (wx_wts["state"] == state.replace("-", ""))
                & (wx_wts["month"] == run_date.month)
            ]["wt"].item()

    y = crop_y * (1 - wx_wt) + y * wx_wt

    # calculate confidence interval
    # we need a better technique for this, perhaps some kind of bootstrapping?
    pred_low = y * (1 - mape / 100)
    pred_high = y * (1 + mape / 100)

    predictions = np.array([pred_low, y, pred_high])

    # what are the strongest features?
    main_feature = np.abs((model.feature_importances_ * x_important)).T.sort_values(
        by=0
    )[::-1]

    main_feature_index = main_feature.index
    main_feature_index = [i for i in main_feature_index if "yield" not in i]
    main_feature = main_feature_index[0]

    # print y
    print(f"Putting into dynamo {state}")
    boto_config = Config(connect_timeout=300, retries={"mode": "standard"})
    dynamo_client = boto3.resource(
        "dynamodb", region_name="us-west-2", config=boto_config
    )
    table = dynamo_client.Table(dynamo_db_name)
    try:
        table.put_item(
            Item={
                "country": f"{country}-{state}",
                "date": f"{run_date:%Y%m%d}",
                "state": state,
                "month": run_month,
                "year": run_date.year,
                "forecast": predictions.tostring(),
                "feature_values": x_important.values.astype("float").tostring(),
                "feature_names": model.feature_names_in_.tolist(),
                "main_feature": main_feature,
            }
        )
    except ClientError as err:
        print(err.response["Error"]["Message"])
        raise err

    print(f"Done with dynamo: {state}")

    return state


if __name__ == "__main__":
    logging.error("Don't run directly.")
