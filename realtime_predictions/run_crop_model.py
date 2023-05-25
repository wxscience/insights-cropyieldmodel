from json import load
import numpy as np

import pandas as pd
import xarray as xr
import sys
import s3fs
import boto3
import datetime as dt
import shapefile
import geopandas as gpd

# sklearn libraries
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error

# shared functions
import custom_libraries.shared_functions as sf


def update_table(country, state, run_date, val):
    # connect to db
    dynamo_client = boto3.resource(service_name="dynamodb", region_name="us-west-2")
    table = dynamo_client.Table("ul-soy-model")
    res = table.update_item(
        Key={
            "country": f"{country}-{state}",
            "date": f"{run_date:%Y%m%d}",
        },
        UpdateExpression=f"set crop_forecast = :val",
        ExpressionAttributeValues={":val": str(val)},
    )

    return res


def main(config, run_date, country, state, fcst_df, wx_fcst):
    if "fcst" in fcst_df.columns:
        fcst_df = fcst_df.rename({"fcst": "yield"}, axis=1)
        obs_df = fcst_df["yield"]
        fcst_df = fcst_df["yield"].reset_index()
        fcst_df = fcst_df.set_index("index", drop=True)
    else:
        obs_df = fcst_df["yield"]
        fcst_df = fcst_df["yield"].reset_index()

    soy_truth = sf.load_soy_truth(config, country, state, anomaly=False)

    ndvi_path = config["training"]["ndvi_features"][country]
    month_list = config["growing-season"][country]

    # set months, remember we skip first month since there's nothing to see
    cur_month = run_date.month
    prev_month = month_list[month_list.index(cur_month) - 1]

    # fcst_df has a col of predictions and a col of obs

    """ The code below is adapted from Rose's notebooks. """

    # GET FEATURES: archived + new data
    feats_files = []

    # the original file is the one for training that includes all past data
    feats_files.append(
        pd.read_pickle(f"{ndvi_path}{country}-ndvi.pickle").reset_index()
    )

    feat_df = feats_files[0]

    if country == "argentina":
        # argentina file doesn't have state data
        admin_data = gpd.read_file(
            f"{config['bucket_in_path']}external_data/argentina_admin.json"
        )[["name_1", "name_2"]]

        admin_data["name_2"] = admin_data["name_2"].str.lower().str.replace(" ", "")

        feat_df = feat_df.merge(admin_data, left_on="county", right_on="name_2")
        feat_df = feat_df.rename({"name_1": "state"}, axis=1)
        feat_df = feat_df.dropna()
        feat_df = feat_df[
            feat_df["state"].str.lower().str.replace(" ", "") == state.replace("-", "")
        ]

        feats_files = []
        feats_files.append(feat_df)

    # check the monthly_update directory. add files if there's anything new.
    # then we append to main csv and move the update
    s3 = s3fs.S3FileSystem()
    feats_dir = s3.ls(f"{ndvi_path}monthly_updates/{country}")

    for file in feats_dir:
        if file.endswith(".csv"):
            print(f"adding {file} to features")
            feats_files.append(pd.read_csv(f"s3://{file}"))

    # concatenate all the pandas files
    feats_df = pd.concat(feats_files)
    feats_df = feats_df.reset_index(drop=True)

    if feats_df.duplicated().sum() > 0:
        print("duplicates found")
        feats_df = feats_df.drop_duplicates()

    # fix the date column
    feats_df["date"] = pd.to_datetime(feats_df["date"])

    feats_df["month"] = feats_df["date"].dt.month
    feats_df["year"] = feats_df["date"].dt.year

    feats_df = feats_df.groupby(["county", "date"]).last().reset_index()

    # select only the current state, check against name and abbreviation
    feats_df = feats_df[
        (feats_df["state"] == state) | (feats_df["state"] == sf.get_abbrev(state))
    ]

    feats_df["obs_rate"] = feats_df["obs_rate"] / 100  # to match other metrics

    feats_df = feats_df.reset_index(drop=True)

    feats_df_month_current = feats_df.loc[
        (feats_df.month == cur_month) | (feats_df.month == prev_month), :
    ]

    # drop 2007 since we don't have the whole year
    feats_df_month_current = feats_df_month_current[feats_df_month_current.year > 2008]

    # separate out to make it easier to deal with
    feats_means_current = feats_df_month_current[["obs_rate", "mean", "month", "year"]]

    feats_means_current["mean"] = (
        feats_means_current["mean"] * feats_means_current["obs_rate"]
    )  # weight by obs rate

    feats_std_current = feats_df_month_current[["std", "month", "year"]]
    feats_min_current = feats_df_month_current[["min", "month", "year"]]
    feats_max_current = feats_df_month_current[["max", "month", "year"]]

    # assemble x
    temp_feats = feats_means_current.copy(deep=True)
    temp_feats_min = feats_min_current.copy(deep=True)
    temp_feats_max = feats_max_current.copy(deep=True)
    temp_feats_std = feats_std_current.copy(deep=True)

    x_temp = pd.DataFrame(
        index=temp_feats.year.unique(),
        columns=[
            "{}_{}".format(m, l)
            for l in ["obs_rate", "mean", "min", "max", "std"]
            for m in [prev_month, cur_month]
        ],
    )

    x_temp.iloc[:, 0:4] = (
        temp_feats.groupby(["year", "month"])
        .mean()
        .reset_index("month")
        .pivot(columns="month")
    ).fillna(method="ffill")

    x_temp.iloc[:, 4:6] = (
        temp_feats_min.groupby(["year", "month"])
        .min()
        .reset_index("month")
        .pivot(columns="month")
    ).fillna(method="ffill")

    x_temp.iloc[:, 6:8] = (
        temp_feats_max.groupby(["year", "month"])
        .max()
        .reset_index("month")
        .pivot(columns="month")
    ).fillna(method="ffill")

    # sqrt of sum of variances might be a meaningful quantity?
    x_temp.iloc[:, 8:] = (
        temp_feats_std.groupby(["year", "month"])["std"]
        .apply(lambda x: np.sqrt(np.sum(x**2)))
        .reset_index("month")
        .pivot(columns="month")
    ).fillna(method="ffill")

    x_train = x_temp.copy(deep=True)
    x_train = x_train.combine_first(fcst_df[["yield"]])
    x_train["yield"] = x_train["yield"].fillna(fcst_df["yield"].mean())

    # try out different regression types
    reg = RandomForestRegressor(n_estimators=1000)

    x_pred = x_train.loc[run_date.year]
    x_train.loc[run_date.year] = np.nan

    x_train = x_train.dropna()
    y_val = soy_truth.loc[soy_truth.index.isin(x_train.index), "yield"].values

    reg.fit(x_train.iloc[0 : len(y_val)], y_val)
    x_pred = x_temp.loc[run_date.year, :]
    x_pred["wx_preds"] = wx_fcst

    crop_forecast = reg.predict(x_pred.values.reshape(1, -1)).item()

    return crop_forecast


if __name__ == "__main__":
    raise NotImplementedError("use only as an imported function")
