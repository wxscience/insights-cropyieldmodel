import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import sys
import wget
import shapefile
import geopandas as gpd
import rioxarray as rxr
import s3fs
import joblib
import fsspec
import boto3
import yaml
from boto3.dynamodb.conditions import Key
from shapely.ops import cascaded_union
import pickle
from dateutil.relativedelta import relativedelta

from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


def get_abbrev(state):
    state_fixed = state.replace("-", "").lower()

    state_abbreviations = {
        "alabama": "AL",
        "arkansas": "AR",
        "delaware": "DE",
        "florida": "FL",
        "georgia": "GA",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maryland": "MD",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "nebraska": "NE",
        "newjersey": "NJ",
        "newyork": "NY",
        "northcarolina": "NC",
        "northdakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "pennsylvania": "PA",
        "southcarolina": "SC",
        "southdakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "virginia": "VA",
        "west-virginia": "WV",
        "wisconsin": "WI",
    }

    return state_abbreviations[state_fixed]


def minmax_scaler(df, feature_range=(-1, 1)):
    # minmax scaler, preserves df column names
    df = df.copy()
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


def month_weight(d):
    # returns a weight defined as percentage of way thru current month
    d2 = (d + relativedelta(months=1)).replace(day=1) - dt.timedelta(days=1)

    wgt = d.day / d2.day

    return wgt


def read_pickle_s3(uri):
    # split up uri
    bucket = uri.split("/")[2]
    key = uri.split(f"{bucket}/")[1]

    session = boto3.Session()
    sts = session.client("sts")
    response = sts.assume_role(
        RoleArn="arn:aws:iam::954234516650:role/appsci-ro",
        RoleSessionName="ul-crop-model",
    )

    new_session = boto3.Session(
        aws_access_key_id=response["Credentials"]["AccessKeyId"],
        aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
        aws_session_token=response["Credentials"]["SessionToken"],
    )

    s3 = new_session.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    df = pickle.loads(obj)

    return df


def aws_batch(mode, state_list, country, run_date=dt.datetime.now()):
    prod = False
    batch = boto3.client("batch")

    # Define the "name" of the job queue and definition
    if prod:
        queue_name = (
            "arn:aws:batch:us-west-2:631799482519:job-queue/CropYieldModel-job-queue"
        )
        image = "631799482519.dkr.ecr.us-west-2.amazonaws.com/insights-cropyieldmodel:latest"
        job_name_str = "insights-crop-model-prod"
        job_definition_arn = "arn:aws:batch:us-west-2:631799482519:job-definition/CropYieldModel-job-def:3"
    else:
        queue_name = (
            "arn:aws:batch:us-west-2:631799482519:job-queue/CropYieldModel-job-queue"
        )
        image = "631799482519.dkr.ecr.us-west-2.amazonaws.com/insights-cropyieldmodel-dev:latest"
        job_name_str = "insights-crop-dev-"
        job_definition_arn = "arn:aws:batch:us-west-2:631799482519:job-definition/CropYieldModel-job-def:2"

    jobs = []

    for state in state_list:
        if mode == "train":
            pass_cmd = {
                "command": [
                    "python",
                    "main.py",
                    "-run_mode",
                    "train",
                    "-country",
                    country,
                    "-state",
                    state,
                    "-no-mp",
                ]
            }
        elif mode == "run":
            pass_cmd = {
                "command": [
                    "python",
                    "main.py",
                    "-run_mode",
                    "run",
                    "-country",
                    country,
                    "-state",
                    state,
                    "-run_date",
                    run_date.strftime("%Y-%m-%d"),
                    "-no-mp",
                ]
            }

        response = batch.submit_job(
            jobName=f"{job_name_str}{state}",
            jobQueue=queue_name,
            jobDefinition=job_definition_arn,
            containerOverrides=pass_cmd,
        )

        jobs.append(response)

    return jobs


def load_settings(filename):
    # This function loads settings from the config.yml file
    with open(filename, "r") as config_file:
        try:
            config_file = yaml.safe_load(config_file)
        except ImportError:
            print("The file could not be loaded.")

    return config_file


def friendly_features(f, rd, map=False):
    import string

    if map:
        # separate the feature and its sign
        f_sign = f.split("/")[1]

        if float(f_sign) < 0:
            part3 = "-"
        else:
            part3 = "+"

        f = f.split("/")[0]

    if (f != "prediction") & (f != "trend"):
        # some features have 3 parts (e.g. precip_past_10), but most have 2 (e.g. precip_1)
        part1 = f.split("_")[0]
        if "rmm" in part1:
            part1 = "MJO"
        elif "enso" in part1:
            part1 = "ENSO"
        elif "vpd" in part1:
            part1 = "VPD"
        else:
            part1 = part1.title()

        part2 = int(f.split("_")[-1])

        if part2 <= 3:
            f_month = part2 + rd.month
            if f_month > 12:
                f_month = f_month - 12
            part2 = str(f_month)
        else:
            part2 = str(part2)
    else:
        if f == "trend":
            f_friendly = "5-yr Trend"
            return f_friendly
        else:
            part1 = "MJO"
            part2 = str(rd.month)
    if not map:
        f_friendly = f"{part1}~{part2}~"
    else:
        # month number to name
        month_dict = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }
        f_friendly = f"{month_dict[int(part2)]} {part1}{part3}"
    return f_friendly


def open_model(f):
    if "s3://" in f:
        file = fsspec.open_local(f, filecache={"cache_storage": "temp_files/"})
        model = joblib.load(file)
    else:
        model = joblib.load(file)

    return model


def save_model(model, f):
    if "s3://" in f:
        s3_fs = s3fs.S3FileSystem()

        with s3_fs.open(f, "wb") as f_obj:
            joblib.dump(model, f_obj)

    else:
        joblib.dump(model, f)

    return True


def save_df_s3(df, f):
    from pickle import dump
    import io

    buffer = io.BytesIO()
    dump(df, buffer)
    buffer.seek(0)
    print(buffer.closed)

    if "s3://" in f:
        s3_fs = s3fs.S3FileSystem()

        obj = s3_fs.Object(s3bucket, f"{s3_upload_path}/{filename}")
        obj.put(Body=buffer.getvalue())

    else:
        raise NotImplementedError("Can only use s3 here.")

    return True


def open_xr(f, decode=False):
    # Loads netcdf, checking to see which platform we're on

    # if we are on AWS
    if "s3://" in f:
        s3_fs = s3fs.S3FileSystem()

        with s3_fs.open(f) as f_obj:
            if decode:
                ds = xr.load_dataset(f_obj, engine="h5netcdf", decode_coords="all")
            else:
                ds = xr.load_dataset(f_obj, engine="h5netcdf")
    else:
        if decode:
            ds = xr.load_dataset(f, engine="netcdf4", decode_coords="all")
        else:
            ds = xr.load_dataset(f, engine="netcdf4")

    # we want all lat/lon coordinates to be labelled as such
    if "latitude" in ds._coord_names:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    return ds


def load_tp(config, run_date, state, country, training=False):
    """
    file1 has data 1979-2022 for training and climatology calculations
    file3 adds in the latest data from GEFS for realtime usage since the model needs
    to know what happened during the entire season to date
    """
    file1 = config["training"]["tp_data"]["nc_archive"]

    # zarr is way faster and better with large files on S3
    tp1 = xr.open_dataset(file1, engine="zarr")

    # tp1 is a large file. We can quickly chop off half the globe to keep memory usage low.
    if country == "usa":
        tp1 = tp1.where(tp1.lat > 0, drop=True)
        tp1 = tp1.where(tp1.lon > 180, drop=True)
    else:
        tp1 = tp1.where(tp1.lat < 0, drop=True)

    climo = tp1.groupby(tp1.time.dt.month).median()

    if training:
        # mask to correct state, cut to correct years
        tp_df_masked = (
            mask_data(tp1, state, country)
            .mean(["lat", "lon"])
            .to_dataframe()[["precip", "tmin", "tmax", "frost"]]
        )

    if not training:
        # get the most recent months using the GFS analysis fields
        # the nc data ends in Dec 2022, so we'll use GFS to fill in from then
        gefs_date_list = pd.date_range(
            start=dt.datetime(2023, 1, 1), end=run_date - dt.timedelta(days=1), freq="D"
        )

        # It's a lot faster, but more combersome, to do one variable at a time so we can use dataarrays instead of datasets
        gefs_tmin = (
            load_gefs_archive(
                config["realtime"]["gefs_path"],
                gefs_date_list,
                "tmin",
                "heightAboveGround",
            )
            - 273.15
        )

        gefs_tmax = (
            load_gefs_archive(
                config["realtime"]["gefs_path"],
                gefs_date_list,
                "tmax",
                "heightAboveGround",
            )
            - 273.15
        )

        gefs_precip = load_gefs_archive(
            config["realtime"]["gefs_path"], gefs_date_list, "tp", "surface"
        )

        gefs_frost = load_gefs_archive(
            config["realtime"]["gefs_path"],
            gefs_date_list,
            "frost",
            "heightAboveGround",
        )

        tp2 = xr.merge([gefs_precip, gefs_tmax, gefs_tmin, gefs_frost])

        # get tp2 to same lat/lon grid as tp1
        tp2 = tp2.interp(lat=tp1.lat, lon=tp1.lon)
        tp = xr.merge([tp1, tp2])

        # mask to correct state, cut to correct years
        tp_df_masked = (
            mask_data(tp, state, country)
            .mean(["lat", "lon"])
            .to_dataframe()[["precip", "tmin", "tmax", "frost"]]
        )

    # mask tp1 to get climos
    climo = tp1.groupby(tp1.time.dt.month).median()

    climo_masked = (
        mask_data(climo, state, country)
        .mean(["lat", "lon"])
        .to_dataframe()[["precip", "tmin", "tmax", "frost"]]
    )

    # create anomalies
    df = pd.DataFrame(
        [
            tp_df_masked.iloc[x, :]
            - climo_masked.iloc[tp_df_masked.index[x].month - 1, :]
            for x in range(len(tp_df_masked))
        ],
        index=tp_df_masked.index,
        columns=tp_df_masked.columns,
    )[["precip", "tmin", "tmax", "frost"]].dropna()

    if not training:
        # correctly weight the most current forecast so we can easily combine later (temps only)
        df.iloc[-1, [1, 2]] = df.iloc[-1, [1, 2]] * month_weight(run_date)

    return df


def read_fcst_db(fcst_dates, country):
    dynamo_client = boto3.client("dynamodb", region_name="us-west-2")

    response = dynamo_client.scan(
        TableName="insights-crop-fcst",
        FilterExpression="contains(country, :country) and contains(#datestr, :year)",
        ExpressionAttributeNames={
            "#datestr": "date",
        },
        ExpressionAttributeValues={
            ":country": {"S": country},
            ":year": {"S": fcst_dates[0][0:4]},
        },
    )

    fcst_db = response["Items"]

    while "LastEvaluatedKey" in response:
        response = dynamo_client.scan(
            TableName="insights-crop-fcst",
            FilterExpression="contains(country, :country) and contains(#datestr, :year)",
            ExpressionAttributeNames={
                "#datestr": "date",
            },
            ExpressionAttributeValues={
                ":country": {"S": country},
                ":year": {"S": fcst_dates[0][0:4]},
            },
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )

        fcst_db.extend(response["Items"])

    # We also want to get top features from each forecast
    fcst_df = pd.DataFrame(
        columns=[
            "country",
            "date",
            "state",
            "month",
            "fcst_min",
            "fcst_max",
            "fcst_mean",
            "feature_values",
            "feature_names",
            "main_feature",
        ]
    )

    counter = 0
    # convert to df, gotta convert binary back too
    for i in fcst_db:
        fcst_df.loc[counter, "country"] = country
        fcst_df.loc[counter, "date"] = i["date"]["S"]
        fcst_df.loc[counter, "state"] = i["state"]["S"]
        fcst_df.loc[counter, "month"] = i["month"]["N"]
        fcst_df.loc[counter, "year"] = i["year"]["N"]
        fcst_df.loc[counter, "fcst_mean"] = np.frombuffer(i["forecast"]["B"])[1]
        fcst_df.loc[counter, "fcst_min"] = np.frombuffer(i["forecast"]["B"])[0]
        fcst_df.loc[counter, "fcst_max"] = np.frombuffer(i["forecast"]["B"])[2]

        feature_names = []
        for j in i["feature_names"]["L"]:
            feature_names.append(j["S"])

        fcst_df.at[counter, "feature_names"] = feature_names
        fcst_df.at[counter, "feature_values"] = np.frombuffer(i["feature_values"]["B"])
        fcst_df.loc[counter, "main_feature"] = i["main_feature"]["S"]

        counter += 1

    fcst_df = fcst_df.sort_values(by="state").reset_index(drop=True)

    # return the dates
    fcst_df = fcst_df[fcst_df["date"].isin(fcst_dates)]

    return fcst_df


def load_crop_truth_usda(config, state, month):
    """
    PCT Planted: Apr-Jun
    PCT Emerged: May-Jun
    PCT Blooming: Jun-Sep
    """

    # This loads the crop truth data from the USDA API.
    truth_file = f'{config["bucket_in_path"]}external_data/crop_progress_usa.csv'

    df = pd.read_csv(truth_file)

    # convert column to pd date time
    df["Week"] = pd.to_datetime(df["Week Ending"])
    df = df[df.State == state.replace("-", " ").upper()]
    df = df.set_index("Week", drop=True).sort_index()
    df = df[["Data Item", "Value"]]

    return df


def load_crop_truth(config, country, state, anomaly=False):
    # This needs to be changed for each country and each crop
    if country == "usa":
        truth_file = f'{config["bucket_in_path"]}external_data/{config["training"]["truth_file"][country]}'

        # This script parses the USDA cropbean truth file and returns the necessary values.
        # load in the annual data, get rid of commas, convert to numbers

        df = pd.read_csv(truth_file)
        df["State"] = [x.lower().replace(" ", "-") for x in df["State"]]
        df = df.loc[df["State"] == state, :]

        df = df[(df.Program == "SURVEY") & (df.Period == "YEAR")]

        new_df = pd.DataFrame(
            index=df.Year.unique(), columns=["yield", "area_harvested"]
        )
        new_df["yield"] = df.loc[
            df["Data Item"] == "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value"
        ].values
        new_df["area_harvested"] = df.loc[
            df["Data Item"] == "CORN, GRAIN - ACRES HARVESTED", "Value"
        ].values

        df = new_df.copy()
        df = df.sort_index(ascending=True)

        df["yield"] = pd.to_numeric(df["yield"])
        df["area_harvested"] = pd.to_numeric(df["area_harvested"].str.replace(",", ""))

        return df


def mask_data(ds, state, country):
    state_shapes = shapefile.Reader("shapes/ne_10m_admin_1_states_provinces")
    gdf = gpd.GeoDataFrame.from_features(state_shapes)

    """
    in the US and Argentina, states are identified as one word (w/hypen if 2)
    in brazil, states are identified by two letter state codes
    """
    # need to remove hyphens; two word states are listed as such
    if "-" in state:
        state = state.replace("-", " ")

    if country == "usa":
        country_full_name = "United States"
        gdf = gdf[
            (gdf.admin.str.contains(country_full_name)) & (gdf.name == state.title())
        ]
    elif country == "brazil":
        country_full_name = "Brazil"
        gdf = gdf[(gdf.admin.str.contains("Brazil")) & (gdf.postal.str.contains(state))]
    elif country == "argentina":
        country_full_name = "Argentina"
        gdf = gdf[
            (gdf.admin.str.contains(country_full_name)) & (gdf.name == state.title())
        ]

    # load data and manipulate to match coordinates in shapefile
    ds = ds.rio.write_crs("WGS84")

    # some datasets are -180 to 180, others are 0 to 360.
    if ds.lon.data.max() > 180:
        # this means we are 0 to 360 and need to convert to -180 to 180
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
        ds = ds.rio.set_spatial_dims("lon", "lat")

    # some states are too small to have gridpoints
    try:
        # the small states are too small to have gridpoints
        small_states = {
            "florida": [-88.417969, 24.567108, -78.244629, 34.288992],
            "delaware": [-77.618408, 37.561997, -74.619141, 40.321420],
            "new-jersey": [-76.684570, 38.186387, -72.641602, 41.343825],
        }

        if state in small_states.keys():
            print("small state, using box")
            ds_masked = ds.rio.clip_box(*small_states[state])
        else:
            ds_masked = ds.rio.clip(gdf.geometry, all_touched=True)
    except:
        print(f"State {state} was too small. Defaulting to bigger shape.")
        gdf = gpd.GeoDataFrame.from_features(state_shapes)
        ds_masked = ds.rio.clip(cascaded_union(gdf.geometry))

    return ds_masked


def load_terraclimate(
    config,
    var,
    state,
    years=np.arange(1990, 2022),
    country="usa",
    return_climo=False,
):
    # nc files are clipped to state and saved as pickle for future access
    nc_path = config["training"]["terraclimate"]["nc_path"]
    pickle_path = config["training"]["terraclimate"]["pickle_path"]

    # uses terraclimate netcdf files (see shell script to download them)
    df_path = f"{pickle_path}{var}_ts/{state}.pickle"

    # check if pickle file exists
    try:
        df = pd.read_pickle(df_path + "1")
    except:
        print("NEED TO GENERATE VPD/SOIL SHAPE DATA")
        df = pd.DataFrame(columns=[var])

        for year in years:
            f = f"{nc_path}TerraClimate_{var}_{year}.nc"
            print(f"Loading file: {f}")
            ds = open_xr(f, decode=True)
            print("Opened ds: ")

            ds = ds.drop("crs")
            ds_masked = mask_data(ds, state, country)
            print("Masked.")

            df_mean = ds_masked.mean(["lat", "lon"]).to_dataframe()
            df = pd.concat([df, df_mean], ignore_index=False)
            print(year)

        df = df[[var]]
        df.to_pickle(df_path)

    # create anomalies
    climo = df.groupby(df.index.month).mean()
    df = pd.DataFrame(
        [
            df.iloc[x, 0] - climo.iloc[df.index[x].month - 1, 0].item()
            for x in range(len(df))
        ],
        index=df.index,
        columns=[var],
    )

    if return_climo:
        return climo
    else:
        return df


def load_ca(config, state, country="usa"):
    # this function loads the constructed analog forecasts

    # the brazil and argentina files are the same, though called "Brazil"
    ca_file = config["training"]["constructed_analog"][country]

    ds = xr.open_zarr(ca_file)

    # set the time dims since this file doesn't have any
    if country == "usa":
        ds["time"] = pd.date_range("1990-01-01", "2021-12-31", freq="MS")
    elif (country == "brazil") | (country == "argentina"):
        ds["time"] = pd.date_range("1991-01-01", "2021-12-31", freq="MS")

    # we just want the first 3 months forecast (months 1-3)
    ds = ds.isel(fcst_time=[1, 2, 3])
    ds = ds.assign({"fcst_time": [1, 2, 3]})

    # interested in planting and mid-season: sep through mar
    # use all months. bad forecasts will end up weighted less.
    ds_masked = mask_data(ds, state, country)
    df_out = (
        ds_masked.mean(["lat", "lon"]).to_dataframe().loc[:, ["precip", "tmax", "tmin"]]
    )

    # convert from multiindex to single index with named cols
    df_out = df_out.unstack()

    df_out.columns = [
        "{}_f{}".format(x[0], x[1]) for x in df_out.columns
    ]  # do this b/c column names must be str

    return df_out


def load_enso(config):
    enso_file = config["data-paths"]["enso"]

    df = read_pickle_s3(enso_file)
    df = df.resample("MS").first()

    df = df.rename({"anomalies": "enso"}, axis=1)

    return df


def load_mjo(config):
    mjo_file = config["data-paths"]["mjo"]

    df = read_pickle_s3(mjo_file)
    df = df.resample("MS").first()

    return df


def detrend_df(y_df, win=5, return_trend=False):
    trend_df = y_df.copy(deep=True)
    # define trend as mean over last 5 years b/c that's what UL wants
    for i in range(win, len(y_df)):
        trend_df.iloc[i] = (y_df.iloc[i - win : i]).mean()

    # quick backfill
    trend_df.iloc[0:win] = trend_df.iloc[win]

    # name columns with _trend
    trend_df.columns = ["{}_trend".format(x) for x in trend_df.columns]

    # detrend
    detrended = [y_df.iloc[i] - trend_df.iloc[i] for i in range(len(trend_df))]

    if return_trend:
        # return the trend
        return trend_df
    else:
        return pd.DataFrame(detrended, index=y_df.index)


def quick_pca(df, num_pcs=10):
    # new shape = [sample x feature]
    pca = PCA(n_components=num_pcs)
    pca.fit(df)
    df_pca = pd.DataFrame(pca.transform(df), index=df.index)

    return pca, df_pca


def load_gefs_archive(gefs_path_base, date_list, var, var_type, country="usa"):
    print(f"Starting GEFS archive function: {var}//{var_type}")
    # var_type is either heightAboveGround or surface
    ds = []

    if var == "frost":
        var_key = "tmin"
    else:
        var_key = var

    for n_date in date_list:
        gefs_path = f"{gefs_path_base}{n_date:%Y%m%d}/gefs_"

        ds_single = xr.open_dataset(
            f"{gefs_path}{var_type}_{n_date:%Y%m%d}.zarr",
            engine="zarr",
            decode_times=True,
        )[var_key][0:4, :, :]

        if var == "tp":
            ds_single = ds_single.sum("step")
        elif var == "frost":
            ds_single = ds_single.min("step")
        else:
            ds_single = ds_single.mean("step")

        if country == "usa":
            ds_single = ds_single.where(ds_single.latitude > 0, drop=True)
            ds_single = ds_single.where(ds_single.longitude > 180, drop=True)
        else:
            ds_single = ds_single.where(ds_single.latitude < 0, drop=True)

        ds.append(ds_single)

    ds = xr.concat(ds, "time")

    if var == "tp":
        ds = ds.groupby(ds.time.dt.month).sum()
        ds = ds.rename("precip")
    elif var == "frost":
        ds = (ds - 273.15 - 3).groupby(ds.time.dt.month).sum()
        ds = ds.rename("frost")
    else:
        ds = ds.groupby(ds.time.dt.month).mean()

    # relabel the new month time so that it's time
    ds["month"] = date_list[~date_list.to_period("m").duplicated()]
    ds = ds.drop_vars(var_type)

    ds = ds.rename({"latitude": "lat", "longitude": "lon", "month": "time"})

    return ds


def load_gefs(config, run_date, state, country):
    gefs_path_base = config["realtime"]["gefs_path"]
    climo_file = config["tp_climo"]

    # GEFS 30 day forecast only comes out on 00z run
    # and it's always a day late
    # we want f000 to f840 by 6 hr intervals
    gefs_path = f"{gefs_path_base}{run_date:%Y%m%d}/gefs_"

    ds_temps = xr.open_zarr(
        f"{gefs_path}heightAboveGround_{run_date:%Y%m%d}.zarr", decode_times=True
    )[["tmax", "tmin"]]

    ds_precip = xr.open_zarr(
        f"{gefs_path}surface_{run_date:%Y%m%d}.zarr", decode_times=True
    )

    # rename to match all other grids
    ds_temps = (
        ds_temps.rename({"latitude": "lat", "longitude": "lon"}) - 273.15
    )  # k to c
    ds_precip = ds_precip.rename({"latitude": "lat", "longitude": "lon"})  # in mm

    # load climo
    climo = xr.open_zarr(climo_file)  # in c and mm

    # need to mask climo separately because of the precip sum problem
    masked_climo_temps = (
        mask_data(climo[["tmax", "tmin", "frost"]], state, country)
        .mean(["lat", "lon"])
        .to_dataframe()
    )
    masked_climo_precip = (
        mask_data(climo[["precip"]], state, country)
        .mean(["lat", "lon"])
        .to_dataframe()
        .rename({"precip": "tp"}, axis=1)
    )

    masked_temps = mask_data(ds_temps, state, country)
    masked_precip = mask_data(ds_precip, state, country)

    """ group into month and day """
    masked_temps["step"] = masked_temps.step.valid_time
    masked_precip["step"] = masked_precip.step.valid_time

    masked_frost = masked_temps[["tmin"]].copy(deep=True)
    masked_frost = masked_frost.rename({"tmin": "frost"})
    masked_frost = masked_frost.resample(step="1D").min()
    masked_frost = (masked_frost - 3).mean(["lat", "lon"]).to_dataframe()[["frost"]]

    masked_temps = masked_temps.resample(step="1D").mean()
    masked_precip = masked_precip.resample(step="1D").sum()

    # monthly averages
    masked_temps = (
        masked_temps.groupby(masked_temps.step.dt.month)
        .mean()
        .mean(["lat", "lon"])
        .to_dataframe()[["tmax", "tmin"]]
    )

    masked_frost = masked_frost.groupby(masked_frost.index.month).sum()

    masked_precip = (
        masked_precip.groupby(masked_precip.step.dt.month)
        .sum()
        .mean(["lat", "lon"])
        .to_dataframe()[["tp"]]
    )

    # create anomalies
    masked_temps = (
        masked_temps - masked_climo_temps.loc[masked_temps.index, masked_temps.columns]
    )
    masked_frost = (
        masked_frost - masked_climo_temps.loc[masked_frost.index, masked_frost.columns]
    )
    masked_precip = (
        masked_precip
        - masked_climo_precip.loc[masked_precip.index, masked_precip.columns]
    )

    wgt = month_weight(run_date)

    # weight the averages (not the sums!)
    masked_temps.iloc[0, :] = masked_temps.iloc[0, :] * (1 - wgt)
    masked_temps.iloc[1, :] = masked_temps.iloc[1, :] * wgt

    # combine them all into one big df
    gefs = pd.concat([masked_temps, masked_frost, masked_precip], axis=1)

    # rename precip to match ca
    gefs = gefs.rename({"tp": "precip"}, axis=1)

    return gefs


def check_file(bucket, key):
    """return the key's size if it exist, else None"""
    s3client = boto3.client("s3")
    response = s3client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
    )
    if len(response.get("Contents", [])) == 0:
        return 404
    else:
        return True
