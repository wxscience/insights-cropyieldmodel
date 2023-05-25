from audioop import cross
from json import load
import numpy as np
import geopandas as gpd

import pandas as pd
import xarray as xr
import datetime as dt
import os
from dateutil.relativedelta import relativedelta

from jinja2 import Environment, FileSystemLoader, select_autoescape

# shared functions
import custom_libraries.shared_functions as sf

import matplotlib.pyplot as plt
import seaborn as sns


def get_strongest_feature(config, country, run_date):
    """
    This is the strongest feature present in the model training.
    NOT necessarily the driving feature for today's forecast.
    """
    ver = config["ver"]
    state_list = config["states"][country]

    df_features = pd.DataFrame(index=state_list)

    for state in state_list:
        prediction_file = f'{config["bucket_out_path"]}validation_{country}/v{ver}/{country}_{state}.csv'
        model_file = f'simplecache::{config["bucket_out_path"]}trained_models/weather_model/v{ver}/{country}/{country}-{state}-{run_date.month}.joblib'
        model = sf.open_model(model_file)

        df = pd.read_csv(prediction_file)
        df = df.set_index("month", drop=True)
        df = df.loc[run_date.month, "important_features"]

        # for some reason that important features list is really a string
        strongest_features = [
            x
            for x in df.split("'")
            if x not in [" ", ", ", "[", "]", ",", "prev_yield", "trend"]
        ]

        df_features.loc[state, "strongest_feature"] = strongest_features[0]
        df_features.loc[state, "strongest_feature_2"] = strongest_features[1]

    return df_features[["strongest_feature"]]


def create_yield_map(df, country, run_date):
    df["strongest_feature"] = [
        sf.friendly_features(i, run_date, map=True) for i in df["strongest_feature"]
    ]

    if country == "usa":
        # define a dictionary with state abbreviations
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

        df["abbrev"] = df.index.map(state_abbreviations)
        df["yield"] = [f"{np.round(x, 2)} bu/A" for x in df["yield"]]
        df["acres"] = [f"{np.round(x/1000,2)}k acres" for x in df["acres"]]

        gdf = gpd.GeoDataFrame(df)
        states_df = gpd.read_file("shapes/cb_2021_us_state_500k.shp").to_crs(
            "EPSG:3395"
        )
        states_df["state"] = [x.lower().replace(" ", "") for x in states_df["NAME"]]
        states_df = states_df.set_geometry("geometry")
        states_df = states_df.set_index("state", drop=True)

        gdf["geometry"] = states_df["geometry"]
        gdf_centers = gdf.copy()
        gdf_centers["center"] = gdf_centers["geometry"].centroid
        gdf_centers = gdf_centers.set_geometry("center")

    f_dict = {"strongest_feature": "Strongest Feature", "main_feature": "Main Feature"}
    for f in f_dict.keys():
        ax = gdf.plot(
            column=f,
            legend=True,
            cmap="Set3",
            edgecolor="k",
            figsize=(8, 8),
        )
        ax.legend_.set_title(f_dict[f], prop={"size": 12, "weight": "bold"})
        ax.legend_.set_frame_on(False)

        for x, y, label in zip(
            gdf_centers.geometry.x, gdf_centers.geometry.y, gdf_centers["abbrev"]
        ):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=10,
                ha="center",
                va="center",
                color="k",
                fontweight="bold",
            )

        for x, y, label in zip(
            gdf_centers.geometry.x, gdf_centers.geometry.y, gdf_centers["yield"]
        ):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, -5),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                va="center",
                color="k",
            )

        for x, y, label in zip(
            gdf_centers.geometry.x, gdf_centers.geometry.y, gdf_centers["acres"]
        ):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, -15),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                va="center",
                color="k",
            )
        plt.title(f"Forecast Run {run_date: %B %d, %Y}", fontsize=12)
        plt.suptitle("2023 Soybean Yield Forecast", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.axis("off")

        plt.savefig(f"yield_forecast_{f}.png", dpi=400)
        os.system(
            f"convert -resize 25% -trim yield_forecast_{f}.png -quality 100 yield_forecast_{f}.png"
        )

    return True


def create_plotable_df(fcst_df, bias_df, fcst_dates):
    # main feature is the one that's normally the strongest
    # let's find the actual strongest...
    # just the current date
    feat_df = pd.DataFrame(index=fcst_df.index)

    for state in fcst_df.index:
        # grab just current date
        f_df = fcst_df.copy(deep=True)
        f_df = f_df[f_df["date"] == fcst_dates[0]]

        f_names = f_df.loc[state, "feature_names"]
        f_values = f_df.loc[state, "feature_values"]

        # remove the trend feature and prev_yield
        f_values = list(np.delete(f_values, f_names.index("trend")))
        f_names = list(np.delete(f_names, f_names.index("trend")))

        f_values = np.delete(f_values, f_names.index("prev_yield"))
        f_names = np.delete(f_names, f_names.index("prev_yield"))

        f_values_sorted, f_names_sorted = zip(*sorted(zip(f_values, f_names)))

        f_sign = np.sign(f_values_sorted[np.abs(f_values_sorted).argmax()])
        f_name = f_names_sorted[np.abs(f_values_sorted).argmax()]

        feat_df.loc[state, "strongest_feature"] = f_name + "/" + str(f_sign)

    fcst_df = fcst_df.copy(deep=True)
    fcst_df["strongest_feature"] = feat_df["strongest_feature"]

    # go thru the list of dates
    df_list = []

    for d in fcst_dates:
        fcst_df_day = fcst_df[fcst_df["date"] == d]

        plot_df = pd.DataFrame(
            index=fcst_df_day.index,
            columns=[
                "date",
                "yield",
                "production_low",
                "production_mean",
                "production_high",
                "acres",
                "main_feature",
                "strongest_feature",
            ],
        )

        plot_df["date"] = d

        if fcst_df_day.empty:
            # if we don't have data, replace with nans
            plot_df = df_list[0].copy()
            plot_df.loc[:] = 0
        else:
            # add in the biases
            bias = bias_df.xs(dt.datetime.strptime(d, "%Y%m%d").month, level="month")
            for c in ["fcst_min", "fcst_mean", "fcst_max"]:
                fcst_df_day[c] = [
                    fcst_df_day.loc[s, c] - bias.loc[s].item()
                    for s in fcst_df_day.index
                ]

            plot_df["yield"] = fcst_df_day["fcst_mean"].astype(float).round(2)
            plot_df["production_low"] = (
                (fcst_df_day["fcst_min"] / 10**6 * fcst_df_day["acres"])
                .astype(float)
                .round(2)
            )
            plot_df["production_mean"] = (
                (fcst_df_day["fcst_mean"] / 10**6 * fcst_df_day["acres"])
                .astype(float)
                .round(2)
            )
            plot_df["production_high"] = (
                (fcst_df_day["fcst_max"] / 10**6 * fcst_df_day["acres"])
                .astype(float)
                .round(2)
            )
            plot_df["acres"] = np.round(fcst_df_day["acres"] / 10**3, 0).astype(int)
            plot_df["main_feature"] = fcst_df_day["main_feature"]
            plot_df["strongest_feature"] = fcst_df_day["strongest_feature"]

            plot_df.index = [
                x.replace("-", "") for x in plot_df.index
            ]  # markdown/latex can't handle hyphens in table

        df_list.append(plot_df)

    # df_list[1][:] = 0  # TEMPORARY

    return df_list


def main(config, country, run_date):
    # read in the latest acreage numbers
    if country == "usa":
        acres_df = pd.read_csv(
            "s3://insights-soyyieldmodel/external_data/usa_yield_production_23.csv"
        )
        bias = pd.read_pickle("usa_bias.pkl")
    elif country == "brazil":
        acres_df = np.nan
    elif country == "argentina":
        acres_df = np.nan

    # fix up acreage estimates so we can combine with fcst_df
    acres_df["State"] = [x.lower() for x in acres_df["State"]]
    acres_df["State"] = [x.replace(" ", "-") for x in acres_df["State"]]
    acres_df = acres_df.rename(columns={"State": "state"})
    acres_df = acres_df.set_index("state", drop=True)
    acres_df["acres"] = acres_df.iloc[:, -1]
    acres_df["acres"] = [float(x.replace(",", "")) for x in acres_df["acres"]]
    acres_sum = acres_df["acres"].sum()

    # today, last week, last month
    fcst_dates = [
        run_date.strftime("%Y%m%d"),
        (run_date - dt.timedelta(days=7)).strftime("%Y%m%d"),
        (run_date - relativedelta(months=1)).strftime("%Y%m%d"),
    ]

    print(f"Current Date: {fcst_dates[0]}")
    print(f"Prev Week: {fcst_dates[1]}")
    print(f"Prev Month: {fcst_dates[2]:}")

    # read in the latest forecast file
    # units are bu/ha
    fcst_df = sf.read_fcst_db(fcst_dates, country)
    fcst_df = fcst_df.set_index("state", drop=True)
    fcst_df["acres"] = acres_df["acres"]

    # Get model's strongest feature
    fcst_df["strongest_feature"] = get_strongest_feature(config, country, run_date)

    # Get friendly feature names
    fcst_df["strongest_feature"] = [
        sf.friendly_features(i, run_date) for i in fcst_df["strongest_feature"]
    ]

    fcst_df["main_feature"] = [
        sf.friendly_features(i, run_date) for i in fcst_df["main_feature"]
    ]

    # Create the plotable dataframe
    plot_df_list = create_plotable_df(fcst_df, bias, fcst_dates)

    #########
    # TEMPORARY
    #########
    """
    plot_df_list[1] = plot_df_list[0].copy()
    plot_df_list[1]["date"] = fcst_dates[1]
    plot_df_list[1][
        ["yield", "production_low", "production_mean", "production_high"]
    ] = 0  

    plot_df_list[2] = plot_df_list[0].copy()
    plot_df_list[2]["date"] = fcst_dates[2]
    plot_df_list[2][
        ["yield", "production_low", "production_mean", "production_high"]
    ] = 0
    """
    
    # plot map
    create_yield_map(plot_df_list[0].copy(), country, run_date)

    # write to a new file based on a template using Jinja
    jenv = Environment(
        loader=FileSystemLoader("realtime_predictions/templates/"),
        autoescape=select_autoescape(),
    )

    # UL FIRST
    template = jenv.get_template("soybean_us_template.md")
    fcst_dates = [
        dt.datetime.strptime(x, "%Y%m%d").strftime("%m-%d") for x in fcst_dates
    ]

    rendered_doc = template.render(
        {
            "production_sum_cur": np.round(
                plot_df_list[0].loc[:, "production_mean"].sum() / 10**6, 0
            ).astype("int"),
            "production_sum_prev_week": np.round(
                plot_df_list[1].loc[:, "production_mean"].sum() / 10**6, 0
            ).astype("int"),
            "production_sum_prev_month": np.round(
                plot_df_list[2].loc[:, "production_mean"].sum() / 10**6, 0
            ).astype("int"),
            "acres_sum": np.round(acres_sum / 10**6, 1),
            "yield_cur": dict(plot_df_list[0]["yield"]),
            "yield_prev_week": dict(plot_df_list[1]["yield"]),
            "yield_prev_month": dict(plot_df_list[2]["yield"]),
            "production_cur": dict(plot_df_list[0]["production_mean"]),
            "production_prev_week": dict(plot_df_list[1]["production_mean"]),
            "production_prev_month": dict(plot_df_list[2]["production_mean"]),
            "acres": dict(plot_df_list[0]["acres"]),
            "dates": dict(zip(["cur", "prev_week", "prev_month"], fcst_dates)),
            "INITIALIZED_DATE": run_date.strftime("%d %B %Y"),
            "national_prod": {
                "cur": np.round(plot_df_list[0].loc[:, "production_mean"].sum(), 2),
                "prev_week": np.round(
                    plot_df_list[1].loc[:, "production_mean"].sum(), 2
                ),
                "prev_month": np.round(
                    plot_df_list[2].loc[:, "production_mean"].sum(), 2
                ),
            },
            "national_yield": {
                "cur": np.round(
                    plot_df_list[0].loc[:, "production_mean"].sum()
                    / acres_sum
                    * 10**6,
                    2,
                ),
                "prev_week": np.round(
                    plot_df_list[1].loc[:, "production_mean"].sum()
                    / acres_sum
                    * 10**6,
                    2,
                ),
                "prev_month": np.round(
                    plot_df_list[2].loc[:, "production_mean"].sum()
                    / acres_sum
                    * 10**6,
                    2,
                ),
            },
            "external_national_yield": 49.5,
            "external_national_production": 49.5 * 86.3,
            "main_feats": dict(fcst_df["main_feature"]),
        }
    )

    # save rendered doc as HTML doc
    output_file = f"soy_{country}_forecast_{run_date:%Y%m%d}"
    with open(output_file + ".md", "w") as f:
        f.write(rendered_doc)

    # run pandoc
    os.system(
        f"pandoc -f markdown+raw_tex {output_file}.md -t latex -o {output_file}.pdf"
    )

    os.system(
        f"aws s3 cp {output_file}.pdf {config['reports_path']}{country}/{output_file}.pdf"
    )


if __name__ == "__main__":
    print("Don't run this script on its own. Use the main script.")
