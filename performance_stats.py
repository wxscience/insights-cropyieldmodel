import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import custom_libraries.shared_functions as sf

""" This is a pseudo notebook that analyzes performance metrics. """

# %%
config = sf.load_settings("config.yml")
ver = config["ver"]
country = "usa"
month_list = [4, 5, 6, 7, 8, 9, 10]
state_list = config["states"][country]
state_list = [i for i in state_list if i != "kentucky"]
df = []

for state in state_list:
    for month in month_list:
        prediction_file = f'{config["bucket_out_path"]}results_{country}/backtest_predictions/v{ver}/{country}_{state}_{month}.csv'
        predictions = df.append(pd.read_csv(prediction_file))

df = pd.concat(df)

# %%
# cheat with the months so they are in seasonal order
month_dict = dict(zip(month_list, np.arange(1, len(month_list) + 1)))

# national average, weighted by acreage since yield and prediction columns are already in prod units
df["production_obs"] = df["acreage"] / 10**6 * df["yield"]
df["production_fcst"] = df["acreage"] / 10**6 * df["prediction"]

# calculate the bias
bias_df = df[
    ["state", "month", "production_obs", "production_fcst", "year", "APE"]
].copy()
bias_df = bias_df[bias_df["year"] > 2010]
bias_df["diff"] = bias_df["production_fcst"] - bias_df["production_obs"]

bias = bias_df.groupby(["state", "month"]).mean()[["diff"]]
bias.to_pickle("usa_bias.pkl")

df = df.dropna().reset_index(drop=True)
df = df[["state", "year", "month", "production_obs", "production_fcst"]]

# sub bias
for row in range(len(df)):
    state = df.iloc[row, 0]
    month = df.iloc[row, 2]

    bias_val = bias.loc[(state, month), "diff"]
    df.iloc[row, -1] = df.iloc[row, -1] - bias_val

df_national = df.groupby(["year", "month"]).mean()
df_national = df_national[df_national.index.get_level_values("year") > 2012]

df_national["accuracy"] = (
    1
    - (
        np.abs(df_national["production_obs"] - df_national["production_fcst"])
        / df_national["production_obs"]
    )
) * 100

df_national = df_national.reset_index()[["year", "month", "accuracy"]]

cutoff = 2018
df_national_1 = df_national[df_national.year < cutoff]
df_national_2 = df_national[df_national.year >= cutoff]
# %%
plt.figure()
sns.set_theme(style="darkgrid")
sns.lineplot(
    data=df_national,
    x="month",
    y="accuracy",
    markers=True,
    estimator="mean",
    dashes=True,
    errorbar="se",
    err_style="band",
)

plt.ylim([80, 100])
plt.title("Soy Model Accuracy, 2013-2022")
# %%
plt.figure()
sns.set_theme(style="darkgrid")
sns.lineplot(
    data=df_national[df_national.month == 5],
    x="year",
    y="accuracy",
    markers=True,
    estimator="mean",
    dashes=True,
    errorbar="sd",
    err_style="band",
)

plt.ylim([80, 100])
plt.title("May Soy Model Accuracy, 2013-2022")
# %%
plt.figure()
sns.set_theme(style="darkgrid")
sns.lineplot(
    data=df_national,
    x="year",
    y="accuracy",
    markers=True,
    estimator="mean",
    dashes=True,
    errorbar="sd",
    err_style="band",
)

plt.ylim([50, 100])
plt.title("Soy Model Accuracy, 2013-2021")
