import logging
import sys
import datetime as dt
import numpy as np
import boto3
import logging
import click

import training_scripts.train_model as train_wx
import realtime_predictions.run_wx_model as run_wx
import realtime_predictions.assemble_forecast as assemble_fcst

import custom_libraries.shared_functions as sf


def _init_logger():
    logger = logging.getLogger("insights-crop-model")
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
_logger = logging.getLogger("insights-crop-model.main")


# read in arguments from CLI
@click.command()
@click.option("-run_mode", default="train", help="train, run, assemble", type=str)
@click.option(
    "-country",
    default="brazil",
    help="countries supported: usa, brazil, argentina",
    type=str,
)
@click.option(
    "-run_date",
    default=dt.datetime.today(),
    help="date to run or assemble. ignored for training.",
    type=click.DateTime(),
)
@click.option(
    "-state",
    default="all",
    help="train single state or all",
    type=str,
)
@click.option(
    "-crop_type",
    default="corn",
    help="the name of the crop: e.g. corn, soy, etc.",
    type=str,
)
@click.option(
    "-mp/-no-mp",
    default=False,
    help="use true for mp when training and running",
    type=bool,
)
def process_arguments(run_mode, country, run_date, state, crop_type, mp):
    if run_mode not in ["train", "run", "assemble"]:
        _logger.warning(f"Mode is invalid. Defaulting to run. Given: {run_mode}")
        run_mode = "run"

    if country not in ["usa", "brazil", "argentina"]:
        _logger.warning(f"Country invalid. Defaulting to usa. Given: {country}")
        country = "usa"

    # print out given commands for posterity
    _logger.info(f"run_mode: {run_mode}")
    _logger.info(f"country: {country}")
    _logger.info(f"run_date: {run_date: %Y-%m-%d}")
    _logger.info(f"multiprocessing: {mp}")

    # load in config settings
    # each crop can have its own config file: crop_config.yml
    config = sf.load_settings(f"{crop_type}_config.yml")

    if state == "all":
        state_list = config["states"][country]
    else:
        state_list = [state]

    if run_mode == "train":
        print("Training")
        train_model(config, state_list, country, mp)
    elif run_mode == "run":
        run_model(config, state_list, country, run_date, mp)
    elif run_mode == "assemble":
        assemble_report(config, country, run_date)


def train_model(config, state_list, country, mp):
    if not mp:
        for state in state_list:
            train_wx.main(config, country, state)
    else:
        _logger.info("Submitting to AWS batch.")

        # submit to AWS batch
        job_response = sf.aws_batch("train", state_list, country)


def run_model(config, state_list, country, run_date, mp):
    # check if proper GEFS files are processed for date in question
    gefs_run_date = run_date - dt.timedelta(days=1)
    _logger.info(f"Checking GEFS for: {gefs_run_date: %Y-%m-%d}")

    key = f"gefs/{gefs_run_date:%Y%m%d}/gefs_depthBelowLandLayer_{gefs_run_date:%Y%m%d}.zarr"
    gefs_status = sf.check_file(config["gefs_bucket"].split("/")[2], key)

    if gefs_status == 404:
        raise FileNotFoundError(f"GEFS data for specified date not found. Try again.")

    if not mp:
        for state in state_list:
            run_wx.main(config, country, state, run_date)
    else:
        _logger.info("Submitting to AWS batch.")

        # submit to AWS batch
        job_response = sf.aws_batch("run", state_list, country, run_date)

    print("Done running model.")


def assemble_report(config, country, run_date):
    assemble_fcst.main(config, country, run_date)


def aws_batch(run_mode, config, state_list, country, run_date=False):
    print("AWS BATCH")
    return True


if __name__ == "__main__":
    process_arguments()
