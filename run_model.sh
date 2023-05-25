#!/bin/bash

SLACK_MSG_PRE='' # either set from tvp or...
DATE=`date -d "yesterday" +%Y-%m-%d` # 

message="${SLACK_MSG_PRE}*Model Run(SOY STEP 3a/4)* $DATE: running the model."
echo $message
/ul-soy-model/slack_md.sh $WEBHOOK "$message"

# call the script here
python main.py -country "usa" -run_mode "run" -run_date $(date +%F) -mp

if [ $? -ne 0 ]
then
    message="${SLACK_MSG_PRE}*Model Run (SOY STEP 3a/4)* $DATE: :no_entry_sign: "
    echo $message
    /ul-soy-model/slack_md.sh $WEBHOOK "$message"
    exit 40
else
    message="${SLACK_MSG_PRE}*Model Run (SOY STEP 3a/4)* $DATE: :white_check_mark: "
    echo $message
    /ul-soy-model/slack_md.sh $WEBHOOK "$message"
    exit 0
fi