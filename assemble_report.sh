#!/bin/bash

SLACK_MSG_PRE='' # either set from tvp or...
DATE=`date -d "yesterday" +%Y-%m-%d` # 

cd /ul-soy-model

message="${SLACK_MSG_PRE}*Assemble Report (SOY STEP 3b/4)* $DATE: running the model."
echo $message
/ul-soy-model/slack_md.sh $WEBHOOK "$message"

# call the script here
python main.py -country "usa" -run_mode "assemble" -run_date $(date +%F) -no-mp

if [ $? -ne 0 ]
then
    message="${SLACK_MSG_PRE}*Assemble Report (SOY STEP 3b/4)* $DATE: :no_entry_sign: "
    echo $message
    /ul-soy-model/slack_md.sh $WEBHOOK "$message"
    exit 40
else
    message="${SLACK_MSG_PRE}*Assemble Report (SOY STEP 3b/4)* $DATE: :white_check_mark: "
    echo $message
    /ul-soy-model/slack_md.sh $WEBHOOK "$message"
    exit 0
fi