#!/bin/bash

WEBHOOK=$1
MSG=$2

generate_post_data()
{
  cat <<EOF
{
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "$MSG"
            }
        }
    ]
}
EOF
}

curl --header "Content-type: application/json" \
     --request POST \
     --data "$(generate_post_data)" \
     $WEBHOOK