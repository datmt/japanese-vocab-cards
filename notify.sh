#!/usr/bin/env bash
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"chat_id\": \"1024803686\", \"text\": \"$1\", \"disable_notification\": false}" \
  "https://api.telegram.org/botREDACTED_TELEGRAM_BOT_TOKEN/sendMessage" > /dev/null
