curl -s \
  --form-string "token=" \
  --form-string "user=" \
  --form-string "message=${1}" \
  https://api.pushover.net/1/messages.json