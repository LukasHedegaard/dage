# load .env
if [ -f .env ]
then
  export $(cat .env | sed 's/#.*//g' | xargs)
fi

curl -s \
  --form-string "token=${NOTIFICATION_TOKEN}" \
  --form-string "user=${NOTIFICATION_USER}" \
  --form-string "message=${1}" \
  https://api.pushover.net/1/messages.json