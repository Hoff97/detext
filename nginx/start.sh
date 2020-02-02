if [ -e /etc/letsencrypt/live/detext.haskai.de/fullchain.pem ]
then
    echo "Certificate ok"
else
    certbot certonly --standalone --preferred-challenges http -d detext.haskai.de -m frithjof97@web.de --agree-tos
fi

certbot certonly --nginx -d detext.haskai.de -m frithjof97@web.de --agree-tos


/etc/init.d/cron start

crontab /code/crontab

nginx -g "daemon off;"