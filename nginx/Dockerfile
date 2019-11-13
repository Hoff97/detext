FROM nginx

RUN su -
RUN apt-get update

RUN apt-get -y install certbot python-certbot-nginx

RUN apt-get -y install cron

COPY nginx.conf /etc/nginx/

WORKDIR /code
COPY start.sh /code
COPY crontab /code

EXPOSE 80
EXPOSE 443

CMD ["sh", "start.sh"]
