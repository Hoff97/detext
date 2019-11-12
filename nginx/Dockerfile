FROM nginx

RUN su -
RUN apt-get update

RUN apt-get -y install certbot python-certbot-nginx

COPY nginx.conf /etc/nginx/

WORKDIR /code
COPY start.sh /code

EXPOSE 80
EXPOSE 443

CMD ["sh", "start.sh"]
