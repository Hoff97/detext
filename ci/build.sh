cd client

docker build -t hoff97/detext-client .
# docker push hoff97/detext-client

cd ../server

docker build -t hoff97/detext-server .
# docker push hoff97/detext-server

cd ../nginx

docker build -t hoff97/detext-nginx .
# docker push hoff97/detext-nginx