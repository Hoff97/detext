docker push hoff97/detext-client
docker push hoff97/detext-nginx
docker push hoff97/detext-server

docker-machine ls

eval $(docker-machine env detext)
docker-compose -f ci/docker-compose-deploy.yml down
docker system prune -a
docker-compose -f ci/docker-compose-deploy.yml pull && docker-compose -f ci/docker-compose-deploy.yml up -d
