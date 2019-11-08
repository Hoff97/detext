docker-machine create \
  --driver generic \
  --generic-ip-address=$DO_IP \
  --generic-ssh-user root \
  --generic-ssh-key ./ci/id_rsa_detext \
  detext

docker-machine ls