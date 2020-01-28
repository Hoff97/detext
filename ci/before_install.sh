eval "$(ssh-agent)"
openssl aes-256-cbc -K $encrypted_b25e5455c7d9_key -iv $encrypted_b25e5455c7d9_iv -in id_rsa_detext.enc -out ./ci/id_rsa_detext -d
chmod 600 ./ci/id_rsa_detext
ssh-add ./ci/id_rsa_detext
base=https://github.com/docker/machine/releases/download/v0.16.0 &&
  curl -L $base/docker-machine-$(uname -s)-$(uname -m) >/tmp/docker-machine &&
  sudo mv /tmp/docker-machine /usr/local/bin/docker-machine &&
  chmod +x /usr/local/bin/docker-machine