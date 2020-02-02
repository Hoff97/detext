NODE_JS_VERSION=10

bash -c "nvm use $NODE_JS_VERSION" || true

bash -c "source ~/.nvm/nvm.sh; nvm install $NODE_JS_VERSION; node --version"