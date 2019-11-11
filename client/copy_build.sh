echo "Copying build"
mkdir -p  /usr/src/app/public
rm -rf /usr/src/app/public/*
cp -r /usr/src/app/client/ /usr/src/app/public
