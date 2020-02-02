bash <(curl -s https://codecov.io/bash) -f client/coverage/cobertura-coverage.xml -f server/coverage.xml

cd server
COV=$(coverage report | grep TOTAL | grep -o '[^ ]*%')
STYLE=$(flake8 ./ | wc -l)

echo $COV
echo $STYLE

curl \
  --header "Authorization: Token $SERIES_CI_TOKEN" \
  --data value="$COV" \
  --data sha="$(git rev-parse HEAD)" \
  https://seriesci.com/api/repos/Hoff97/detext/cov/combined

curl \
  --header "Authorization: Token $SERIES_CI_TOKEN" \
  --data value="$STYLE" \
  --data sha="$(git rev-parse HEAD)" \
  https://seriesci.com/api/repos/Hoff97/detext/style/combined