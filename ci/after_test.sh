bash <(curl -s https://codecov.io/bash)

cd server
COV=$(coverage report | grep TOTAL | grep -o '[^ ]*%')
STYLE=$(flake8 ./ | wc -l)

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