cd server
pip install -r requirements.txt

docker run -d --name tex2svg -p 0.0.0.0:9000:8000 hoff97/tex2svg

coverage run --source='.' manage.py test detext.server detext/tests

docker stop tex2svg