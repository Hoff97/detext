cd server
pip install -r requirements.txt
coverage run --source='.' manage.py test detext.server detext/tests