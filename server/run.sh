./copy_static.sh
python /code/manage.py migrate --settings=detext.settings_prod
python /code/manage.py runserver --settings=detext.settings_prod 0.0.0.0:8000