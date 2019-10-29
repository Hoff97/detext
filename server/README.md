# Setup

```
$ conda create -n detext python=3.7
$ source activate detext
(detext) $ pip install -r requirements.txt
```

# Running

```
(detext) $ python manage.py migrate
(detext) $ python manage.py createsuperuser --email adn@example.com --username admin
(detext) $ python manage.py runserver
```