# Setup

```
$ conda create -n detext python=3.7
$ source activate detext
(detext) $ pip install -r requirements.txt
(detext) $ conda install pygraphviz #Optional
```

# Running

```
(detext) $ python manage.py migrate
(detext) $ python manage.py createsuperuser --email adn@example.com --username admin
(detext) $ python manage.py runserver
```

Create visualization of model:
```
(detext) $ python manage.py graph_models server auth --pygraphviz --output res/model.dot
(detext) $ dot -Tpng res/model.dot -o res/model.png
```

Train (and save) model:
```
(detext) $ python manage.py runscript train
```

## ML

```
(detext) $ python src/train.py
```

## Tests

Run all tests with coverage
```
(detext) $ coverage run --source='.' manage.py test detext.server detext/tests
(detext) $ coverage html
```

Run a single test
```
(detext) $ python manage.py test detext.tests.test_ml
```