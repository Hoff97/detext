[![Build Status](https://travis-ci.com/Hoff97/detext.svg?branch=develop)](https://travis-ci.com/Hoff97/detext) [![codecov](https://codecov.io/gh/Hoff97/detext/branch/develop/graph/badge.svg)](https://codecov.io/gh/Hoff97/detext)

# Whats this?

This is a simple app for detecting and classifying math symbols. It displays the class of the input of the user
and further information (like the latex code) for it.

Its run at https://detext.haskai.de

# Running

Start the server
```
 $ cd server
 $ pip install -r requirements.txt
 $ python manage.py migrate        #This can take a while on the first run
 $ python manage.py createsuperuser --email admin@example.com --username admin
 $ python manage.py runserver
```

Then start the client:
```
 $ cd client
 $ npm install
 $ npm run dev
```

And tex2svg (creates SVG's from latex codes)

```
 $ docker run -p 0.0.0.0:9000:8000 hoff97/tex2svg
```

Alternatively, you can run

```
 $ docker-compose -f docker-compose-dev.yml up
```
