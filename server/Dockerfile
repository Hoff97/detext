# Pull base image
FROM python:3.7-slim

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1

# Install dependencies
COPY requirements.txt /code/
RUN pip install -r requirements.txt

RUN apt install libxext6

# Copy project
COPY . /code/

# Collect static files
RUN python manage.py collectstatic --settings=detext.settings_prod

CMD ["bash", "run.sh"]