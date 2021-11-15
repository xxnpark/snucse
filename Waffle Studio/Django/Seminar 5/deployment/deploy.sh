#!/bin/bash

source ~/.bash_profile

cd waffle-rookies-19.5-backend-2 || exit
git pull origin final

cd waffle_backend || exit
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install gunicorn
pip3 install django_extensions
pip3 install jupyter

./manage.py migrate
python3 ./manage.py check --deploy

./manage.py collectstatic

sudo pkill gunicorn
gunicorn waffle_backend.wsgi --bind 0.0.0.0:8000 --daemon

sudo nginx -t && sudo nginx -s reload