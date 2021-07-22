FROM python:3.8.11-slim-buster

WORKDIR /usr/app

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY main.py ./
COPY .env ./
CMD [ "python", "main.py"]