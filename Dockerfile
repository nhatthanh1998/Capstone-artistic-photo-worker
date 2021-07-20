FROM python:3

WORKDIR /usr/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY main.py ./
COPY .env ./
CMD [ "python", "main.py"]