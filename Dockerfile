# FROM python:3.6
# WORKDIR /app
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# # #### RUN apk add --no-cache gcc musl-dev linux-headers
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt
# EXPOSE 8050
# COPY . .
# CMD ["python","app.py"]

FROM python:3.6

USER root

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

EXPOSE 8001

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# ENV NAME World

CMD ["python", "app.py"]

