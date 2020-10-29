FROM python:3.7

RUN apt update -y && apt install -y build-essential && apt-get install -y build-essential && apt install -y libgl1-mesa-glx

RUN mkdir /app
WORKDIR /app
ADD . /app/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . app/
EXPOSE 12000

ENTRYPOINT ["python", "/app/app.py"]

