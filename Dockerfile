FROM ubuntu:20.04
RUN apt update -y
RUN apt install -y git

# python
RUN apt install -y gcc libpq-dev
RUN apt install -y python3-dev python3-pip python3-venv python3-wheel
RUN ln -fs /usr/bin/python3 /usr/bin/python
RUN python -m pip install -U pip wheel
RUN echo 'alias venv="python -m venv"' >> ~/.bashrc

# spark
RUN apt install --no-install-recommends -y openjdk-11-jdk-headless ca-certificates-java
RUN pip install pyspark==3.2.1
RUN pip install pandas pyspark[sql]
ENV PYSPARK_PYTHON=/usr/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python
ENV JAVA_HOME=/usr/lib/jvm/default-java

# install requirements
COPY ./requirements.txt ./
COPY ./requirements-dev.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt
