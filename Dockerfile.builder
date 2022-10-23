FROM cnstark/pytorch:1.12.1-py3.9.12-cuda11.6.2-ubuntu20.04

RUN apt-get -y update && \
    apt-get -y upgrade

ADD . /home/
WORKDIR /home/

RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

RUN apt-get install vim