FROM python:3.9-slim

ENV ECCODES_VERSION 2.27.0

RUN echo 'deb http://deb.debian.org/debian bullseye-backports main' >> /etc/apt/sources.list \
    && apt-get update -y \
    && apt-get -y -t bullseye-backports install libaec-dev libsz2 \
    && apt-get install -y libhdf5-dev linux-libc-dev zlib1g-dev libc-dev-bin libc6-dev libjpeg-dev libjpeg62-turbo libjpeg62-turbo-dev libbsd-dev libgomp1 libquadmath0 gcc wget cmake git \
    && pip install cython \
    && pip install numpy \
    && cd /tmp \
    && wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.27.0-Source.tar.gz \
    && tar xvzf eccodes-${ECCODES_VERSION}-Source.tar.gz \
    && rm -f eccodes-${ECCODES_VERSION}-Source.tar.gz \
    && mkdir /tmp/build \
    && cd /tmp/build \
    && cmake -DENABLE_FORTRAN=OFF -DENABLE_PNG=OFF /tmp/eccodes-${ECCODES_VERSION}-Source \
    && make \
    && make install

ENV PYTHONPATH=/ul-soy-model
ENV AWS_STS_REGIONAL_ENDPOINTS='regional'
ENV AWS_DEFAULT_REGION='us-west-2'
COPY . ul-soy-model/.
WORKDIR /ul-soy-model
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel  
RUN pip install Cython
RUN pip install -r requirements.txt
RUN pip install openpyxl
RUN pip install descarteslabs
RUN pip install --upgrade numpy

RUN apt update
RUN apt-get -y install wget
RUN apt-get -y install curl
RUN wget https://github.com/jgm/pandoc/releases/download/3.1/pandoc-3.1-1-amd64.deb
RUN dpkg -i pandoc-3.1-1-amd64.deb

RUN apt-get install -y imagemagick
RUN apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra
RUN apt-get install -y awscli

RUN mkdir /tmp/models
RUN chmod a+x run_model.sh
RUN chmod a+x run_model_past.sh
RUN chmod a+x assemble_report.sh
RUN chmod a+x slack_md.sh

CMD ["./run_model.sh"] 