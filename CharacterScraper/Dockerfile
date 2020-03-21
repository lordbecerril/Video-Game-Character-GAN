FROM python:3.6-alpine3.7

# update apk repo
RUN echo "http://dl-4.alpinelinux.org/alpine/v3.7/main" >> /etc/apk/repositories && \
    echo "http://dl-4.alpinelinux.org/alpine/v3.7/community" >> /etc/apk/repositories

# install chromedriver / base for pandas/numpy related libraries
RUN apk update && apk add chromium chromium-chromedriver && apk add build-base

# upgrade pip / install selenium
RUN pip install --upgrade pip
RUN pip install selenium
RUN pip install Cython
RUN pip install numpy
RUN pip install pandas
RUN pip install webdriver_manager
RUN pip install pyjwt
RUN pip install requests

WORKDIR /out

ADD riverdis.py /out
ADD lastdata.py /out
ADD config.py /out

RUN adduser -D -g '' independentus
USER independentus
