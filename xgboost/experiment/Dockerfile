# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
	 git-all \
	 make \
	 g++ \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN git clone --recursive https://github.com/dmlc/xgboost && \
      cd xgboost && \
      make -j4 && \
      cd python-package && \
      python setup.py install && \
      cd ../../ && \
      export PYTHONPATH=$PYTHONPATH:./xgboost/python-package
#RUN pip install numpy scipy scikit-learn flask gevent gunicorn && \
#        (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
#        rm -rf /root/.cache
#
## Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
## output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
## keeps Python from writing the .pyc files which are unnecessary in this case. We also update
## PATH so that the train and serve programs are found when the container is invoked.
#
#ENV PYTHONUNBUFFERED=TRUE
#ENV PYTHONDONTWRITEBYTECODE=TRUE
#ENV PATH="/opt/program:${PATH}"
#
## Set up the program in the image
#COPY xgboost /opt/program
#WORKDIR /opt/program

