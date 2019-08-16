FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get clean && apt-get update && apt-get install -y locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV SHELL /bin/bash

RUN apt-get update && \
    apt-get install -y curl bzip2 xvfb ffmpeg git libxrender1

WORKDIR /aaio

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/bin/conda create -n python36 python=3.6 numpy

ENV PATH /opt/conda/envs/python36/bin:/opt/conda/envs/bin:$PATH

RUN pip install animalai

COPY agent.py /aaio/agent.py
COPY games_configurations.py /aaio/games_configurations.py
COPY models.py /aaio/models.py
COPY networks.py /aaio/networks.py
COPY tensorflow_utils.py /aaio/tensorflow_utils.py
COPY tf_moving_mean_std.py /aaio/tf_moving_mean_std.py
COPY tr_helpers.py /aaio/tr_helpers.py
COPY players.py /aaio/players.py

COPY requirements.txt /aaio/requirements.txt
COPY data /aaio/data

ENV HTTP_PROXY ""
ENV HTTPS_PROXY ""
ENV http_proxy ""
ENV https_proxy ""

ENV PYTHONPATH "${PYTHONPATH}:/aaio"


########################################################################################################################
# YOUR COMMANDS GO HERE

# For example, if your agent requires the animalai-train library
# you can add the following (remove if not needed):
#RUN pip install animalai-train
RUN pip install -r requirements.txt

########################################################################################################################

CMD ["/bin/bash"]

