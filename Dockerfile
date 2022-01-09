FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    apt-get -y install gcc g++ \
    && sudo apt-get -y install libz-dev

# Create a working directory
RUN mkdir /app
WORKDIR /app

RUN conda install -y -c paperspace \
    jupyter \
    mock \
    certifi \
    configparser \
    enum34 \
    funcsigs \
    pathlib2 \
    pbr \
    scandir \
    singledispatch \
    webencodings

RUN git clone https://github.com/jerryji1993/DNABERT \
    && cd DNABERT \
    && python3 -m pip install --editable . \
    && cd examples \
    && python3 -m pip install -r requirements.txt

ADD . /app/nebis

RUN python3 -m pip install ./nebis \
    && python3 -m pip install -r ./nebis/requirements.txt

# Set the default command to python3
CMD ["python3"]