FROM nvidia/cuda:10.0-base-ubuntu18.04

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

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.7
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python==3.7 \
    && conda clean -ya

# CUDA 10.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.3.1=py3.7_cuda10.0.130_cudnn7.6.3_0" \
    "torchvision=0.4.2=py37_cu100" \
    && conda clean -ya

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