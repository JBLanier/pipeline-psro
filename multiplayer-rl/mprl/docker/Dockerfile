FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
	apt-get install -y \
	apt-utils \
	build-essential \
	git \
	wget \
	unzip \
	tmux \
	python3-dev \
	python3-pip \
	python3-setuptools \
	python3-wheel \
	openssh-server \
	openmpi-bin \
	openmpi-doc \
	libopenmpi-dev \
	netbase \
	htop \
    sudo

WORKDIR /home/deploy


RUN pip3 install -U virtualenv

ENV VIRTUAL_ENV=/python_venv
RUN virtualenv --system-site-packages -p python3.6 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN touch /home/deploy/.bashrc \
 && echo "source /python_venv/bin/activate" >> /home/deploy/.bashrc \
 && source /home/deploy/.bashrc

RUN wget https://nextcloud.jblanier.net/s/Z2wYiKksG2xHYkC/download -O pip_reqs.txt
RUN wget https://nextcloud.jblanier.net/s/YQN3njsgJ8i5BLG/download -O ray-0.8.0.dev4actorcreationpatched-cp36-cp36m-manylinux1_x86_64.whl 

RUN pip install cmake \
	&& pip install -r pip_reqs.txt \
	&& pip install mpi4py==3.0.2 psutil==5.6.3 tensorflow-gpu==1.13.1 \
	&& pip install ray-0.8.0.dev4actorcreationpatched-cp36-cp36m-manylinux1_x86_64.whl

RUN rm pip_reqs.txt ray-0.8.0.dev4actorcreationpatched-cp36-cp36m-manylinux1_x86_64.whl

WORKDIR /home/deploy

EXPOSE 7000 
EXPOSE 8080
EXPOSE 19002
EXPOSE 19003

RUN cd /home/deploy \
    && git clone https://github.com/JBLanier/distributed-rl-for-imperfect-info-games \
	&& cd pomprl \
	&& pip install -e .

RUN git clone https://github.com/deepmind/open_spiel \
    && cd open_spiel \
    && git checkout 222ba03f73d643658838d0d95331e9c8a4f77cf1 \
    && ./install.sh \
    && mkdir build \
    && cd build \
    && CXX=g++ cmake -DPython_TARGET_VERSION=3.6 -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel \
    && make -j$(nproc) \
    && echo "export PYTHONPATH=/home/deploy/open_spiel:/home/deploy/open_spiel/build/python:$PYTHONPATH" >> /home/deploy/.bashrc \
    && source /home/deploy/.bashrc

COPY tmux.conf.local /home/deploy/.tmux.conf.local
COPY tmux.conf /home/deploy/.tmux.conf
COPY bash_profile /home/deploy/.bash_profile

RUN mkdir -p /var/run/sshd

RUN adduser deploy && \
    echo "deploy ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/deploy && \
    chmod 0440 /etc/sudoers.d/deploy

RUN chown -R deploy /home/deploy

RUN wget https://dl.min.io/server/minio/release/linux-amd64/minio \
    && chmod +x minio

RUN echo "RAY_HEAD_NODE="120.0.0.1:55751"" >> /home/deploy/.bashrc

ENTRYPOINT bash
