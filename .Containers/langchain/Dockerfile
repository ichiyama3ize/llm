FROM python:3.9

RUN apt-get -y update && \
    apt-get install -y git \
    make

# USER
ARG DOCKER_UID=${DOCKER_UID}
ARG DOCKER_GID=${DOCKER_GID}
ARG DOCKER_USER=${DOCKER_USER}
ARG DOCKER_PASSWORD=${DOCKER_PASSWORD}

RUN useradd -m --uid $DOCKER_UID --groups sudo $DOCKER_USER \
  && echo $DOCKER_USER:$DOCKER_PASSWORD | chpasswd

USER $DOCKER_USER

# pyenv:  for manage py versions

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT /home/$DOCKER_USER/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/bin/shims:$PATH

# poetry: for manage py library versions
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH /home/$DOCKER_USER/.local/bin:$PATH

# python
RUN pip install --upgrade pip

# langchain
# ENV OPENAI_LOG debug

WORKDIR /home/$DOCKER_USER/work