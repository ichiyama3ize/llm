FROM python:3.9

RUN apt-get -y update && \
    apt-get install -y git \
    make

# pyenv:  for manage py versions
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc &&\
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc &&\ 
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

# poetry: for manage py library versions
RUN curl -sSL https://install.python-poetry.org | python -
RUN echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc &&\
    echo 'alias python="poetry run python"' >> ~/.bashrc

# langchain
# ENV OPENAI_LOG debug

WORKDIR /root/work
