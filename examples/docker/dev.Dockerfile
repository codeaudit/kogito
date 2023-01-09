FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04@sha256:a6831f0d6328ea7301fa196ae2a376d2e67caae384af4ffd93fb196b527c0a0f

ENV HOME=/root
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV KOGITO_DIR=${HOME}/kogito
ENV POETRY=${HOME}/.local/bin/poetry

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install dependencies
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends openssh-server vim wget unzip tmux

WORKDIR ${HOME}

# Cluster setup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O anaconda.sh
RUN bash anaconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN git config --global user.name "Mete Ismayil"
RUN git config --global user.email "mismayilza@gmail.com"
RUN git config pull.rebase false
RUN echo "export LANG=en_US.UTF-8" >> ~/.bashrc

# Setup kogito env
RUN ${CONDA} create --name kogito -y python=3.8
RUN ${CONDA} run -n kogito curl -sSL https://install.python-poetry.org | python3 -
RUN ${CONDA} install -n kogito -y pytorch cudatoolkit=11.5 -c pytorch

ARG GITHUB_PERSONAL_TOKEN

# Clone kogito
RUN git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/epfl-nlp/kogito.git

WORKDIR ${KOGITO_DIR}

# Setup kogito dependencies
RUN ${CONDA} run -n kogito ${POETRY} install
RUN ${CONDA} run -n kogito python -c "import nltk;nltk.download('punkt');nltk.download('wordnet');nltk.download('omw-1.4')"
RUN ${CONDA} run -n kogito python -m spacy download en_core_web_sm

# Install training data
ENV KOGITO_DATA_DIR=${KOGITO_DIR}/data
RUN mkdir ${KOGITO_DATA_DIR}
RUN wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
RUN unzip atomic2020_data-feb2021.zip -d ${KOGITO_DATA_DIR}

# Set up SSH server
RUN apt-get update && apt-get install -y openssh-server tmux vim
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

COPY ./train.py .
COPY ./entrypoint.sh .

ENTRYPOINT ["/usr/sbin/sshd", "-D"]
# ENTRYPOINT ["./entrypoint.sh"]