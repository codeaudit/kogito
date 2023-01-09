FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04@sha256:a6831f0d6328ea7301fa196ae2a376d2e67caae384af4ffd93fb196b527c0a0f

ENV HOME=/root
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV KOGITO_DIR=${HOME}/kogito

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install dependencies
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends openssh-server vim wget unzip tmux

# Set up SSH server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

WORKDIR ${HOME}

# Cluster setup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O anaconda.sh
RUN bash anaconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN echo "export LANG=en_US.UTF-8" >> ~/.bashrc
RUN ${CONDA} create --name kogito -y python=3.10
RUN ${CONDA} install -n kogito ipykernel --force-reinstall

WORKDIR ${KOGITO_DIR}

ARG ENV_TAINT=0

# Setup project dependencies
RUN ${CONDA} run -n kogito pip install kogito --extra-index-url https://download.pytorch.org/whl/cu116
RUN ${CONDA} run -n kogito python -c "import nltk;nltk.download('punkt');nltk.download('wordnet');nltk.download('omw-1.4')"
RUN ${CONDA} run -n kogito python -m spacy download en_core_web_sm

ARG VERSION_TAINT=0

# Setup data
COPY ./data  .
COPY ./train.py .
COPY ./train.sh .

CMD ["/usr/sbin/sshd", "-D"]
# CMD ["./train.sh"]